"""
Performance metrics for backtesting.

Includes: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor,
Deflated Sharpe Ratio (Lopez de Prado 2018), and t-test on OOS returns.
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, Any


TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe Ratio."""
    excess = returns - risk_free / TRADING_DAYS
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sortino Ratio (downside deviation only)."""
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return float((excess.mean() / downside_std) * math.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (negative number, e.g. -0.25 = 25% DD)."""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def calmar_ratio(returns: np.ndarray) -> float:
    """Annualised Return / |Max Drawdown|."""
    ann_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("inf")
    return float(ann_return / mdd)


def win_rate(returns: np.ndarray) -> float:
    """Fraction of positive return days."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def profit_factor(returns: np.ndarray) -> float:
    """Sum of gains / sum of losses (|losses|). > 1 means profitable."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def annualised_return(returns: np.ndarray) -> float:
    """Compound annualised return."""
    if len(returns) == 0:
        return 0.0
    return float((1 + returns.mean()) ** TRADING_DAYS - 1)


def annualised_volatility(returns: np.ndarray) -> float:
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def avg_drawdown_duration(returns: np.ndarray) -> float:
    """Average number of days spent in a drawdown."""
    cumulative = pd.Series(np.cumprod(1 + returns))
    peak = cumulative.cummax()
    in_drawdown = (cumulative < peak).astype(int)
    # Count consecutive drawdown days
    durations = []
    count = 0
    for v in in_drawdown:
        if v:
            count += 1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


def annualised_turnover(weights: np.ndarray) -> float:
    """Annualised turnover from a series of target weights."""
    if len(weights) < 2:
        return 0.0
    daily_turnover = np.abs(np.diff(weights)).mean()
    return float(daily_turnover * TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance
# ─────────────────────────────────────────────────────────────────────────────

def ttest_returns(returns: np.ndarray) -> Dict[str, float]:
    """One-sample t-test: H0: mean return = 0."""
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    T: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio (Lopez de Prado, 2018).

    Corrects for selection bias when the reported Sharpe is the best of
    `n_trials` independent experiments.

    Args:
        sharpe:           Reported (best) Sharpe Ratio.
        n_trials:         Number of strategy/hyperparameter trials tested.
        T:                Number of OOS return observations.
        skewness:         Skewness of the return distribution.
        excess_kurtosis:  Excess kurtosis of the return distribution.

    Returns:
        DSR in [0, 1] — the probability that the true Sharpe > 0.
    """
    from scipy.stats import norm

    if n_trials <= 1 or T <= 1:
        return float(norm.cdf(sharpe * math.sqrt(T)))

    EULER_GAMMA = 0.5772156649  # Euler-Mascheroni constant

    # Expected maximum Sharpe across n_trials under H0
    e_max_sr = (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
        + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )

    # Variance of the Sharpe estimator
    sr_var = (
        (1 - skewness * sharpe + (excess_kurtosis - 1) / 4 * sharpe**2)
        / (T - 1)
    )
    sr_std = math.sqrt(max(sr_var, 1e-12))

    z = (sharpe - e_max_sr) / sr_std
    return float(norm.cdf(z))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute all metrics at once
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    returns: np.ndarray,
    weights: np.ndarray | None = None,
    n_trials: int = 1,
    label: str = "",
) -> Dict[str, Any]:
    """
    Returns a dict with every performance metric for a returns series.

    Args:
        returns:   1-D array of daily (or per-step) returns.
        weights:   Optional array of target weights (for turnover).
        n_trials:  How many hyperparameter combos were tried (for DSR).
        label:     Optional label to include in the output dict.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return {}

    sr = sharpe_ratio(returns)
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis())
    ttest = ttest_returns(returns)

    result: Dict[str, Any] = {
        "label": label,
        "n_observations": len(returns),
        "annualised_return": round(annualised_return(returns), 4),
        "annualised_volatility": round(annualised_volatility(returns), 4),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(sortino_ratio(returns), 4),
        "calmar_ratio": round(calmar_ratio(returns), 4),
        "max_drawdown": round(max_drawdown(returns), 4),
        "avg_drawdown_duration_days": round(avg_drawdown_duration(returns), 1),
        "win_rate": round(win_rate(returns), 4),
        "profit_factor": round(profit_factor(returns), 4),
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "t_stat": round(ttest["t_stat"], 4),
        "p_value": round(ttest["p_value"], 4),
        "deflated_sharpe_ratio": round(
            deflated_sharpe_ratio(sr, n_trials, len(returns), skew, kurt), 4
        ),
    }

    if weights is not None:
        result["annualised_turnover"] = round(annualised_turnover(np.asarray(weights)), 4)

    return result
