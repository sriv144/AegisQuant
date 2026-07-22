"""One benchmark-relative metrics implementation for v3 research and runtime."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


TRADING_SESSIONS = 252


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def annualized_return(returns: Sequence[float], periods: int = TRADING_SESSIONS) -> float:
    values = _finite(returns)
    if len(values) == 0:
        return 0.0
    if np.any(values <= -1):
        return -1.0
    return float(np.prod(1.0 + values) ** (periods / len(values)) - 1.0)


def annualized_volatility(returns: Sequence[float], periods: int = TRADING_SESSIONS) -> float:
    values = _finite(returns)
    return 0.0 if len(values) < 2 else float(values.std(ddof=1) * math.sqrt(periods))


def sharpe_ratio(returns: Sequence[float], periods: int = TRADING_SESSIONS) -> float:
    values = _finite(returns)
    if len(values) < 2:
        return 0.0
    volatility = float(values.std(ddof=1))
    return 0.0 if volatility <= 0 else float(values.mean() / volatility * math.sqrt(periods))


def max_drawdown_magnitude(returns: Sequence[float]) -> float:
    """Return a positive drawdown magnitude (0.15 means fifteen percent)."""

    values = _finite(returns)
    if len(values) == 0:
        return 0.0
    equity = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdown = 1.0 - equity / peak
    return float(max(0.0, np.max(drawdown)))


def probability_sharpe_ratio(
    returns: Sequence[float],
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probability that the true per-period Sharpe exceeds a threshold."""

    values = _finite(returns)
    if len(values) < 30:
        return 0.0
    sigma = float(values.std(ddof=1))
    if sigma <= 0:
        return 0.0
    observed = float(values.mean() / sigma)
    skew = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    denominator = 1.0 - skew * observed + ((kurtosis - 1.0) / 4.0) * observed**2
    if denominator <= 0:
        return 0.0
    z = (observed - benchmark_sharpe) * math.sqrt(len(values) - 1) / math.sqrt(denominator)
    return float(np.clip(stats.norm.cdf(z), 0.0, 1.0))


def expected_max_sharpe(n_trials: int, sharpe_variance: float = 1.0) -> float:
    """Expected best Sharpe under independent zero-edge trials."""

    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329
    first = stats.norm.ppf(1.0 - 1.0 / n_trials)
    second = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(max(0.0, sharpe_variance)) * ((1.0 - gamma) * first + gamma * second))


def deflated_sharpe_ratio(returns: Sequence[float], n_trials: int) -> float:
    """PSR adjusted for the expected best result among all attempted trials."""

    if n_trials < 1:
        raise ValueError("n_trials must include every attempted trial")
    values = _finite(returns)
    if len(values) < 30:
        return 0.0
    # The null threshold is on the same per-period Sharpe scale used by PSR.
    null_sharpe = expected_max_sharpe(n_trials) / math.sqrt(len(values))
    return probability_sharpe_ratio(values, benchmark_sharpe=null_sharpe)


def probability_backtest_overfit(
    trial_returns: np.ndarray,
    *,
    n_splits: int = 8,
) -> float:
    """Combinatorially symmetric cross-validation PBO foundation.

    Rows are ordered return observations and columns are every attempted
    strategy.  The statistic is the fraction of splits where the in-sample
    winner ranks at or below the out-of-sample median.
    """

    matrix = np.asarray(trial_returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("trial_returns must have shape observations x trials")
    observations, trials = matrix.shape
    if trials < 2:
        return 0.0
    if n_splits < 4 or n_splits % 2:
        raise ValueError("n_splits must be an even integer of at least four")
    if observations < n_splits * 4:
        raise ValueError("at least four observations per split are required")
    if not np.isfinite(matrix).all():
        raise ValueError("trial_returns cannot contain missing or infinite values")

    chunks = np.array_split(np.arange(observations), n_splits)
    below_median = 0
    comparisons = 0
    all_chunks = set(range(n_splits))
    for in_sample_chunks in itertools.combinations(range(n_splits), n_splits // 2):
        out_sample_chunks = sorted(all_chunks.difference(in_sample_chunks))
        in_sample = matrix[np.concatenate([chunks[index] for index in in_sample_chunks])]
        out_sample = matrix[np.concatenate([chunks[index] for index in out_sample_chunks])]
        in_sharpes = np.array([sharpe_ratio(in_sample[:, i], periods=1) for i in range(trials)])
        out_sharpes = np.array([sharpe_ratio(out_sample[:, i], periods=1) for i in range(trials)])
        winner = int(np.argmax(in_sharpes))
        oos_rank = float(stats.rankdata(out_sharpes, method="average")[winner])
        below_median += int(oos_rank <= (trials + 1) / 2.0)
        comparisons += 1
    return float(below_median / comparisons)


@dataclass(frozen=True, slots=True)
class SpyRelativeMetrics:
    observations: int
    portfolio_annualized_return: float
    spy_annualized_return: float
    net_annualized_excess_return: float
    portfolio_annualized_volatility: float
    tracking_error: float
    information_ratio: float
    beta: float
    portfolio_max_drawdown: float
    spy_max_drawdown: float
    positive_rolling_12m_fraction: float
    psr: float
    dsr: float
    pbo: float | None


def compute_spy_relative_metrics(
    portfolio_returns: Sequence[float] | pd.Series,
    spy_returns: Sequence[float] | pd.Series,
    *,
    n_trials: int = 1,
    trial_returns: np.ndarray | None = None,
    pbo_splits: int = 8,
) -> SpyRelativeMetrics:
    """Compute promotion metrics from aligned daily total returns."""

    if n_trials < 1:
        raise ValueError("n_trials must include every attempted trial")
    if trial_returns is not None:
        trial_matrix = np.asarray(trial_returns)
        if trial_matrix.ndim != 2:
            raise ValueError("trial_returns must have shape observations x trials")
        if n_trials < trial_matrix.shape[1]:
            raise ValueError("n_trials cannot omit attempted strategies in trial_returns")

    if isinstance(portfolio_returns, pd.Series) and isinstance(spy_returns, pd.Series):
        aligned = pd.concat(
            [portfolio_returns.rename("portfolio"), spy_returns.rename("spy")], axis=1, join="inner"
        ).dropna()
        portfolio = aligned["portfolio"].to_numpy(dtype=float)
        spy = aligned["spy"].to_numpy(dtype=float)
    else:
        portfolio = np.asarray(portfolio_returns, dtype=float).reshape(-1)
        spy = np.asarray(spy_returns, dtype=float).reshape(-1)
        if len(portfolio) != len(spy):
            raise ValueError("portfolio and SPY returns must have equal length")
        mask = np.isfinite(portfolio) & np.isfinite(spy)
        portfolio, spy = portfolio[mask], spy[mask]
    if len(portfolio) < 2:
        raise ValueError("at least two aligned returns are required")

    excess = portfolio - spy
    tracking_error = annualized_volatility(excess)
    information_ratio = 0.0 if tracking_error <= 0 else float(excess.mean() * TRADING_SESSIONS / tracking_error)
    spy_variance = float(np.var(spy, ddof=1))
    beta = 0.0 if spy_variance <= 0 else float(np.cov(portfolio, spy, ddof=1)[0, 1] / spy_variance)
    if len(excess) >= TRADING_SESSIONS:
        portfolio_rolling = pd.Series(1.0 + portfolio).rolling(TRADING_SESSIONS).apply(np.prod, raw=True)
        spy_rolling = pd.Series(1.0 + spy).rolling(TRADING_SESSIONS).apply(np.prod, raw=True)
        rolling_excess = (portfolio_rolling - spy_rolling).dropna()
        positive_fraction = float((rolling_excess > 0).mean()) if len(rolling_excess) else 0.0
    else:
        positive_fraction = 0.0

    portfolio_ann = annualized_return(portfolio)
    spy_ann = annualized_return(spy)
    pbo = None if trial_returns is None else probability_backtest_overfit(trial_returns, n_splits=pbo_splits)
    return SpyRelativeMetrics(
        observations=len(portfolio),
        portfolio_annualized_return=portfolio_ann,
        spy_annualized_return=spy_ann,
        net_annualized_excess_return=portfolio_ann - spy_ann,
        portfolio_annualized_volatility=annualized_volatility(portfolio),
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        beta=beta,
        portfolio_max_drawdown=max_drawdown_magnitude(portfolio),
        spy_max_drawdown=max_drawdown_magnitude(spy),
        positive_rolling_12m_fraction=positive_fraction,
        psr=probability_sharpe_ratio(excess),
        dsr=deflated_sharpe_ratio(excess, n_trials=n_trials),
        pbo=pbo,
    )
