"""
Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Headline Sharpe ratios from backtests are biased upward by:

  1. **Selection bias under multiple testing** — if you tried 200 hyperparameter
     configurations and report the best one, even pure noise produces a high
     Sharpe by chance.
  2. **Non-Normal returns** — Sharpe assumes Gaussian; fat tails / skew break it.

The Deflated Sharpe Ratio (DSR) corrects for both. It returns the
probability that the observed Sharpe came from an actual edge (vs. noise +
selection bias). The plan's gate is DSR > 0.4 (i.e. >40% probability of true
edge after correcting for everything).

References:
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
  - AFML Ch. 14
"""
from __future__ import annotations

from math import sqrt, log, isfinite
from typing import Sequence

import numpy as np
from scipy import stats


def _emc() -> float:
    """Euler-Mascheroni constant."""
    return 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sharpe: float = 1.0) -> float:
    """
    Expected maximum Sharpe across `n_trials` random strategies, assuming
    each individual Sharpe is iid N(0, var_sharpe). Used as the null
    benchmark for the deflated Sharpe.

    Bailey-López de Prado equation 6.
    """
    if n_trials < 2:
        return 0.0
    sd = sqrt(var_sharpe)
    inv_n = 1.0 / n_trials
    z1 = stats.norm.ppf(1.0 - inv_n)
    z2 = stats.norm.ppf(1.0 - inv_n * np.e ** -1)
    emax = (1.0 - _emc()) * z1 + _emc() * z2
    return sd * emax


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int = 1,
    annualization_factor: int = 252,
) -> float:
    """
    Returns the probability (in [0, 1]) that the observed in-sample Sharpe
    reflects a TRUE edge, accounting for:
      - sample size (T)
      - non-Normality (skew, kurtosis) of returns
      - multiple-testing selection bias (n_trials)

    A DSR of 0.95 = 95% confidence the edge is real. The plan uses 0.4 as a
    looser pragmatic gate ("at least 40% chance the edge is real").

    Parameters
    ----------
    returns : sequence of per-period returns (e.g. daily portfolio returns)
    n_trials : number of distinct backtest configurations searched
        (e.g. if you tried 5 lookback windows × 3 thresholds, n_trials=15)
    annualization_factor : 252 for daily, 12 for monthly, etc.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return 0.0   # too short for any inference
    T = len(r)

    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma <= 0:
        return 0.0
    sr = mu / sigma  # per-period SR
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))  # excess kurtosis (Normal = 0)

    # Expected max Sharpe under null (no edge), per Bailey & López de Prado eq. 6
    sr0 = expected_max_sharpe(max(n_trials, 1))

    # DSR formula (Bailey-LdP eq. 9)
    # Variance of estimated SR under non-Normality
    var_sr = (1.0 - skew * sr + (kurt / 4.0) * sr ** 2) / (T - 1)
    if var_sr <= 0:
        return 0.0
    se_sr = sqrt(var_sr)

    z = (sr - sr0) / se_sr
    p = float(stats.norm.cdf(z))
    return max(0.0, min(1.0, p))


def probability_backtest_overfit(
    sharpe_grid: np.ndarray,
    n_splits: int = 16,
) -> float:
    """
    Probability of Backtest Overfitting (PBO) — López de Prado et al. (2013).

    Takes a matrix of per-period returns for N candidate strategies (cols)
    over T periods (rows), splits time into S sub-periods, computes the
    in-sample/out-of-sample rank correlation under all combinations, and
    returns the fraction where the IS-best strategy is OOS-below-median.

    A PBO of 0.5 = the best IS strategy is equally likely to be a coin flip
    OOS. PBO < 0.2 is a reasonable threshold.

    Parameters
    ----------
    sharpe_grid : ndarray of shape (T, N)
        Per-period returns for N candidate strategies.
    n_splits : even integer
        Number of sub-periods (16 is conventional). T must be >> n_splits.
    """
    from itertools import combinations
    M = np.asarray(sharpe_grid, dtype=float)
    if M.ndim != 2:
        raise ValueError("sharpe_grid must be 2-D (T x N)")
    T, N = M.shape
    if N < 2:
        return 0.0
    if T < n_splits * 4:
        return 0.0  # not enough data

    # Split T into n_splits roughly-equal chunks
    chunk = T // n_splits
    M_trim = M[: chunk * n_splits]
    submats = M_trim.reshape(n_splits, chunk, N)   # (S, t, N)

    # Per-submat per-strategy Sharpe
    means = submats.mean(axis=1)      # (S, N)
    stds = submats.std(axis=1, ddof=1)   # (S, N)
    sharpe_per_split = np.where(stds > 0, means / stds, 0.0)   # (S, N)

    # Enumerate all (S choose S/2) IS/OOS splits
    overfit_count = 0
    total = 0
    indices = list(range(n_splits))
    half = n_splits // 2
    for is_subset in combinations(indices, half):
        is_set = list(is_subset)
        oos_set = [i for i in indices if i not in is_set]

        is_sharpe = sharpe_per_split[is_set].mean(axis=0)    # (N,)
        oos_sharpe = sharpe_per_split[oos_set].mean(axis=0)  # (N,)

        best_is = int(np.argmax(is_sharpe))
        oos_rank = (oos_sharpe < oos_sharpe[best_is]).sum() + 1  # 1-based rank
        oos_quantile = oos_rank / N
        if oos_quantile <= 0.5:
            overfit_count += 1
        total += 1

    if total == 0:
        return 0.0
    return overfit_count / total
