"""
Triple-Barrier Labeling
=======================

López de Prado (AFML Ch. 3). For each entry timestamp, label the outcome by
which of three barriers is hit first:

  - Upper (profit target):  price * (1 + pt_mult * sigma)
  - Lower (stop loss):      price * (1 - sl_mult * sigma)
  - Vertical (time):        max_hold_days bars forward

The label is +1 if upper hit first, -1 if lower hit first, 0 if vertical hit
first (timeout). `sigma` is a per-entry volatility estimate (typically EWMA
of returns, scaled to bar frequency).

This produces a clean SUPERVISED LEARNING target that:
  - Doesn't bias toward look-ahead (no next-bar return label)
  - Respects the actual asymmetric stop/target structure of trading
  - Allows meta-labeling (Ch. 3.4): a 2nd model that decides whether to ACT
    on the primary model's signal, raising precision without changing recall

Usage:
    labels = triple_barrier_labels(prices, entry_dates, sigma_series,
                                    pt_mult=2.0, sl_mult=1.0, max_hold=60)
    # labels: pd.Series indexed by entry_dates with values in {-1, 0, +1}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BarrierEvent:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    label: int           # +1 / 0 / -1
    return_pct: float    # actual realized return at exit
    barrier_hit: str     # 'upper' / 'lower' / 'vertical'


def triple_barrier_labels(
    prices: pd.Series,
    entry_dates: pd.DatetimeIndex,
    sigma: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 60,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for a series of entry dates.

    Parameters
    ----------
    prices : pd.Series
        Adjusted-close prices indexed by trading dates.
    entry_dates : pd.DatetimeIndex
        Subset of prices.index where we entered positions.
    sigma : pd.Series
        Per-date volatility estimate (same index as prices). Use EWMA std of
        returns scaled to the bar frequency, e.g.
            sigma = prices.pct_change().ewm(span=20).std()
    pt_mult, sl_mult : float
        Profit-target and stop-loss multipliers in units of sigma.
        Conventional: pt=2.0, sl=1.0 (asymmetric: let winners run).
    max_hold : int
        Vertical barrier — maximum bars to hold before timing out.

    Returns
    -------
    pd.DataFrame indexed by entry_date with columns:
        exit_date, label, return_pct, barrier_hit
    """
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices.index must be monotonically increasing")
    events = []
    p_arr = prices.values
    p_idx = prices.index

    for entry in entry_dates:
        if entry not in p_idx:
            continue
        i0 = p_idx.get_loc(entry)
        entry_price = p_arr[i0]
        if entry_price <= 0:
            continue
        sig = sigma.get(entry, np.nan)
        if not np.isfinite(sig) or sig <= 0:
            continue

        upper = entry_price * (1.0 + pt_mult * sig)
        lower = entry_price * (1.0 - sl_mult * sig)
        i_max = min(i0 + max_hold, len(p_arr) - 1)

        # Scan forward bar by bar
        hit = None
        exit_i = i_max
        for j in range(i0 + 1, i_max + 1):
            p = p_arr[j]
            if p >= upper:
                hit, exit_i = "upper", j
                break
            if p <= lower:
                hit, exit_i = "lower", j
                break
        if hit is None:
            hit = "vertical"
            exit_i = i_max

        exit_price = p_arr[exit_i]
        ret = (exit_price / entry_price) - 1.0
        label = {"upper": 1, "lower": -1, "vertical": 0}[hit]
        events.append({
            "entry_date": entry,
            "exit_date": p_idx[exit_i],
            "label": label,
            "return_pct": float(ret),
            "barrier_hit": hit,
        })

    if not events:
        return pd.DataFrame(columns=["entry_date", "exit_date", "label", "return_pct", "barrier_hit"])
    df = pd.DataFrame(events).set_index("entry_date")
    return df
