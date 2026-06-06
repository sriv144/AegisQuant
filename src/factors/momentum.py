"""
Momentum Factor — Wes Gray (Alpha Architect) "Quantitative Momentum" style.

Two enhancements over vanilla 12-1 momentum:

1. **12-1 month return** (skip last month) — Jegadeesh & Titman (1993) baseline.
   The skip-1-month accounts for short-term mean reversion (Lo & MacKinlay).

2. **Smoothness filter** (Gray & Vogel) — among the high-momentum names, prefer
   those with *smooth* paths. Smoothness = % of trading days with positive return
   over the lookback. The intuition: a stock that climbed steadily reflects
   sustained buyer demand; a stock that climbed in 2 huge gaps may have been a
   one-off news event. Smooth-momentum dominates spiky-momentum in OOS tests
   ("frog in the pan" — Da, Gurun, Warachka 2014).

Composite score = z(12_1_return) + z(smoothness) — both equally weighted.
Reference: Wes Gray & Jack Vogel, "Quantitative Momentum" (2016).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorResult

logger = logging.getLogger(__name__)


class MomentumFactor(Factor):
    name = "momentum"
    rebalance_freq = "monthly"
    requires = ["prices"]

    # Lookback windows in trading days (≈ 21/month)
    LOOKBACK_DAYS = 252       # 12 months
    SKIP_DAYS = 21            # skip last month
    MIN_DAYS_REQUIRED = 200   # need at least ~9.5 months of data

    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        as_of = pd.Timestamp(as_of or datetime.utcnow())
        # Pull ~14 months of data so we can compute 12-1 return cleanly
        start = (as_of - timedelta(days=420)).strftime("%Y-%m-%d")
        end = as_of.strftime("%Y-%m-%d")
        prices = self.dp.get_prices(universe, start=start, end=end)

        if prices is None or prices.empty:
            logger.warning("MomentumFactor: empty price data")
            return FactorResult(self.name, as_of, {}, {}, {}, notes="no price data")

        ret_12_1: Dict[str, float] = {}
        smoothness: Dict[str, float] = {}
        raw_metrics: Dict[str, Dict[str, float]] = {}

        for t in universe:
            if t not in prices.columns:
                continue
            series = prices[t].dropna()
            if len(series) < self.MIN_DAYS_REQUIRED:
                continue

            # 12-1 return: from 12 months ago to 1 month ago
            try:
                p_lookback = series.iloc[-self.LOOKBACK_DAYS]
                p_skip = series.iloc[-self.SKIP_DAYS]
            except IndexError:
                continue
            if p_lookback <= 0:
                continue
            r_12_1 = float(p_skip / p_lookback - 1.0)

            # Smoothness: fraction of positive-return days over the 12-1 window
            window = series.iloc[-self.LOOKBACK_DAYS:-self.SKIP_DAYS]
            daily_ret = window.pct_change().dropna()
            if len(daily_ret) < 100:
                continue
            smooth = float((daily_ret > 0).mean())   # 0.45–0.55 typical

            ret_12_1[t] = r_12_1
            smoothness[t] = smooth
            raw_metrics[t] = {"ret_12_1": r_12_1, "smoothness": smooth}

        # Z-score each and average
        z_ret = self.zscore(ret_12_1, winsorize=3.0)
        z_smooth = self.zscore(smoothness, winsorize=3.0)

        scores: Dict[str, float] = {}
        confidence: Dict[str, float] = {}
        for t in ret_12_1:
            zr = z_ret.get(t)
            zs = z_smooth.get(t)
            if zr is None and zs is None:
                continue
            parts = [z for z in (zr, zs) if z is not None]
            scores[t] = float(np.mean(parts))
            confidence[t] = 1.0 if len(parts) == 2 else 0.6

        return FactorResult(
            factor_name=self.name,
            as_of=as_of,
            scores=scores,
            confidence=confidence,
            raw=raw_metrics,
            notes=f"Momentum (12-1m + Gray smoothness) on {len(scores)} tickers",
        )
