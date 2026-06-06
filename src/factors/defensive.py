"""
Defensive Factor — Frazzini & Pedersen "Betting Against Beta" (2014).

Theory: leverage-constrained investors overpay for high-beta stocks, so low-beta
stocks earn positive risk-adjusted returns. Score = -beta (higher = more
defensive = preferred).

Implementation:
  - 252-day rolling beta vs SPY (US) using daily returns.
  - Cross-sectional z-score of negative beta.

Confidence drops for tickers with <100 days of overlapping data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorResult

logger = logging.getLogger(__name__)


class DefensiveFactor(Factor):
    name = "defensive"
    rebalance_freq = "monthly"
    requires = ["prices"]

    BENCHMARK = "SPY"
    BETA_WINDOW = 252
    MIN_OVERLAP = 150

    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        as_of = pd.Timestamp(as_of or datetime.utcnow())
        start = (as_of - timedelta(days=420)).strftime("%Y-%m-%d")
        end = as_of.strftime("%Y-%m-%d")

        tickers_plus_bench = sorted(set(universe) | {self.BENCHMARK})
        prices = self.dp.get_prices(tickers_plus_bench, start=start, end=end)
        if prices is None or prices.empty or self.BENCHMARK not in prices.columns:
            logger.warning("DefensiveFactor: missing SPY benchmark")
            return FactorResult(self.name, as_of, {}, {}, {}, notes="no benchmark data")

        bench_ret = prices[self.BENCHMARK].pct_change().dropna()
        bench_ret = bench_ret.iloc[-self.BETA_WINDOW:] if len(bench_ret) > self.BETA_WINDOW else bench_ret

        betas: Dict[str, float] = {}
        raw: Dict[str, Dict[str, float]] = {}
        confidence: Dict[str, float] = {}

        for t in universe:
            if t == self.BENCHMARK or t not in prices.columns:
                continue
            ret = prices[t].pct_change().dropna()
            joined = pd.concat([ret, bench_ret], axis=1, join="inner").dropna()
            if len(joined) < self.MIN_OVERLAP:
                continue
            x = joined.iloc[:, 1].values   # bench
            y = joined.iloc[:, 0].values   # ticker
            cov = float(np.cov(y, x, ddof=0)[0, 1])
            var = float(np.var(x, ddof=0))
            if var == 0:
                continue
            beta = cov / var
            betas[t] = beta
            raw[t] = {"beta": beta, "n_days": len(joined)}
            confidence[t] = min(1.0, len(joined) / self.BETA_WINDOW)

        # Score = negative beta (low beta → high score)
        neg_beta = {t: -b for t, b in betas.items()}
        scores = self.zscore(neg_beta, winsorize=3.0)

        return FactorResult(
            factor_name=self.name,
            as_of=as_of,
            scores=scores,
            confidence=confidence,
            raw=raw,
            notes=f"Defensive (BAB) on {len(scores)} tickers, bench={self.BENCHMARK}",
        )
