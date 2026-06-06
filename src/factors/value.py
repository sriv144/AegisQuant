"""
Value Factor — AQR / Asness style.

Composite of four standard value metrics, each z-scored cross-sectionally
then averaged. Tickers missing 2+ metrics are dropped.

Metrics (all "higher = cheaper = better" after sign flip where needed):
  - 1/trailingPE (earnings yield)
  - 1/priceToBook (book yield)
  - 1/enterpriseToEbitda (EBITDA yield)
  - freeCashflowYield (already a yield)

Negative earnings (negative PE → negative earnings yield) are kept as-is but
get a confidence penalty since value at a loss is ambiguous (could be a turn-
around or a value trap — the agent layer can disambiguate).

Reference: Asness, Frazzini, Pedersen (2014) "Quality Minus Junk" supplement;
Alpha Architect "Quantitative Value" by Gray & Carlisle.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.factors.base import Factor, FactorResult


class ValueFactor(Factor):
    name = "value"
    rebalance_freq = "monthly"
    requires = ["fundamentals"]

    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        as_of = as_of or datetime.utcnow()
        raw_metrics: Dict[str, Dict[str, float]] = {}

        for t in universe:
            f = self.dp.get_fundamentals(t)
            if not f:
                continue

            pe = f.get("trailingPE")
            pb = f.get("priceToBook")
            ev_ebitda = f.get("enterpriseToEbitda")
            fcf_y = f.get("freeCashflowYield")

            # Convert to yields (higher = cheaper)
            metrics = {}
            if pe and pe != 0:
                metrics["earnings_yield"] = 1.0 / pe   # negative if loss-making
            if pb and pb > 0:
                metrics["book_yield"] = 1.0 / pb
            if ev_ebitda and ev_ebitda != 0:
                metrics["ebitda_yield"] = 1.0 / ev_ebitda
            if fcf_y is not None:
                metrics["fcf_yield"] = float(fcf_y)

            if metrics:
                raw_metrics[t] = metrics

        # Z-score each metric independently across the universe
        z_by_metric: Dict[str, Dict[str, float]] = {}
        for metric in ["earnings_yield", "book_yield", "ebitda_yield", "fcf_yield"]:
            vals = {t: m[metric] for t, m in raw_metrics.items() if metric in m}
            z_by_metric[metric] = self.zscore(vals, winsorize=3.0)

        # Composite = mean of available z-scores per ticker
        scores: Dict[str, float] = {}
        confidence: Dict[str, float] = {}
        for t, metrics in raw_metrics.items():
            present_zs = [z_by_metric[m].get(t) for m in metrics if t in z_by_metric.get(m, {})]
            present_zs = [z for z in present_zs if z is not None]
            if not present_zs:
                continue
            scores[t] = float(np.mean(present_zs))
            # Confidence scales with coverage: 4/4 metrics = 1.0, 2/4 = 0.5
            confidence[t] = min(1.0, len(present_zs) / 4.0)

            # Penalize negative earnings yield (value trap risk)
            if metrics.get("earnings_yield", 0) < 0:
                confidence[t] *= 0.6

        return FactorResult(
            factor_name=self.name,
            as_of=as_of,
            scores=scores,
            confidence=confidence,
            raw=raw_metrics,
            notes=f"Value composite of {len(raw_metrics)} tickers, "
                  f"avg metrics/ticker = {np.mean([len(m) for m in raw_metrics.values()]):.1f}"
                  if raw_metrics else "no data",
        )
