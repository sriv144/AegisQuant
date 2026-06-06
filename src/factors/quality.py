"""
Quality Factor — Asness/Frazzini/Pedersen "Quality Minus Junk" style.

Composite of profitability, safety, and growth metrics. Each z-scored
cross-sectionally; final score = weighted average.

Metrics:
  Profitability: returnOnEquity, returnOnAssets, profitMargins, operatingMargins
  Safety:        1/debtToEquity (inverted — low debt = safe), currentRatio
  Growth:        earningsGrowth, revenueGrowth

Reference: AQR (2018) "Quality Minus Junk", Journal of Financial Economics.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.factors.base import Factor, FactorResult


class QualityFactor(Factor):
    name = "quality"
    rebalance_freq = "monthly"
    requires = ["fundamentals"]

    PROFITABILITY = ["returnOnEquity", "returnOnAssets", "profitMargins", "operatingMargins"]
    SAFETY_NORMAL = ["currentRatio"]
    SAFETY_INVERTED = ["debtToEquity"]   # higher D/E = worse → invert
    GROWTH = ["earningsGrowth", "revenueGrowth"]

    # Composite weights (sum to 1.0). QMJ paper uses roughly equal thirds.
    WEIGHTS = {"profitability": 0.45, "safety": 0.30, "growth": 0.25}

    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        as_of = as_of or datetime.utcnow()
        raw_metrics: Dict[str, Dict[str, float]] = {}

        for t in universe:
            f = self.dp.get_fundamentals(t)
            if not f:
                continue

            m: Dict[str, float] = {}
            for k in self.PROFITABILITY + self.SAFETY_NORMAL + self.GROWTH:
                v = f.get(k)
                if v is not None and np.isfinite(v):
                    m[k] = float(v)
            for k in self.SAFETY_INVERTED:
                v = f.get(k)
                if v is not None and v > 0 and np.isfinite(v):
                    m[k] = -float(v)  # negate so higher = better
            if m:
                raw_metrics[t] = m

        # Z-score per metric, build sleeve composites
        def composite(group_keys: List[str]) -> Dict[str, float]:
            per_metric = {}
            for k in group_keys:
                vals = {t: rm[k] for t, rm in raw_metrics.items() if k in rm}
                per_metric[k] = self.zscore(vals, winsorize=3.0)
            out = {}
            for t in raw_metrics:
                zs = [per_metric[k].get(t) for k in group_keys if t in per_metric.get(k, {})]
                zs = [z for z in zs if z is not None]
                if zs:
                    out[t] = float(np.mean(zs))
            return out

        prof_z = composite(self.PROFITABILITY)
        safe_z = composite(self.SAFETY_NORMAL + self.SAFETY_INVERTED)
        grow_z = composite(self.GROWTH)

        scores: Dict[str, float] = {}
        confidence: Dict[str, float] = {}
        for t in raw_metrics:
            parts, weights = [], []
            if t in prof_z:
                parts.append(prof_z[t]); weights.append(self.WEIGHTS["profitability"])
            if t in safe_z:
                parts.append(safe_z[t]); weights.append(self.WEIGHTS["safety"])
            if t in grow_z:
                parts.append(grow_z[t]); weights.append(self.WEIGHTS["growth"])
            if not parts:
                continue
            scores[t] = float(np.average(parts, weights=weights))
            # Confidence = sum of weights present (1.0 if all 3 groups available)
            confidence[t] = float(sum(weights))

        return FactorResult(
            factor_name=self.name,
            as_of=as_of,
            scores=scores,
            confidence=confidence,
            raw=raw_metrics,
            notes=f"Quality (QMJ) on {len(scores)} tickers",
        )
