"""
Factor base class and result schema.

Design notes
------------
- Every factor returns a FactorResult with z-scored cross-sectional scores.
  Sign convention: higher score = more attractive (we go long the top decile).
- Confidence is a separate dimension. A factor may have a strong score with low
  confidence (e.g. extreme P/B but stale fundamentals) and should be discounted
  by the sleeve before sizing.
- Factors are stateless besides the DataProvider they hold. All temporal logic
  lives in compute(as_of). This makes purged k-fold CV trivial.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class FactorResult:
    """Output of a Factor.compute() call."""
    factor_name: str
    as_of: pd.Timestamp
    scores: Dict[str, float]                     # ticker -> cross-sectional z-score
    confidence: Dict[str, float] = field(default_factory=dict)  # ticker -> 0..1
    raw: Dict[str, Dict[str, float]] = field(default_factory=dict)  # ticker -> raw metric values
    notes: str = ""

    def top_n(self, n: int = 10) -> List[str]:
        """Return the top N tickers by score."""
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return [t for t, _ in ranked[:n]]

    def bottom_n(self, n: int = 10) -> List[str]:
        """Return the bottom N tickers by score."""
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1])
        return [t for t, _ in ranked[:n]]

    def to_frame(self) -> pd.DataFrame:
        """Return as a tidy DataFrame: index=ticker, cols=score, confidence, *raw_metrics."""
        rows = []
        for t, s in self.scores.items():
            row = {"ticker": t, "score": s, "confidence": self.confidence.get(t, 1.0)}
            row.update(self.raw.get(t, {}))
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("ticker").sort_values("score", ascending=False)
        return df


class Factor(ABC):
    """
    Base class for cross-sectional factors.

    Subclasses must implement compute(). They should NOT cache results
    internally — the DataProvider handles all data caching.
    """

    name: str = "unnamed"
    rebalance_freq: str = "monthly"      # 'daily', 'weekly', 'monthly', 'event'
    requires: List[str] = []             # data deps, e.g. ['prices', 'fundamentals']

    def __init__(self, data_provider=None):
        from src.factors.data_provider import get_data_provider
        self.dp = data_provider or get_data_provider()

    @abstractmethod
    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        """
        Compute the factor for the given universe at the given point-in-time.
        as_of=None means "now". Backtests pass historical timestamps.
        """
        ...

    # ── shared helpers ──────────────────────────────────────────────────────

    @staticmethod
    def zscore(values: Dict[str, float], winsorize: float = 3.0) -> Dict[str, float]:
        """
        Cross-sectional z-score with optional winsorization.

        Args:
            values: ticker -> raw metric. NaN-valued tickers are dropped.
            winsorize: clip z-scores to ±this many std devs (3.0 is conventional).

        Returns:
            ticker -> z-score (only for tickers with finite values).
        """
        clean = {t: v for t, v in values.items() if v is not None and np.isfinite(v)}
        if len(clean) < 5:
            # Not enough cross-section to be meaningful
            return {t: 0.0 for t in clean}

        arr = np.array(list(clean.values()), dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=0))
        if sd == 0.0:
            return {t: 0.0 for t in clean}

        out = {}
        for t, v in clean.items():
            z = (v - mu) / sd
            if winsorize:
                z = max(-winsorize, min(winsorize, z))
            out[t] = float(z)
        return out

    @staticmethod
    def rank_pct(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
        """
        Convert raw metric values to cross-sectional percentile rank in [0, 1].
        Useful when distributions are non-normal (e.g. fundamentals with outliers).
        """
        clean = {t: v for t, v in values.items() if v is not None and np.isfinite(v)}
        if not clean:
            return {}
        series = pd.Series(clean).rank(pct=True, ascending=higher_is_better)
        return series.to_dict()

    def __repr__(self):
        return f"<Factor name={self.name!r} freq={self.rebalance_freq}>"
