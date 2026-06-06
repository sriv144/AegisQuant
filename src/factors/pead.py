"""
Post-Earnings Announcement Drift (PEAD) Factor.

Theory: stocks that beat earnings expectations drift higher (and missers drift
lower) for ~60 trading days post-announcement. Documented since Bernard & Thomas
(1989) and replicated extensively. Sharpe ~0.5 net at retail size.

Signal construction:
  - For each ticker with an earnings event in the last `LOOKBACK_DAYS`:
    * SUE = standardised earnings surprise (yfinance surprise_pct as a proxy)
    * Days since earnings (preference for fresh events)
  - Score = z(SUE) * decay(days_since)
  - decay = max(0, (HOLD_DAYS - days_since) / HOLD_DAYS)  — linear decay over 60d

Tickers with no recent earnings get score 0 / confidence 0.

Reference:
  - Bernard & Thomas (1989) "Post-Earnings-Announcement Drift"
  - QuantPedia summary: https://quantpedia.com/strategies/post-earnings-announcement-effect
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorResult

logger = logging.getLogger(__name__)


class PEADFactor(Factor):
    name = "pead"
    rebalance_freq = "weekly"
    requires = ["earnings"]

    HOLD_DAYS = 60          # drift window
    LOOKBACK_DAYS = 75      # how far back to look for earnings events
    MIN_ABS_SURPRISE = 1.0  # ignore tiny surprises (<1% — noise)

    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        as_of = pd.Timestamp(as_of or datetime.utcnow())
        if as_of.tz is not None:
            as_of = as_of.tz_localize(None)
        sue_signed: Dict[str, float] = {}     # SUE × decay
        raw: Dict[str, Dict[str, float]] = {}
        confidence: Dict[str, float] = {}

        for t in universe:
            df = self.dp.get_earnings(t)
            if df is None or df.empty:
                continue

            # Normalize date column
            if "date" not in df.columns:
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
            df = df.dropna(subset=["date"]).sort_values("date", ascending=False)

            # Find the most recent earnings event with a real surprise
            past = df[df["date"] <= as_of]
            if past.empty:
                continue
            most_recent = past.iloc[0]
            days_since = float((as_of - most_recent["date"]).days)
            if days_since > self.LOOKBACK_DAYS or days_since < 0:
                continue

            surprise = most_recent.get("surprise_pct")
            if surprise is None or not np.isfinite(surprise):
                continue
            if abs(surprise) < self.MIN_ABS_SURPRISE:
                continue

            decay = max(0.0, (self.HOLD_DAYS - days_since) / self.HOLD_DAYS)
            sue_signed[t] = float(surprise) * decay
            raw[t] = {
                "surprise_pct": float(surprise),
                "days_since": days_since,
                "decay": decay,
            }
            confidence[t] = decay   # fresher = more confident

        scores = self.zscore(sue_signed, winsorize=3.0)

        return FactorResult(
            factor_name=self.name,
            as_of=as_of,
            scores=scores,
            confidence=confidence,
            raw=raw,
            notes=f"PEAD ({len(scores)} active drift candidates of {len(universe)})",
        )
