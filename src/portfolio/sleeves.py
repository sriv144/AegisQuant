"""
Portfolio Sleeves
=================

A "sleeve" is one independent strategy. Each sleeve:

  1. Defines a candidate universe
  2. Scores tickers by some signal (composite of factors or single factor)
  3. Picks a position set (typically top decile / top N)
  4. Assigns weights summing to 1.0 within the sleeve

The PORTFOLIO COMBINER (PMAgent, Phase 4) then takes K sleeves and assigns
*sleeve-level* weights via risk parity, so the total portfolio weights are:

    total_weight[t] = sum_k(sleeve_weight[k] * within_sleeve_weight[k, t])

Design principles
-----------------
- Sleeves are stateless. All data lives in the FactorEngine / DataProvider.
- Sleeves never call brokers or check positions — they output target weights.
- Each sleeve has its own rebalance_freq; the orchestrator (Phase 4) decides
  when to refresh each.
- Empty/insufficient data → sleeve returns no positions (the combiner reduces
  total exposure accordingly).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.factors import (
    ValueFactor, QualityFactor, MomentumFactor,
    DefensiveFactor, PEADFactor, InsiderFactor,
    sp500_tickers, sp100_tickers,
)

logger = logging.getLogger(__name__)


@dataclass
class SleeveResult:
    """Output of Sleeve.weights() — used by the PM combiner."""
    sleeve_name: str
    as_of: datetime
    weights: Dict[str, float]                   # ticker -> in-sleeve weight (sum to 1.0)
    raw_scores: Dict[str, float] = field(default_factory=dict)
    n_candidates: int = 0
    notes: str = ""

    def is_active(self) -> bool:
        return bool(self.weights)


class Sleeve(ABC):
    """Base sleeve. Subclasses implement score() and may override universe()/weights()."""

    name: str = "unnamed"
    rebalance_freq: str = "monthly"   # 'monthly' / 'weekly' / 'event'
    target_positions: int = 20         # top-N to hold
    max_position_weight: float = 0.08  # cap any single name at 8% of sleeve
    min_score_for_inclusion: float = 0.5  # minimum composite z-score (top half-ish)

    def universe(self) -> List[str]:
        """Default universe = S&P 500. Subclasses may restrict (e.g. exclude micro caps)."""
        return sp500_tickers()

    @abstractmethod
    def score(self, as_of: Optional[datetime] = None) -> Dict[str, float]:
        """Composite score per ticker (higher = more attractive)."""
        ...

    def weights(self, as_of: Optional[datetime] = None) -> SleeveResult:
        """
        Default weighting:
          1. Drop NaN and tickers below min_score_for_inclusion.
          2. Take top-N by score.
          3. Assign rank-based weights (N for #1, N-1 for #2, ..., 1 for #N).
             This is more stable than score-proportional when scores are extreme.
          4. Iteratively cap at max_position_weight (water-filling) until stable.
        """
        as_of = as_of or datetime.utcnow()
        raw = self.score(as_of)
        raw = {t: float(s) for t, s in raw.items() if s is not None and np.isfinite(s)}
        if not raw:
            return SleeveResult(self.name, as_of, {}, raw, 0, "no candidates")

        # Filter on min score; fall back to top-N if nothing qualifies
        eligible = {t: s for t, s in raw.items() if s >= self.min_score_for_inclusion}
        if not eligible:
            eligible = dict(sorted(raw.items(), key=lambda kv: -kv[1])[: self.target_positions])

        top = sorted(eligible.items(), key=lambda kv: -kv[1])[: self.target_positions]
        if not top:
            return SleeveResult(self.name, as_of, {}, raw, 0, "no top-N picks")

        # Rank-based weights: position i (0-indexed best) gets (N - i)
        n = len(top)
        rank_weights = np.arange(n, 0, -1, dtype=float)   # [N, N-1, ..., 1]
        rank_weights = rank_weights / rank_weights.sum()

        # Iterative cap + redistribute (a.k.a. water-filling). At most n passes.
        cap = self.max_position_weight
        w = rank_weights.copy()
        for _ in range(n):
            over = w > cap
            if not over.any():
                break
            excess = float((w[over] - cap).sum())
            w[over] = cap
            free_mask = (~over) & (w > 0)
            if not free_mask.any():
                # Can't redistribute anywhere → leave as is (sum may be <1)
                break
            w[free_mask] += excess * (w[free_mask] / w[free_mask].sum())

        # Final tidy renormalize (handles any tiny numerical drift)
        s = w.sum()
        if s > 0:
            w = w / s

        weight_map = {t: float(wi) for (t, _), wi in zip(top, w) if wi > 1e-9}
        return SleeveResult(
            sleeve_name=self.name,
            as_of=as_of,
            weights=weight_map,
            raw_scores={t: float(s) for t, s in raw.items()},
            n_candidates=len(raw),
            notes=f"{len(weight_map)} positions, top score={top[0][1]:.2f}, weight_top={max(weight_map.values()):.3f}"
                  if weight_map else "no positions",
        )

    def __repr__(self):
        return f"<Sleeve name={self.name!r} freq={self.rebalance_freq} target_n={self.target_positions}>"


# ── Concrete sleeves ──────────────────────────────────────────────────────────


class ValueQualityMomentumSleeve(Sleeve):
    """
    Long-only equity sleeve combining three classic factors.

    Weights: value 35% / quality 35% / momentum 30% (Asness-style "all-weather"
    factor mix; momentum is slightly underweighted because we have a dedicated
    momentum sleeve too).

    Universe: S&P 500.
    Rebalance: monthly.
    """
    name = "value_quality_momentum"
    rebalance_freq = "monthly"
    target_positions = 25
    max_position_weight = 0.06   # cap at 6% of sleeve = ~1.5% of book if sleeve is 25% of NAV
    min_score_for_inclusion = 0.3

    FACTOR_WEIGHTS = {"value": 0.35, "quality": 0.35, "momentum": 0.30}

    def score(self, as_of: Optional[datetime] = None) -> Dict[str, float]:
        as_of = as_of or datetime.utcnow()
        u = self.universe()

        v_res = ValueFactor().compute(u, as_of)
        q_res = QualityFactor().compute(u, as_of)
        m_res = MomentumFactor().compute(u, as_of)

        composite: Dict[str, float] = {}
        for t in set(v_res.scores) | set(q_res.scores) | set(m_res.scores):
            parts, weights = [], []
            if t in v_res.scores:
                parts.append(v_res.scores[t] * v_res.confidence.get(t, 1.0))
                weights.append(self.FACTOR_WEIGHTS["value"])
            if t in q_res.scores:
                parts.append(q_res.scores[t] * q_res.confidence.get(t, 1.0))
                weights.append(self.FACTOR_WEIGHTS["quality"])
            if t in m_res.scores:
                parts.append(m_res.scores[t] * m_res.confidence.get(t, 1.0))
                weights.append(self.FACTOR_WEIGHTS["momentum"])
            if parts:
                composite[t] = float(np.average(parts, weights=weights))
        return composite


class CrossSectionalMomentumSleeve(Sleeve):
    """
    Pure momentum sleeve. Long-only top-decile of cross-sectional momentum.

    Universe: S&P 500.
    Rebalance: monthly.

    The factor here is just MomentumFactor (12-1m + Gray smoothness). We pick
    the top ~50 names (top decile of S&P 500) and equal-weight them. This is
    the closest analogue to the Alpha Architect QMOM ETF methodology.
    """
    name = "xs_momentum"
    rebalance_freq = "monthly"
    target_positions = 50
    max_position_weight = 0.04   # 4% cap = ~equal-weight with a haircut on the very top
    min_score_for_inclusion = 0.5

    def score(self, as_of: Optional[datetime] = None) -> Dict[str, float]:
        return MomentumFactor().compute(self.universe(), as_of).scores


class PEADSleeve(Sleeve):
    """
    Post-Earnings Announcement Drift sleeve. Event-driven, weekly rebalance.

    Holds tickers within 60 days of a beat-or-miss earnings event, weighted by
    z(SUE) * decay. Typically 5–25 active positions depending on the calendar.
    """
    name = "pead"
    rebalance_freq = "weekly"
    target_positions = 20
    max_position_weight = 0.10   # higher cap — fewer candidates means more concentration is OK
    min_score_for_inclusion = 0.5   # only act on meaningfully positive surprises

    def score(self, as_of: Optional[datetime] = None) -> Dict[str, float]:
        return PEADFactor().compute(self.universe(), as_of).scores


class InsiderBuyingSleeve(Sleeve):
    """
    Insider Buying sleeve. Event-driven, weekly rebalance.

    Holds tickers with significant opportunistic insider purchases in last 90d
    (Cohen-Malloy-Pomorski style). Tighter universe — only S&P 100 — because
    EDGAR Form 4 fetching is slow and we want fewer candidates to vet.
    """
    name = "insider_buying"
    rebalance_freq = "weekly"
    target_positions = 15
    max_position_weight = 0.12
    min_score_for_inclusion = 0.5

    def universe(self) -> List[str]:
        # Tighter universe — EDGAR fetching ~5s/ticker even cached
        return sp100_tickers()

    def score(self, as_of: Optional[datetime] = None) -> Dict[str, float]:
        return InsiderFactor().compute(self.universe(), as_of).scores


# ── Convenience ──────────────────────────────────────────────────────────────


def all_sleeves() -> Dict[str, Sleeve]:
    """Return one instance of each sleeve."""
    return {
        "value_quality_momentum": ValueQualityMomentumSleeve(),
        "xs_momentum": CrossSectionalMomentumSleeve(),
        "pead": PEADSleeve(),
        "insider_buying": InsiderBuyingSleeve(),
    }
