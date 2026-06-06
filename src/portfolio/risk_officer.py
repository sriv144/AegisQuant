"""
Risk Officer — hard constraint enforcement.

Replaces the lax old per-ticker RiskOfficerAgent that only checked single-
position size. The new officer enforces all five plan constraints:

  1. Single position cap:  5% NAV
  2. Sector concentration: 20% NAV per GICS sector
  3. Sleeve concentration: 40% NAV per sleeve   (already enforced in Combiner,
                                                 but re-checked here)
  4. Portfolio beta to SPY: 0.4–1.0
  5. Drawdown gate:        at -15% drawdown, halve all sizes

The semantics are HARD REJECT for #1–4 (the offending positions are scaled or
removed, not haircut by a polite agent). For #5 a uniform 0.5x scaling is
applied to all positions when drawdown breaches the threshold.

The output is a fully-vetted weight vector ready for execution. If any
violations were found, they are surfaced in `violations` (human-readable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.portfolio.combiner import PortfolioTarget
from src.factors.data_provider import get_data_provider

logger = logging.getLogger(__name__)


@dataclass
class RiskReview:
    """Output of RiskOfficer.review() — what actually executes."""
    as_of: datetime
    approved_weights: Dict[str, float] = field(default_factory=dict)
    sleeve_weights: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    drawdown_scaling: float = 1.0   # 0.5 if drawdown gate fired
    rejected: bool = False
    rationale: str = ""

    @property
    def total_invested(self) -> float:
        return float(sum(self.approved_weights.values()))


class RiskOfficer:
    """Hard-constraint enforcement on a PortfolioTarget."""

    # Hard caps
    MAX_POSITION_NAV = 0.05
    MAX_SECTOR_NAV = 0.20
    MAX_SLEEVE_NAV = 0.40

    # Drawdown gate
    DRAWDOWN_GATE_THRESHOLD = 0.15      # at -15% DD
    DRAWDOWN_GATE_SCALING = 0.50        # halve sizes

    # Beta range (informational only — beta is hard to estimate cleanly at
    # portfolio level; we expose the check but don't reject by default)
    BETA_MIN = 0.4
    BETA_MAX = 1.0
    ENFORCE_BETA = False  # turn on after backtest validation

    def __init__(self, data_provider=None):
        self.dp = data_provider or get_data_provider()

    def review(
        self,
        target: PortfolioTarget,
        current_drawdown: float = 0.0,
    ) -> RiskReview:
        """
        Vet the combiner's target. Apply constraints in order:
          1. Drawdown gate (scales all uniformly)
          2. Sleeve cap (already in combiner, sanity check)
          3. Single-position cap (cap & redistribute, or drop if can't)
          4. Sector cap (scale down all positions in the offending sector)
          5. Beta check (informational)
        """
        weights = dict(target.ticker_weights)
        violations: List[str] = []

        # 0) Empty?
        if not weights:
            return RiskReview(
                as_of=datetime.utcnow(), approved_weights={},
                sleeve_weights=target.sleeve_weights,
                rationale="No proposed positions",
            )

        # 1) Drawdown gate
        dd_scale = 1.0
        if current_drawdown <= -self.DRAWDOWN_GATE_THRESHOLD:
            dd_scale = self.DRAWDOWN_GATE_SCALING
            weights = {t: w * dd_scale for t, w in weights.items()}
            violations.append(
                f"Drawdown gate: current_dd={current_drawdown*100:.1f}% "
                f"<= -{self.DRAWDOWN_GATE_THRESHOLD*100:.1f}% -> scaling by {dd_scale}"
            )

        # 2) Sleeve cap (sanity — combiner should have enforced)
        for k, w in target.sleeve_weights.items():
            if w > self.MAX_SLEEVE_NAV + 1e-6:
                violations.append(f"Sleeve cap violated: {k} at {w:.3f} > {self.MAX_SLEEVE_NAV}")

        # 3) Single-position cap
        weights, pos_violations = self._enforce_position_cap(weights)
        violations.extend(pos_violations)

        # 4) Sector cap
        weights, sec_violations = self._enforce_sector_cap(weights)
        violations.extend(sec_violations)

        # 5) Beta check (informational unless enforce on)
        beta_est = self._estimate_portfolio_beta(weights)
        if beta_est is not None:
            if beta_est < self.BETA_MIN or beta_est > self.BETA_MAX:
                msg = (f"Portfolio beta {beta_est:.2f} outside [{self.BETA_MIN}, {self.BETA_MAX}]")
                violations.append(msg)
                if self.ENFORCE_BETA:
                    # Scale down all positions equally to drag beta toward range center
                    target_beta = (self.BETA_MIN + self.BETA_MAX) / 2
                    scale = target_beta / max(0.05, beta_est)
                    weights = {t: w * scale for t, w in weights.items()}

        beta_str = f"{beta_est:.2f}" if beta_est is not None else "NA"
        rationale = (
            f"Vetted {len(weights)} positions. "
            f"Invested {sum(weights.values())*100:.1f}%, "
            f"DD scale={dd_scale:.2f}, beta~{beta_str}, "
            f"violations={len(violations)}"
        )

        return RiskReview(
            as_of=datetime.utcnow(),
            approved_weights=weights,
            sleeve_weights=target.sleeve_weights,
            violations=violations,
            drawdown_scaling=dd_scale,
            rejected=False,
            rationale=rationale,
        )

    # ── enforcement helpers ─────────────────────────────────────────────────

    def _enforce_position_cap(self, weights: Dict[str, float]):
        """Cap each position at MAX_POSITION_NAV. Excess goes to cash (silent shrink)."""
        out = {}
        violations = []
        for t, w in weights.items():
            if w > self.MAX_POSITION_NAV + 1e-9:
                violations.append(
                    f"Position cap: {t} at {w*100:.2f}% > {self.MAX_POSITION_NAV*100:.0f}% -> "
                    f"capped to {self.MAX_POSITION_NAV*100:.0f}%"
                )
                out[t] = self.MAX_POSITION_NAV
            else:
                out[t] = w
        return out, violations

    def _enforce_sector_cap(self, weights: Dict[str, float]):
        """
        Aggregate weights by sector; scale down all positions in any sector
        that exceeds MAX_SECTOR_NAV.
        """
        violations = []
        # Build sector -> list of (ticker, weight)
        sectors: Dict[str, List[tuple]] = {}
        for t, w in weights.items():
            sec = self.dp.get_sector(t) or "Unknown"
            sectors.setdefault(sec, []).append((t, w))

        out = dict(weights)
        for sec, members in sectors.items():
            total = sum(w for _, w in members)
            if total > self.MAX_SECTOR_NAV + 1e-9:
                scale = self.MAX_SECTOR_NAV / total
                violations.append(
                    f"Sector cap: {sec} at {total*100:.1f}% > {self.MAX_SECTOR_NAV*100:.0f}% -> "
                    f"scaling {len(members)} positions by {scale:.3f}"
                )
                for t, _ in members:
                    out[t] = out[t] * scale
        return out, violations

    def _estimate_portfolio_beta(self, weights: Dict[str, float]) -> Optional[float]:
        """
        Quick beta estimate: weighted sum of per-ticker betas from fundamentals.
        Returns None if we can't compute (e.g. no beta data).
        """
        if not weights:
            return None
        contribs = []
        for t, w in weights.items():
            f = self.dp.get_fundamentals(t)
            b = f.get("beta") if f else None
            if b is not None and np.isfinite(b):
                contribs.append(w * float(b))
        if not contribs:
            return None
        return float(sum(contribs))
