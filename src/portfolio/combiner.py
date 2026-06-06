"""
Portfolio Combiner — risk-parity sleeve aggregator.

Replaces the old per-ticker PMAgent. Takes K sleeves (each with in-sleeve
weights summing to 1.0) and combines them into a single ticker-level target
weight vector at the portfolio level.

Algorithm
---------
1. **Risk-parity sleeve weights** — inverse-volatility weighting of the K
   sleeves, where each sleeve's vol is approximated by the median 60-day
   realized vol of its constituent tickers. Cap each sleeve at MAX_SLEEVE_NAV.

2. **Macro regime overlay** — the MacroRegimeAgent emits a score in [-3, +3].
   In risk-off regimes (score < -1) we shift weight FROM equity sleeves
   (VQM, XS momentum) TO defensive / cash. Shift magnitude is capped at ±20%
   per the plan ("LLM regime overlay can shift sleeve weights ±20%").

3. **Combine** — portfolio_weight[ticker] = sum over sleeves of:
       sleeve_weight[k] * within_sleeve_weight[k][ticker]

4. **Cash buffer** — total invested cannot exceed `max_total_invested`
   (default 1.0 = fully invested). Anything left over is cash.

Output is a PortfolioTarget; the RiskOfficer then enforces hard constraints.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.portfolio.sleeves import Sleeve, SleeveResult
from src.factors.data_provider import get_data_provider

logger = logging.getLogger(__name__)


@dataclass
class PortfolioTarget:
    """The combiner's proposal — pre-RiskOfficer."""
    as_of: datetime
    ticker_weights: Dict[str, float] = field(default_factory=dict)   # ticker -> NAV fraction
    sleeve_weights: Dict[str, float] = field(default_factory=dict)   # sleeve_name -> NAV fraction
    cash_weight: float = 1.0
    macro_regime: float = 0.0
    rationale: str = ""

    @property
    def total_invested(self) -> float:
        return float(sum(self.ticker_weights.values()))

    @property
    def n_positions(self) -> int:
        return len(self.ticker_weights)


class Combiner:
    """Risk-parity sleeve combiner with macro overlay."""

    # Hard ceilings (re-checked in RiskOfficer too — belt and suspenders)
    MAX_SLEEVE_NAV = 0.40
    MAX_TOTAL_INVESTED = 1.0

    # Sleeves with high equity beta — get DOWNWEIGHTED in risk-off regimes
    EQUITY_BETA_SLEEVES = {"value_quality_momentum", "xs_momentum", "pead"}

    # Macro overlay: each unit of regime score shifts equity sleeves by this fraction
    # (signed). At regime=-3 (full risk-off), equity sleeves shrink by 60%.
    MACRO_SHIFT_PER_UNIT = 0.20

    def __init__(
        self,
        data_provider=None,
        max_sleeve_nav: Optional[float] = None,
        max_total_invested: Optional[float] = None,
    ):
        self.dp = data_provider or get_data_provider()
        self.max_sleeve_nav = float(max_sleeve_nav if max_sleeve_nav is not None else self.MAX_SLEEVE_NAV)
        self.max_total_invested = float(
            max_total_invested if max_total_invested is not None else self.MAX_TOTAL_INVESTED
        )

    # ── public API ──────────────────────────────────────────────────────────

    def combine(
        self,
        sleeve_results: Dict[str, SleeveResult],
        sleeve_vol_estimates: Optional[Dict[str, float]] = None,
        macro_regime_score: float = 0.0,
        macro_regime_confidence: float = 0.0,
    ) -> PortfolioTarget:
        """
        Combine K sleeves into portfolio weights.

        Args:
            sleeve_results: {sleeve_name -> SleeveResult}
            sleeve_vol_estimates: optional {sleeve_name -> annualized vol}.
                If None, we estimate from median ticker vol over last 60 days.
            macro_regime_score: from MacroRegimeAgent, in [-3, +3].
            macro_regime_confidence: from MacroRegimeAgent, in [0, 1].
                Score is scaled by confidence before applying.
        """
        active = {k: r for k, r in sleeve_results.items() if r.is_active()}
        if not active:
            return PortfolioTarget(
                as_of=datetime.utcnow(), cash_weight=1.0,
                rationale="No active sleeves",
            )

        # 1) Risk-parity sleeve weights
        vols = sleeve_vol_estimates or self._estimate_sleeve_vols(active)
        sleeve_w = self._risk_parity_weights(active, vols)

        # 2) Macro overlay
        adjusted_macro = macro_regime_score * max(0.0, min(1.0, macro_regime_confidence))
        sleeve_w = self._apply_macro_overlay(sleeve_w, adjusted_macro)

        # 3) Cap each sleeve at MAX_SLEEVE_NAV (then renormalize so total <= MAX_TOTAL_INVESTED)
        sleeve_w = self._cap_and_normalize(sleeve_w)

        # 4) Combine to ticker weights
        ticker_w: Dict[str, float] = {}
        for k, w in sleeve_w.items():
            for t, in_sleeve_w in active[k].weights.items():
                ticker_w[t] = ticker_w.get(t, 0.0) + w * in_sleeve_w

        cash = max(0.0, 1.0 - sum(ticker_w.values()))
        rationale = (
            f"sleeves={list(sleeve_w.keys())} weights={ {k: round(v, 3) for k, v in sleeve_w.items()} } "
            f"macro_regime={macro_regime_score:+.2f}*conf{macro_regime_confidence:.2f}={adjusted_macro:+.2f} "
            f"n_positions={len(ticker_w)} cash={cash:.2%}"
        )
        return PortfolioTarget(
            as_of=datetime.utcnow(),
            ticker_weights=ticker_w,
            sleeve_weights=sleeve_w,
            cash_weight=cash,
            macro_regime=adjusted_macro,
            rationale=rationale,
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _estimate_sleeve_vols(self, sleeves: Dict[str, SleeveResult]) -> Dict[str, float]:
        """
        Approximate each sleeve's annualized volatility using the median 60-day
        realized vol of its top constituents. This is the "poor man's portfolio
        vol" — much faster than computing the full covariance matrix and good
        enough for risk-parity weighting at this scale.
        """
        # Collect all unique tickers across sleeves
        all_tickers = sorted({t for r in sleeves.values() for t in r.weights})
        if not all_tickers:
            return {k: 0.20 for k in sleeves}   # default 20% vol

        try:
            prices = self.dp.get_prices(all_tickers, period="6mo")
            if prices is None or prices.empty:
                raise RuntimeError("no price data")
            daily_ret = prices.pct_change(fill_method=None).dropna(how="all")
            vol_60d = daily_ret.tail(60).std(ddof=0) * np.sqrt(252)
        except Exception as e:
            logger.warning(f"Combiner: vol estimation failed ({e}), using uniform 0.20")
            return {k: 0.20 for k in sleeves}

        out = {}
        for k, r in sleeves.items():
            tickers_in_sleeve = [t for t in r.weights if t in vol_60d.index]
            if not tickers_in_sleeve:
                out[k] = 0.20
                continue
            sleeve_vols = vol_60d.loc[tickers_in_sleeve].dropna()
            out[k] = float(sleeve_vols.median()) if len(sleeve_vols) else 0.20
        return out

    @staticmethod
    def _risk_parity_weights(
        sleeves: Dict[str, SleeveResult],
        vols: Dict[str, float],
    ) -> Dict[str, float]:
        """Inverse-vol weights: w_k ∝ 1/vol_k, normalized to sum 1.0."""
        inv = {}
        for k in sleeves:
            v = vols.get(k, 0.20)
            inv[k] = 1.0 / max(v, 0.05)   # floor vol at 5% to avoid blow-up
        total = sum(inv.values())
        if total <= 0:
            n = len(sleeves)
            return {k: 1.0 / n for k in sleeves}
        return {k: v / total for k, v in inv.items()}

    def _apply_macro_overlay(
        self,
        sleeve_w: Dict[str, float],
        regime_score: float,
    ) -> Dict[str, float]:
        """
        Shift weight FROM equity-beta sleeves TO defensive when regime < 0.
        Shift magnitude per sleeve: regime * MACRO_SHIFT_PER_UNIT, capped at ±60%.
        """
        if abs(regime_score) < 0.1:
            return sleeve_w   # no-op for tiny regimes

        shift_frac = max(-0.60, min(0.60, regime_score * self.MACRO_SHIFT_PER_UNIT))
        # In risk-off (shift_frac < 0), equity sleeves get scaled by (1 + shift_frac)
        adjusted = {}
        equity_total = 0.0
        defensive_total = 0.0
        for k, w in sleeve_w.items():
            if k in self.EQUITY_BETA_SLEEVES:
                new_w = w * (1.0 + shift_frac)
                adjusted[k] = max(0.0, new_w)
                equity_total += w
            else:
                adjusted[k] = w
                defensive_total += w

        # Conservation: distribute the freed/required weight to defensive sleeves
        # (if shift_frac < 0, equity shrank; redistribute to defensives or cash).
        delta = sum(sleeve_w.values()) - sum(adjusted.values())   # weight that shrank
        if defensive_total > 0 and delta > 0:
            # Pro-rata add to defensive sleeves
            for k in adjusted:
                if k not in self.EQUITY_BETA_SLEEVES:
                    adjusted[k] += delta * (sleeve_w[k] / defensive_total)
        # If no defensive sleeves, delta stays as cash (sum < 1).
        return adjusted

    def _cap_and_normalize(self, sleeve_w: Dict[str, float]) -> Dict[str, float]:
        """Cap each sleeve at configured limits; if total is too high, scale down."""
        # Apply per-sleeve cap (iterative water-filling)
        w = dict(sleeve_w)
        for _ in range(len(w)):
            over = {k: v for k, v in w.items() if v > self.max_sleeve_nav}
            if not over:
                break
            excess = sum(v - self.max_sleeve_nav for v in over.values())
            for k in over:
                w[k] = self.max_sleeve_nav
            under = [k for k, v in w.items() if v < self.max_sleeve_nav and k not in over]
            if not under:
                break
            under_total = sum(w[k] for k in under)
            if under_total <= 0:
                break
            for k in under:
                w[k] += excess * (w[k] / under_total)

        # Cap total
        total = sum(w.values())
        if total > self.max_total_invested:
            scale = self.max_total_invested / total
            w = {k: v * scale for k, v in w.items()}
        return w
