"""
Consensus Weight Engine — Buffett Two-Tranche
==============================================
Takes analyst decisions (BUY/HOLD/EXIT) per ticker and converts them
into a diversified portfolio weight vector using a two-tranche model:

  CORE tranche (80%): up to 8 high-conviction positions at 10% each
    → Required: confidence >= 0.55
    → Hold for months; only exit on broken thesis

  TACTICAL tranche (20%): up to 4 quality picks at 5% each
    → Required: 0.40 <= confidence < 0.55
    → Must be in BUFFETT_FAVORITES or QUALITY_SECTORS (no random momentum)
    → Anti-churn handled by the 2% delta threshold in broker_base.py

Constraints:
  - Long-only (all weights >= 0)
  - Core: max 10% per ticker, max 8 positions
  - Tactical: max 5% per ticker, max 4 positions (quality-gated)
  - 100% capital utilization (idle cash is not a sin — 0% is fine)
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Buffett-style quality universe — tactical bucket is gated to these
BUFFETT_FAVORITES = frozenset({
    "AAPL", "KO", "AXP", "BAC", "JPM", "V", "MA", "JNJ",
    "PG", "WMT", "COST", "UNH", "CVX", "OXY", "AMZN", "MSFT",
    "GOOGL", "BRK-B", "MCO", "SCHW",
})

QUALITY_SECTORS = frozenset({"CONSUMER", "FINANCE", "HEALTH", "INDUSTRIAL", "TECH"})

# Rough sector map for quality gating (extended at runtime from screener)
_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "AMZN": "TECH",
    "NVDA": "TECH", "META": "TECH", "AVGO": "TECH", "ORCL": "TECH",
    "V": "FINANCE", "MA": "FINANCE", "JPM": "FINANCE", "BAC": "FINANCE",
    "AXP": "FINANCE", "GS": "FINANCE", "MS": "FINANCE", "SCHW": "FINANCE",
    "MCO": "FINANCE", "BRK-B": "FINANCE",
    "JNJ": "HEALTH", "UNH": "HEALTH", "ABT": "HEALTH", "PFE": "HEALTH",
    "ABBV": "HEALTH", "TMO": "HEALTH", "DHR": "HEALTH",
    "PG": "CONSUMER", "KO": "CONSUMER", "PEP": "CONSUMER", "WMT": "CONSUMER",
    "COST": "CONSUMER", "HD": "CONSUMER", "LOW": "CONSUMER",
    "UNP": "INDUSTRIAL", "UPS": "INDUSTRIAL", "HON": "INDUSTRIAL", "CAT": "INDUSTRIAL",
    "GE": "INDUSTRIAL", "DE": "INDUSTRIAL",
    "CVX": "ENERGY", "XOM": "ENERGY", "OXY": "ENERGY", "COP": "ENERGY",
}


class ConsensusWeightEngine:
    """
    Converts analyst BUY/HOLD/EXIT decisions into two-tranche target portfolio weights.
    """

    def __init__(
        self,
        max_positions: int = 15,
        max_per_ticker: float = 0.10,
        core_slots: int = 8,       # max CORE positions (80% bucket)
        tactical_slots: int = 4,   # max TACTICAL positions (20% bucket)
    ):
        self.max_positions = max_positions
        self.max_per_ticker = max_per_ticker
        self.core_slots = core_slots
        self.tactical_slots = tactical_slots

    def _is_quality(self, ticker: str) -> bool:
        """Return True if ticker qualifies for the tactical bucket (quality gate)."""
        t = ticker.upper()
        if t in BUFFETT_FAVORITES:
            return True
        sector = _SECTOR_MAP.get(t, "OTHER")
        return sector in QUALITY_SECTORS

    def compute_target_weights(
        self,
        tickers: List[str],
        analyst_decisions: Dict[str, Dict[str, Any]],
        held_positions: Optional[Dict[str, Any]] = None,  # {ticker: Position}
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Convert analyst decisions into portfolio weights using two-tranche model.

        Args:
            tickers: All screened tickers (held positions included by main_us.py)
            analyst_decisions: {ticker: {action, confidence, allocation_pct, reasoning, ...}}
            held_positions: Current open positions from PositionManager (for HOLD bias)

        Returns:
            target_weights: np.ndarray of shape (len(tickers),)
            actions:        {ticker: BUY|HOLD|EXIT}
        """
        held = set(held_positions.keys()) if held_positions else set()

        # ── 1. Collect BUY decisions ───────────────────────────────────────────
        buy_candidates: List[Tuple[str, float, float]] = []
        actions: Dict[str, str] = {}

        for ticker in tickers:
            decision = analyst_decisions.get(ticker, {})
            action = decision.get("action", "HOLD").upper()
            actions[ticker] = action

            if action == "BUY":
                confidence = float(decision.get("confidence", 0.3))
                suggested_alloc = float(decision.get("allocation_pct", 0.05))
                buy_candidates.append((ticker, confidence, suggested_alloc))

        # Sort by confidence descending
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        # ── 2. Split into CORE and TACTICAL buckets ────────────────────────────
        core_candidates = [
            (t, c, a) for t, c, a in buy_candidates if c >= 0.55
        ]
        tactical_candidates = [
            (t, c, a) for t, c, a in buy_candidates
            if 0.40 <= c < 0.55 and self._is_quality(t)
        ]

        selected_core = core_candidates[: self.core_slots]
        selected_tactical = tactical_candidates[: self.tactical_slots]

        # Mark overflow as HOLD
        selected_tickers = {t for t, _, _ in selected_core + selected_tactical}
        for ticker in tickers:
            if actions[ticker] == "BUY" and ticker not in selected_tickers:
                actions[ticker] = "HOLD"  # Didn't make the cut

        # ── 3. Assign weights ─────────────────────────────────────────────────
        target_weights = np.zeros(len(tickers))
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        # CORE: up to 10% each (total up to 80%)
        for ticker, conf, alloc in selected_core:
            if ticker in ticker_idx:
                target_weights[ticker_idx[ticker]] = min(0.10, alloc)

        # TACTICAL: up to 5% each (total up to 20%)
        for ticker, conf, alloc in selected_tactical:
            if ticker in ticker_idx:
                target_weights[ticker_idx[ticker]] = min(0.05, alloc * 0.5)

        # ── 4. Normalize so weights sum to ≤ 1.0 ─────────────────────────────
        total = target_weights.sum()
        if total > 1.0:
            # Cap overflow: scale down proportionally
            target_weights = target_weights / total

        # ── 5. Log summary ────────────────────────────────────────────────────
        core_count = len(selected_core)
        tact_count = len(selected_tactical)
        exit_count = sum(1 for a in actions.values() if a == "EXIT")
        hold_count = sum(1 for a in actions.values() if a == "HOLD")
        active_weight = target_weights.sum()

        logger.info(
            f"[Weights] CORE={core_count} TACTICAL={tact_count} HOLD={hold_count} EXIT={exit_count} "
            f"| Active weight={active_weight:.2%} across {sum(target_weights > 0)} positions"
        )
        print(
            f"[Weights] CORE={core_count} TACTICAL={tact_count} HOLD={hold_count} EXIT={exit_count} "
            f"| Active weight={active_weight:.2%} across {sum(target_weights > 0)} positions"
        )

        return target_weights, actions
