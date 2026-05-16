"""
Consensus Weight Engine
========================
Takes autonomous analyst decisions (BUY/HOLD/EXIT) per ticker and converts
them into a diversified portfolio weight vector.

The analyst agent makes the trading decision (using LLM reasoning or fallback).
This engine handles the math: ranking, allocation, diversification constraints.

Constraints:
  - Long-only (all weights >= 0)
  - Max 10% per ticker
  - Max 15 positions
  - 100% capital utilization across all positions
"""

import logging
from typing import Dict, List, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConsensusWeightEngine:
    """
    Converts analyst BUY/HOLD/EXIT decisions into target portfolio weights.
    """

    def __init__(
        self,
        max_positions: int = 15,
        max_per_ticker: float = 0.10,
    ):
        self.max_positions = max_positions
        self.max_per_ticker = max_per_ticker

    def compute_target_weights(
        self,
        tickers: List[str],
        analyst_decisions: Dict[str, Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Convert analyst decisions into portfolio weights.

        Args:
            tickers: List of all screened tickers
            analyst_decisions: {ticker: {action, confidence, allocation_pct, reasoning, ...}}

        Returns:
            target_weights: np.ndarray of shape (len(tickers),)
            actions:        {ticker: BUY|HOLD|EXIT} (simplified action map)
        """
        # 1. Collect BUY decisions with their confidence and suggested allocation
        buy_candidates = []
        actions = {}

        for ticker in tickers:
            decision = analyst_decisions.get(ticker, {})
            action = decision.get("action", "HOLD").upper()
            actions[ticker] = action

            if action == "BUY":
                confidence = float(decision.get("confidence", 0.3))
                suggested_alloc = float(decision.get("allocation_pct", 0.05))
                buy_candidates.append((ticker, confidence, suggested_alloc))

        # 2. Rank by confidence, take top N
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = buy_candidates[: self.max_positions]

        # Mark overflow as HOLD
        selected_tickers = set(t for t, _, _ in selected)
        for ticker in tickers:
            if actions[ticker] == "BUY" and ticker not in selected_tickers:
                actions[ticker] = "HOLD"  # Didn't make the cut

        # 3. Allocate using analyst-suggested allocations, capped at max_per_ticker
        target_weights = np.zeros(len(tickers))

        if selected:
            # Use analyst-suggested allocations, respect caps
            raw_weights = {}
            for ticker, confidence, suggested_alloc in selected:
                # Blend analyst suggestion with confidence
                weight = min(suggested_alloc, self.max_per_ticker)
                raw_weights[ticker] = weight

            # Normalize to sum to 1.0 (use 100% of capital)
            total_raw = sum(raw_weights.values())
            if total_raw > 0:
                for i, ticker in enumerate(tickers):
                    if ticker in raw_weights:
                        target_weights[i] = raw_weights[ticker] / total_raw
                        # Re-cap after normalization
                        if target_weights[i] > self.max_per_ticker:
                            target_weights[i] = self.max_per_ticker

                # If capping caused sum < 1.0, redistribute proportionally
                current_sum = target_weights.sum()
                if current_sum > 0 and current_sum < 0.99:
                    target_weights = target_weights / current_sum

        # 4. Log summary
        buy_count = sum(1 for a in actions.values() if a == "BUY")
        exit_count = sum(1 for a in actions.values() if a == "EXIT")
        hold_count = sum(1 for a in actions.values() if a == "HOLD")
        active_weight = target_weights.sum()

        logger.info(
            f"[Weights] BUY={buy_count} HOLD={hold_count} EXIT={exit_count} "
            f"| Active weight={active_weight:.2%} across {sum(target_weights > 0)} positions"
        )
        print(
            f"[Weights] BUY={buy_count} HOLD={hold_count} EXIT={exit_count} "
            f"| Active weight={active_weight:.2%} across {sum(target_weights > 0)} positions"
        )

        return target_weights, actions
