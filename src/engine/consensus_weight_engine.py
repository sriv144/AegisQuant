"""
Consensus Weight Engine
========================
Replaces RL-driven weight generation with an agent + strategy consensus
approach, filtered by Bollinger Band entry/exit rules.

Algorithm:
  1. Score each ticker: consensus = 0.6 * agent_score + 0.4 * strategy_score
  2. Bollinger Band filter:
     - ENTER if BB_Position > 1.0 (close > upper BB) AND consensus >= 0.30
     - EXIT  if BB_Position < 0.0 (close < lower BB) for held positions
     - HOLD  if held and BB_Position >= 0.0
  3. Rank qualified tickers by score, take top 15
  4. Confidence-weighted allocation capped at 10% per ticker
  5. Exit tickers that fail BB or aren't in top 15
"""

import logging
from typing import Dict, List, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConsensusWeightEngine:
    """
    Builds target portfolio weights from agent signals + strategy signals + BB filter.
    Long-only: all weights >= 0. No shorts.
    """

    def __init__(
        self,
        agent_weight: float = 0.6,
        strategy_weight: float = 0.4,
        min_consensus: float = 0.30,
        max_positions: int = 15,
        max_per_ticker: float = 0.10,
    ):
        self.agent_weight = agent_weight
        self.strategy_weight = strategy_weight
        self.min_consensus = min_consensus
        self.max_positions = max_positions
        self.max_per_ticker = max_per_ticker

    def compute_consensus_scores(
        self,
        tickers: List[str],
        agent_signals: Dict[str, List[Dict[str, Any]]],
        strategy_signals: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """
        Compute consensus score for each ticker.

        agent_signals:    {ticker: [{action, confidence, ...}, ...]}  from 4 research agents
        strategy_signals: {ticker: [{action, confidence, ...}, ...]}  from 9 strategies

        Only PROPOSE_LONG / LONG actions contribute. SHORT/HOLD = 0.
        """
        scores = {}
        for ticker in tickers:
            # Agent score: average confidence of LONG-proposing agents
            a_signals = agent_signals.get(ticker, [])
            a_long_confs = [
                float(s.get("confidence", 0.0))
                for s in a_signals
                if s.get("action") in ("PROPOSE_LONG",)
            ]
            agent_score = sum(a_long_confs) / max(len(a_signals), 1) if a_signals else 0.0

            # Strategy score: average confidence of LONG strategies
            s_signals = strategy_signals.get(ticker, [])
            s_long_confs = [
                float(s.get("confidence", 0.0))
                for s in s_signals
                if s.get("action") in ("LONG",)
            ]
            strategy_score = sum(s_long_confs) / max(len(s_signals), 1) if s_signals else 0.0

            consensus = self.agent_weight * agent_score + self.strategy_weight * strategy_score
            scores[ticker] = round(consensus, 4)

        return scores

    def apply_bb_filter(
        self,
        tickers: List[str],
        consensus_scores: Dict[str, float],
        indicators: Dict[str, Dict[str, Any]],
        held_tickers: set,
    ) -> Dict[str, str]:
        """
        Apply Bollinger Band entry/exit rules.

        Returns: {ticker: action} where action is ENTER, EXIT, HOLD, or SKIP.

        Rules:
          - Held AND BB_Position < 0.0 → EXIT (close below lower BB)
          - Held AND BB_Position >= 0.0 → HOLD (keep holding)
          - Not held AND BB_Position > 1.0 AND consensus >= min → ENTER
          - Else → SKIP
        """
        actions = {}
        for ticker in tickers:
            bb_pos = indicators.get(ticker, {}).get("BB_Position", 0.5)
            score = consensus_scores.get(ticker, 0.0)
            is_held = ticker in held_tickers

            if is_held and bb_pos < 0.0:
                actions[ticker] = "EXIT"
            elif is_held:
                actions[ticker] = "HOLD"
            elif bb_pos > 1.0 and score >= self.min_consensus:
                actions[ticker] = "ENTER"
            else:
                actions[ticker] = "SKIP"

        return actions

    def compute_target_weights(
        self,
        tickers: List[str],
        agent_signals: Dict[str, List[Dict[str, Any]]],
        strategy_signals: Dict[str, List[Dict[str, Any]]],
        indicators: Dict[str, Dict[str, Any]],
        held_tickers: set,
    ) -> Tuple[np.ndarray, Dict[str, str], Dict[str, float]]:
        """
        Full pipeline: consensus scoring → BB filter → rank → allocate.

        Returns:
            target_weights: np.ndarray of shape (len(tickers),)
            actions:        {ticker: ENTER|EXIT|HOLD|SKIP}
            scores:         {ticker: consensus_score}
        """
        # 1. Consensus scores
        scores = self.compute_consensus_scores(tickers, agent_signals, strategy_signals)

        # 2. BB filter
        actions = self.apply_bb_filter(tickers, scores, indicators, held_tickers)

        # 3. Collect eligible tickers (ENTER + HOLD)
        eligible = []
        for ticker in tickers:
            if actions[ticker] in ("ENTER", "HOLD"):
                eligible.append((ticker, scores.get(ticker, 0.0)))

        # 4. Rank by score, take top N
        eligible.sort(key=lambda x: x[1], reverse=True)
        top_tickers = set(t for t, _ in eligible[: self.max_positions])

        # Tickers not in top N that are held → mark for EXIT
        for ticker in tickers:
            if actions[ticker] in ("ENTER", "HOLD") and ticker not in top_tickers:
                actions[ticker] = "EXIT"

        # 5. Confidence-weighted allocation, capped at max_per_ticker
        total_score = sum(s for t, s in eligible[: self.max_positions]) or 1.0
        target_weights = np.zeros(len(tickers))

        for i, ticker in enumerate(tickers):
            if ticker in top_tickers and actions[ticker] in ("ENTER", "HOLD"):
                raw_weight = scores[ticker] / total_score
                target_weights[i] = min(raw_weight, self.max_per_ticker)
            elif actions[ticker] == "EXIT":
                target_weights[i] = 0.0  # Signal to close
            # SKIP → 0.0 (no position)

        # Normalize so total = 1.0 (use 100% of capital)
        alloc_sum = target_weights.sum()
        if alloc_sum > 0:
            target_weights = target_weights / alloc_sum

        # Re-cap after normalization
        capped = False
        for i in range(len(target_weights)):
            if target_weights[i] > self.max_per_ticker:
                target_weights[i] = self.max_per_ticker
                capped = True
        if capped:
            # Re-normalize remaining
            remaining = 1.0 - sum(w for w in target_weights if w >= self.max_per_ticker)
            uncapped_sum = sum(w for w in target_weights if w < self.max_per_ticker)
            if uncapped_sum > 0 and remaining > 0:
                for i in range(len(target_weights)):
                    if target_weights[i] < self.max_per_ticker and target_weights[i] > 0:
                        target_weights[i] = target_weights[i] / uncapped_sum * remaining

        # Log summary
        enter_count = sum(1 for a in actions.values() if a == "ENTER")
        hold_count = sum(1 for a in actions.values() if a == "HOLD")
        exit_count = sum(1 for a in actions.values() if a == "EXIT")
        skip_count = sum(1 for a in actions.values() if a == "SKIP")
        active_weight = target_weights.sum()

        logger.info(
            f"[Consensus] ENTER={enter_count} HOLD={hold_count} EXIT={exit_count} "
            f"SKIP={skip_count} | Active weight={active_weight:.2%}"
        )

        return target_weights, actions, scores
