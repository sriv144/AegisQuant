"""
Mean Reversion Strategy
=======================
Buy oversold (RSI < 30), sell overbought (RSI > 70).
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class MeanReversionStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="mean_reversion", description="RSI-based mean reversion")
        BaseAgent.__init__(self, name="MeanReversion_Strategy", role="Mean reversion trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule: RSI_14_Z < -1.0 (oversold) → LONG, RSI_14_Z > 1.5 (overbought) → SHORT.
        Current drawdown is a risk filter: if drawdown > 15%, suppress new LONG signals.
        """
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        drawdown = portfolio_state.get("current_drawdown", 0.0)

        # Oversold = LONG opportunity
        if rsi_z < -1.0:
            action = "LONG"
            confidence = min(0.8, 0.6 + abs(rsi_z) * 0.1)
            rationale = f"RSI oversold ({rsi_z:.2f})"

            # Drawdown filter: if we're in a major drawdown, reduce confidence
            if drawdown > 0.15:
                confidence *= 0.6
                rationale += " | but high drawdown, reduced conviction"

        # Overbought = SHORT opportunity
        elif rsi_z > 1.5:
            action = "SHORT"
            confidence = min(0.8, 0.5 + rsi_z * 0.1)
            rationale = f"RSI overbought ({rsi_z:.2f})"

        # Neutral zone
        else:
            action = "HOLD"
            confidence = 0.5
            rationale = "RSI in neutral zone"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
