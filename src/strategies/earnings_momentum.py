"""
Earnings Momentum Strategy
===========================
Buy 3 days before earnings, exit after announcement.
LLM screens analyst estimates vs. whisper numbers.
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class EarningsMomentumStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="earnings_momentum", description="Pre-earnings momentum play")
        BaseAgent.__init__(self, name="EarningsMomentum_Strategy", role="Earnings analyst")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule: If news volume is high (suggesting earnings announcement is near) and sentiment is building,
        buy into the momentum. Exit if consensus is weak or negative.

        Note: In production, this would require a calendar of earnings dates + LLM analysis of estimates.
        For now, use news_volume as proxy for earnings season activity.
        """
        news_volume = alt_data.get("news_volume", 0)
        sentiment = alt_data.get("sentiment", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)

        # High news volume near earnings season
        if news_volume > 2:
            # Strong pre-earnings momentum
            if sentiment > 0.1:
                action = "LONG"
                confidence = min(0.75, 0.5 + sentiment * 0.25)
                rationale = f"Pre-earnings momentum, sentiment={sentiment:.2f}, volume={news_volume}"
            elif sentiment < -0.2:
                action = "SHORT"
                confidence = min(0.6, 0.4 + abs(sentiment) * 0.2)
                rationale = f"Pre-earnings negative sentiment, expect miss"
            else:
                action = "HOLD"
                confidence = 0.4
                rationale = f"Earnings chatter but sentiment neutral"
        else:
            # No earnings activity, check general momentum
            if rsi_z > 0.5 and sentiment > 0.0:
                action = "LONG"
                confidence = 0.5
                rationale = "Building momentum outside earnings season"
            else:
                action = "HOLD"
                confidence = 0.3
                rationale = "No imminent earnings activity"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
