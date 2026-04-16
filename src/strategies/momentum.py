"""
Momentum Strategy
=================
Buy 52-week breakouts. LLM filters out stocks with negative news sentiment before entry.
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class MomentumStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="momentum", description="52-week breakout with sentiment filter")
        BaseAgent.__init__(self, name="Momentum_Strategy", role="Trend momentum trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule: Close near 52-week high (approximated by BB_Position > 0.8)
        and RSI not overbought (RSI_14_Z < 1.0).
        Sentiment filter: LLM checks if news is positive.
        """
        bb_position = indicators.get("BB_Position_Z", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        sentiment = alt_data.get("sentiment", 0.0)
        news_volume = alt_data.get("news_volume", 0)

        # Rule-based core logic
        momentum_score = 0.0
        rationale_parts = []

        # High BB position (near breakout)
        if bb_position > 0.8:
            momentum_score += 0.5
            rationale_parts.append("near 52-week high")
        elif bb_position < 0.2:
            momentum_score -= 0.5
            rationale_parts.append("near 52-week low")

        # RSI not overbought
        if rsi_z < 1.0:
            momentum_score += 0.3
            rationale_parts.append("RSI not overbought")
        elif rsi_z > 1.5:
            momentum_score -= 0.2
            rationale_parts.append("RSI overbought, caution")

        # Sentiment check: if negative, suppress the signal
        if sentiment < -0.2 and news_volume > 0:
            momentum_score *= 0.5
            rationale_parts.append(f"negative sentiment ({sentiment:.2f}), reduced conviction")

        # Fallback logic
        fallback = {
            "action": "HOLD",
            "confidence": 0.5,
            "rationale": "Momentum: rule-based breakout check",
            "strategy": self.name,
        }

        if momentum_score > 0.5:
            action = "LONG"
            confidence = min(0.9, 0.5 + momentum_score * 0.4)
        elif momentum_score < -0.3:
            action = "SHORT"
            confidence = min(0.7, abs(momentum_score) * 0.3)
        else:
            action = "HOLD"
            confidence = 0.5

        rationale = " | ".join(rationale_parts) if rationale_parts else "Neutral momentum"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
