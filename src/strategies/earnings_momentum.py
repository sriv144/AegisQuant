"""
Earnings Momentum Strategy
===========================
Pre/post-earnings momentum play. High news volume + building sentiment =
approaching earnings. Combines news flow, price momentum, and volume
confirmation to ride earnings drift.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


class EarningsMomentumStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="earnings_momentum", description="Earnings drift with sentiment + momentum")
        BaseAgent.__init__(self, name="EarningsMomentum_Strategy", role="Earnings analyst")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        news_volume = alt_data.get("news_volume", 0)
        sentiment = alt_data.get("sentiment", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        macd_z = indicators.get("MACD_Z", 0.0)
        mom_z = indicators.get("mom_12m_Z", 0.0)
        volume_z = indicators.get("Volume_Z", 0.0)
        adx = indicators.get("ADX_14", 20.0)

        score = 0.0
        reasons = []

        # Earnings proximity proxy: elevated news volume
        if news_volume < 2:
            return {
                "action": "HOLD",
                "confidence": 0.2,
                "rationale": f"low news activity ({news_volume}), no earnings catalyst detected",
                "strategy": self.name,
            }

        reasons.append(f"elevated news flow ({news_volume} articles)")

        # Sentiment direction — strong sentiment pre-earnings is a directional signal
        if sentiment > 0.2:
            score += 0.3
            reasons.append(f"positive pre-earnings sentiment ({sentiment:.2f})")
        elif sentiment < -0.2:
            score -= 0.3
            reasons.append(f"negative pre-earnings sentiment ({sentiment:.2f})")
        else:
            reasons.append(f"neutral sentiment ({sentiment:.2f})")

        # Price momentum confirmation: if momentum aligns with sentiment, stronger signal
        if (mom_z > 0.3 and sentiment > 0) or (mom_z < -0.3 and sentiment < 0):
            score += 0.2
            reasons.append(f"price momentum confirms sentiment (mom={mom_z:.2f})")
        elif (mom_z < -0.3 and sentiment > 0.1) or (mom_z > 0.3 and sentiment < -0.1):
            score *= 0.6
            reasons.append(f"momentum diverges from sentiment — risky")

        # MACD trend alignment
        if (macd_z > 0.3 and score > 0) or (macd_z < -0.3 and score < 0):
            score += 0.1
            reasons.append("MACD confirms direction")

        # Volume surge = smart money positioning pre-earnings
        if volume_z > 1.0:
            score *= 1.25
            reasons.append(f"unusual volume ({volume_z:.1f}σ) — institutional positioning")

        # RSI overbought/oversold check — don't chase extremes
        if rsi_z > 1.5 and score > 0:
            score *= 0.6
            reasons.append(f"RSI stretched ({rsi_z:.2f}), late entry risk")
        elif rsi_z < -1.5 and score < 0:
            score *= 0.6
            reasons.append(f"RSI deeply oversold ({rsi_z:.2f}), may bounce")

        if score > 0.2:
            action = "LONG"
            confidence = min(0.8, 0.5 + score)
        elif score < -0.2:
            action = "SHORT"
            confidence = min(0.7, 0.4 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons),
            "strategy": self.name,
        }
