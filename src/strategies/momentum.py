"""
Momentum Strategy
=================
Multi-signal momentum: 12m price momentum + trend strength (ADX) + volume
confirmation + Bollinger breakout. Sentiment filter suppresses entries on
negative news.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


class MomentumStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="momentum", description="Multi-signal momentum with ADX and volume")
        BaseAgent.__init__(self, name="Momentum_Strategy", role="Trend momentum trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        bb_pos = indicators.get("BB_Position_Z", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        macd_z = indicators.get("MACD_Z", 0.0)
        mom_12m = indicators.get("mom_12m_Z", 0.0)
        adx = indicators.get("ADX_14", 20.0)
        vol_z = indicators.get("Volume_Z", 0.0)
        sentiment = alt_data.get("sentiment", 0.0)
        news_volume = alt_data.get("news_volume", 0)

        score = 0.0
        reasons = []

        # 12-month momentum (strongest alpha signal)
        if mom_12m > 0.5:
            score += 0.3
            reasons.append(f"strong 12m momentum ({mom_12m:.2f})")
        elif mom_12m < -0.5:
            score -= 0.3
            reasons.append(f"weak 12m momentum ({mom_12m:.2f})")

        # Price near 52-week high (Bollinger breakout)
        if bb_pos > 0.8:
            score += 0.2
            reasons.append("price near 52w high")
        elif bb_pos < 0.2:
            score -= 0.2
            reasons.append("price near 52w low")

        # Trend confirmation via MACD
        if macd_z > 0.3:
            score += 0.15
            reasons.append(f"MACD bullish ({macd_z:.2f})")
        elif macd_z < -0.3:
            score -= 0.15
            reasons.append(f"MACD bearish ({macd_z:.2f})")

        # ADX trend strength filter — only trade when trend is strong
        if adx > 25:
            score *= 1.3
            reasons.append(f"strong trend (ADX={adx:.0f})")
        elif adx < 15:
            score *= 0.5
            reasons.append(f"weak trend (ADX={adx:.0f}), reduced")

        # Volume confirmation — high volume validates the move
        if vol_z > 1.0:
            score *= 1.2
            reasons.append("volume surge confirming")
        elif vol_z < -1.0:
            score *= 0.7
            reasons.append("low volume, suspect move")

        # RSI overbought filter — don't chase overbought
        if rsi_z > 1.5 and score > 0:
            score *= 0.6
            reasons.append(f"RSI overbought ({rsi_z:.2f}), trimmed")

        # Sentiment filter
        if sentiment < -0.3 and news_volume > 0:
            score *= 0.4
            reasons.append(f"negative sentiment ({sentiment:.2f})")

        if score > 0.25:
            action = "LONG"
            confidence = min(0.9, 0.5 + score)
        elif score < -0.25:
            action = "SHORT"
            confidence = min(0.7, 0.4 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons) if reasons else "no momentum signals",
            "strategy": self.name,
        }
