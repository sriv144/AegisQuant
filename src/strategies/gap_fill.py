"""
Gap Fill Strategy
=================
Trades overnight gaps that are likely noise (no fundamental driver). Uses
volatility z-score as gap proxy + news volume filter. Works on any liquid stock,
not just ETFs.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


class GapFillStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="gap_fill", description="Overnight gap fill on noise gaps")
        BaseAgent.__init__(self, name="GapFill_Strategy", role="Gap fill trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        vol_z = indicators.get("Volatility_20_Z", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        bb_pos = indicators.get("BB_Position_Z", 0.0)
        volume_z = indicators.get("Volume_Z", 0.0)
        adx = indicators.get("ADX_14", 20.0)
        sentiment = alt_data.get("sentiment", 0.0)
        news_volume = alt_data.get("news_volume", 0)

        score = 0.0
        reasons = []

        # Gap detection: high short-term volatility = likely gap
        if vol_z < 0.8:
            return {
                "action": "HOLD",
                "confidence": 0.2,
                "rationale": f"low volatility ({vol_z:.2f}), no gap detected",
                "strategy": self.name,
            }

        reasons.append(f"gap detected (vol_z={vol_z:.2f})")

        # Key filter: was the gap on news or noise?
        if news_volume >= 3:
            score *= 0.3
            reasons.append(f"fundamental gap (news_vol={news_volume}), skip fill")
            return {
                "action": "HOLD",
                "confidence": 0.25,
                "rationale": " | ".join(reasons),
                "strategy": self.name,
            }
        elif news_volume == 0:
            score += 0.2
            reasons.append("no news — noise gap, likely to fill")
        else:
            reasons.append(f"low news ({news_volume}), possible noise gap")

        # Direction of the gap: oversold gap-down = buy, overbought gap-up = sell
        if rsi_z < -0.5:
            score += 0.3
            reasons.append(f"gap-down, RSI oversold ({rsi_z:.2f}) — expect fill up")
        elif rsi_z > 0.5:
            score -= 0.3
            reasons.append(f"gap-up, RSI elevated ({rsi_z:.2f}) — expect fill down")

        # BB position extreme confirms the gap
        if bb_pos < -0.3:
            score += 0.15
            reasons.append("below Bollinger band (gap-down confirmation)")
        elif bb_pos > 1.3:
            score -= 0.15
            reasons.append("above Bollinger band (gap-up confirmation)")

        # Volume filter: low volume gaps fill faster (no conviction behind the move)
        if volume_z < 0:
            score *= 1.2
            reasons.append("low-volume gap, higher fill probability")
        elif volume_z > 1.5:
            score *= 0.6
            reasons.append("high-volume gap, may be real breakout")

        # Trend filter: gaps in range-bound markets fill more reliably
        if adx > 25:
            score *= 0.6
            reasons.append(f"trending market (ADX={adx:.0f}), gap may not fill")
        elif adx < 18:
            score *= 1.2
            reasons.append(f"range-bound (ADX={adx:.0f}), fill likely")

        if score > 0.2:
            action = "LONG"
            confidence = min(0.75, 0.5 + score)
        elif score < -0.2:
            action = "SHORT"
            confidence = min(0.65, 0.4 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons),
            "strategy": self.name,
        }
