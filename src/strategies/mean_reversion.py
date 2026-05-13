"""
Mean Reversion Strategy
=======================
Multi-signal mean reversion: RSI extremes + Bollinger band deviation + volume
divergence. Drawdown-aware — reduces exposure in high-drawdown regimes.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


class MeanReversionStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="mean_reversion", description="RSI + Bollinger band mean reversion")
        BaseAgent.__init__(self, name="MeanReversion_Strategy", role="Mean reversion trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        bb_pos = indicators.get("BB_Position_Z", 0.0)
        macd_z = indicators.get("MACD_Z", 0.0)
        vol_z = indicators.get("Volatility_20_Z", 0.0)
        volume_z = indicators.get("Volume_Z", 0.0)
        adx = indicators.get("ADX_14", 20.0)
        drawdown = portfolio_state.get("current_drawdown", 0.0)

        score = 0.0
        reasons = []

        # RSI extremes (primary signal)
        if rsi_z < -1.0:
            score += 0.35
            reasons.append(f"RSI oversold ({rsi_z:.2f})")
        elif rsi_z < -1.5:
            score += 0.5
            reasons.append(f"RSI deeply oversold ({rsi_z:.2f})")
        elif rsi_z > 1.5:
            score -= 0.35
            reasons.append(f"RSI overbought ({rsi_z:.2f})")
        elif rsi_z > 2.0:
            score -= 0.5
            reasons.append(f"RSI extremely overbought ({rsi_z:.2f})")

        # Bollinger band deviation (confirmation)
        if bb_pos < -0.5:
            score += 0.2
            reasons.append("below lower Bollinger band")
        elif bb_pos > 1.5:
            score -= 0.2
            reasons.append("above upper Bollinger band")

        # Volume divergence: high volume at extremes strengthens reversion signal
        if abs(rsi_z) > 1.0 and volume_z > 1.5:
            score *= 1.3
            reasons.append("volume spike at extreme — capitulation likely")

        # Trend filter — mean reversion works poorly in strong trends
        if adx > 30:
            score *= 0.5
            reasons.append(f"strong trend (ADX={adx:.0f}), reversion risky")
        elif adx < 20:
            score *= 1.2
            reasons.append(f"range-bound (ADX={adx:.0f}), reversion favored")

        # MACD divergence from price: if RSI oversold but MACD turning up, stronger signal
        if rsi_z < -1.0 and macd_z > -0.3:
            score += 0.15
            reasons.append("MACD divergence: turning up while price oversold")

        # Drawdown risk scaling
        if drawdown > 0.15:
            score *= 0.4
            reasons.append(f"high drawdown ({drawdown*100:.1f}%), sharply reduced")
        elif drawdown > 0.08:
            score *= 0.7
            reasons.append(f"elevated drawdown ({drawdown*100:.1f}%), reduced")

        # Elevated volatility increases reversion potential but also risk
        if vol_z > 1.5:
            score *= 0.8
            reasons.append("high vol environment, tighter sizing")

        if score > 0.25:
            action = "LONG"
            confidence = min(0.85, 0.5 + score)
        elif score < -0.25:
            action = "SHORT"
            confidence = min(0.75, 0.4 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons) if reasons else "no mean reversion signals",
            "strategy": self.name,
        }
