"""
Factor Investing Strategy
=========================
Multi-factor scoring: momentum factor (12m return) + quality factor (low
volatility) + value factor (RSI mean-reversion) + sentiment factor.
Combines factors into a composite score with regime-adaptive weighting.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


class FactorInvestingStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="factor_investing", description="Multi-factor: momentum + quality + value + sentiment")
        BaseAgent.__init__(self, name="FactorInvesting_Strategy", role="Factor analyst")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        mom_z = indicators.get("mom_12m_Z", 0.0)
        vol_z = indicators.get("Volatility_20_Z", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        bb_pos = indicators.get("BB_Position_Z", 0.0)
        sentiment = alt_data.get("sentiment", 0.0)
        news_volume = alt_data.get("news_volume", 0)
        vix_raw = portfolio_state.get("vix_raw", 20.0)

        reasons = []

        # Momentum factor: positive 12m momentum is historically the strongest alpha source
        mom_score = 0.0
        if mom_z > 0.5:
            mom_score = min(0.4, mom_z * 0.25)
            reasons.append(f"momentum+ ({mom_z:.2f})")
        elif mom_z < -0.5:
            mom_score = max(-0.4, mom_z * 0.25)
            reasons.append(f"momentum- ({mom_z:.2f})")

        # Quality factor: low volatility stocks tend to outperform risk-adjusted
        quality_score = 0.0
        if vol_z < -0.5:
            quality_score = 0.2
            reasons.append(f"low-vol quality ({vol_z:.2f})")
        elif vol_z > 1.0:
            quality_score = -0.15
            reasons.append(f"high-vol ({vol_z:.2f})")

        # Value factor: oversold stocks with positive momentum divergence
        value_score = 0.0
        if rsi_z < -1.0 and mom_z > 0:
            value_score = 0.25
            reasons.append(f"value: oversold RSI + positive momentum divergence")
        elif rsi_z > 1.5 and mom_z < 0:
            value_score = -0.2
            reasons.append(f"overvalued: overbought RSI + fading momentum")

        # Sentiment factor
        sent_score = 0.0
        if news_volume > 0:
            sent_score = sentiment * 0.15
            if abs(sentiment) > 0.2:
                reasons.append(f"sentiment {'positive' if sentiment > 0 else 'negative'} ({sentiment:.2f})")

        # Regime-adaptive weighting: in high-VIX, overweight quality; in low-VIX, overweight momentum
        if vix_raw > 22:
            weights = {"mom": 0.2, "quality": 0.4, "value": 0.25, "sent": 0.15}
            reasons.append(f"risk-off regime (VIX={vix_raw:.0f}), favoring quality")
        elif vix_raw < 15:
            weights = {"mom": 0.45, "quality": 0.15, "value": 0.25, "sent": 0.15}
            reasons.append(f"risk-on regime (VIX={vix_raw:.0f}), favoring momentum")
        else:
            weights = {"mom": 0.3, "quality": 0.25, "value": 0.25, "sent": 0.2}

        composite = (
            weights["mom"] * mom_score +
            weights["quality"] * quality_score +
            weights["value"] * value_score +
            weights["sent"] * sent_score
        )

        # Normalize composite to [-1, 1] range for action thresholds
        composite = max(-1.0, min(1.0, composite * 3))

        if composite > 0.2:
            action = "LONG"
            confidence = min(0.85, 0.5 + composite * 0.4)
        elif composite < -0.2:
            action = "SHORT"
            confidence = min(0.7, 0.4 + abs(composite) * 0.3)
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons) if reasons else "factors neutral",
            "strategy": self.name,
        }
