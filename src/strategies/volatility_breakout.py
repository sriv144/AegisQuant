"""
Volatility Breakout Strategy
=============================
Buys into VIX spikes (fear = opportunity). Requires capitulation signals:
volume surge + RSI oversold + elevated VIX. Exits when VIX normalizes.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


class VolatilityBreakoutStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="volatility_breakout", description="VIX spike capitulation buying")
        BaseAgent.__init__(self, name="VolatilityBreakout_Strategy", role="Vol spike trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        vix_raw = portfolio_state.get("vix_raw", 20.0)
        vol_z = indicators.get("Volatility_20_Z", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        bb_pos = indicators.get("BB_Position_Z", 0.0)
        volume_z = indicators.get("Volume_Z", 0.0)
        macd_z = indicators.get("MACD_Z", 0.0)
        sentiment = alt_data.get("sentiment", 0.0)
        drawdown = portfolio_state.get("current_drawdown", 0.0)

        score = 0.0
        reasons = []

        # VIX regime classification
        if vix_raw < 16:
            return {
                "action": "HOLD",
                "confidence": 0.2,
                "rationale": f"VIX calm ({vix_raw:.1f}), vol breakout strategy inactive",
                "strategy": self.name,
            }

        if vix_raw > 25:
            score += 0.2
            reasons.append(f"VIX spike ({vix_raw:.1f}), fear regime")
        if vix_raw > 30:
            score += 0.15
            reasons.append("extreme fear zone")

        # Capitulation detection: volume surge + oversold RSI = panic selling
        if volume_z > 1.5 and rsi_z < -1.0:
            score += 0.3
            reasons.append(f"capitulation: volume surge ({volume_z:.1f}) + RSI oversold ({rsi_z:.2f})")
        elif volume_z > 1.0 and rsi_z < -0.5:
            score += 0.15
            reasons.append("mild capitulation signals")

        # Bollinger band breach = statistical extreme
        if bb_pos < -0.5:
            score += 0.15
            reasons.append("below Bollinger lower band")

        # MACD early reversal signal: if MACD turns up from deep negative, momentum shifting
        if macd_z > -0.5 and macd_z < 0.3 and vol_z > 1.0:
            score += 0.1
            reasons.append("MACD turning up from bottom")

        # Sentiment check: extreme negative sentiment + high VIX = peak fear (contrarian buy)
        if sentiment < -0.3:
            score += 0.1
            reasons.append(f"extreme negative sentiment ({sentiment:.2f}) — contrarian")
        elif sentiment > 0.2:
            score -= 0.15
            reasons.append("sentiment not fearful enough for vol strategy")

        # Portfolio drawdown circuit breaker
        if drawdown > 0.12:
            score *= 0.5
            reasons.append(f"high portfolio drawdown ({drawdown*100:.1f}%), reduced size")

        if score > 0.3:
            action = "LONG"
            confidence = min(0.85, 0.5 + score)
        elif score < -0.1:
            action = "SHORT"
            confidence = min(0.6, 0.3 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons) if reasons else "VIX elevated but no capitulation",
            "strategy": self.name,
        }
