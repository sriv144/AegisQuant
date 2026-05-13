"""
Trend Following Strategy
========================
EMA crossover (SMA10 vs SMA50) + MACD momentum + ADX trend strength.
VIX filter pauses entries in extreme volatility. Uses ATR for position sizing.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


class TrendFollowingStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="trend_following", description="EMA crossover + ADX + VIX filter")
        BaseAgent.__init__(self, name="TrendFollowing_Strategy", role="Trend follower")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        macd_z = indicators.get("MACD_Z", 0.0)
        adx = indicators.get("ADX_14", 20.0)
        vol_z = indicators.get("Volatility_20_Z", 0.0)
        bb_pos = indicators.get("BB_Position_Z", 0.0)
        mom_z = indicators.get("mom_12m_Z", 0.0)
        vix_raw = portfolio_state.get("vix_raw", 20.0)

        # Try to get raw SMA crossover (if available in indicators from OHLCV)
        sma_10 = indicators.get("SMA_10", 0.0)
        sma_50 = indicators.get("SMA_50", 0.0)
        close = indicators.get("close", 0.0)

        score = 0.0
        reasons = []

        # SMA crossover: price above SMA50 = uptrend, below = downtrend
        if close > 0 and sma_50 > 0:
            if close > sma_50 * 1.02:
                score += 0.25
                reasons.append("price >2% above SMA50")
            elif close < sma_50 * 0.98:
                score -= 0.25
                reasons.append("price >2% below SMA50")

            if sma_10 > 0 and sma_10 > sma_50:
                score += 0.15
                reasons.append("SMA10 > SMA50 (golden cross)")
            elif sma_10 > 0 and sma_10 < sma_50:
                score -= 0.15
                reasons.append("SMA10 < SMA50 (death cross)")

        # MACD trend confirmation
        if macd_z > 0.5:
            score += 0.2
            reasons.append(f"MACD bullish ({macd_z:.2f})")
        elif macd_z < -0.5:
            score -= 0.2
            reasons.append(f"MACD bearish ({macd_z:.2f})")

        # 12-month momentum alignment
        if (mom_z > 0.3 and score > 0) or (mom_z < -0.3 and score < 0):
            score *= 1.2
            reasons.append(f"12m momentum confirms ({mom_z:.2f})")
        elif (mom_z < -0.3 and score > 0) or (mom_z > 0.3 and score < 0):
            score *= 0.6
            reasons.append(f"12m momentum diverges ({mom_z:.2f}), caution")

        # ADX trend strength — only trade strong trends
        if adx > 25:
            score *= 1.3
            reasons.append(f"strong trend (ADX={adx:.0f})")
        elif adx < 15:
            score *= 0.3
            reasons.append(f"no trend (ADX={adx:.0f}), sitting out")

        # VIX filter: extreme VIX kills trend-following
        if vix_raw > 28:
            score *= 0.4
            reasons.append(f"extreme VIX ({vix_raw:.0f}), risk-off")
        elif vix_raw > 22:
            score *= 0.7
            reasons.append(f"elevated VIX ({vix_raw:.0f}), cautious")

        if score > 0.2:
            action = "LONG"
            confidence = min(0.9, 0.5 + score)
        elif score < -0.2:
            action = "SHORT"
            confidence = min(0.75, 0.4 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons) if reasons else "no trend signal",
            "strategy": self.name,
        }
