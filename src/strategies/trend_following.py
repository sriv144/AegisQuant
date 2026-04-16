"""
Trend Following Strategy
========================
EMA 20/50 crossover. Pauses when India VIX > 20 (high volatility environment).
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class TrendFollowingStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="trend_following", description="EMA 20/50 crossover with VIX filter")
        BaseAgent.__init__(self, name="TrendFollowing_Strategy", role="Trend follower")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Approximation: Use MACD_Z as proxy for EMA crossover signal.
        MACD_Z > 0.5 → LONG, MACD_Z < -0.5 → SHORT.
        VIX filter: If India VIX > 20, reduce confidence (high volatility regime).
        """
        macd_z = indicators.get("MACD_Z", 0.0)
        volatility = indicators.get("Volatility_20_Z", 0.0)
        vix_raw = portfolio_state.get("vix_raw", 20.0)

        # Strong uptrend
        if macd_z > 0.5:
            action = "LONG"
            confidence = min(0.9, 0.6 + macd_z * 0.2)
            rationale = f"MACD positive ({macd_z:.2f}), uptrend"

            # VIX filter: high vol reduces confidence
            if vix_raw > 20.0:
                confidence *= 0.75
                rationale += f" | high VIX ({vix_raw:.1f}), reduced confidence"

        # Strong downtrend
        elif macd_z < -0.5:
            action = "SHORT"
            confidence = min(0.7, 0.5 + abs(macd_z) * 0.15)
            rationale = f"MACD negative ({macd_z:.2f}), downtrend"

            if vix_raw > 20.0:
                confidence *= 0.8
                rationale += f" | high VIX, cautious"

        # Flat/mean-reverting zone
        else:
            action = "HOLD"
            confidence = 0.5
            rationale = "MACD in neutral zone, no clear trend"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
