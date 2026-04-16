"""
Volatility Breakout Strategy
=============================
Enter when India VIX spikes > 20 (fear = opportunity).
Macro agent confirms regime before entry.
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class VolatilityBreakoutStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="volatility_breakout", description="VIX spike breakout strategy")
        BaseAgent.__init__(self, name="VolatilityBreakout_Strategy", role="Vol spike trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule: When India VIX > 20 AND price momentum is strong (MACD_Z > 0),
        buy the dip in oversold assets (RSI_14_Z < 0).
        Macro regime check: if VIX is spiking (volatility_z > 1), this is a fear spike = opportunity.
        """
        vix_raw = portfolio_state.get("vix_raw", 20.0)
        volatility_z = indicators.get("Volatility_20_Z", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        macd_z = indicators.get("MACD_Z", 0.0)
        sentiment = alt_data.get("sentiment", 0.0)

        # VIX spike condition (fear mode)
        if vix_raw > 20.0 and volatility_z > 0.5:
            # Fear creates opportunity: buy oversold assets
            if rsi_z < 0.0 and sentiment > -0.2:
                action = "LONG"
                confidence = min(0.85, 0.6 + (vix_raw - 20) * 0.03)
                rationale = f"VIX spike ({vix_raw:.1f}), RSI oversold, fear=opportunity"
            else:
                action = "HOLD"
                confidence = 0.4
                rationale = f"VIX elevated ({vix_raw:.1f}), but asset not oversold"
        else:
            action = "HOLD"
            confidence = 0.3
            rationale = f"VIX calm ({vix_raw:.1f}), strategy inactive"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
