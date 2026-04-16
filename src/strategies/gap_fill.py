"""
Gap Fill Strategy
=================
Trade overnight gaps on index ETFs.
News agent validates if gap is fundamental or noise.
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class GapFillStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="gap_fill", description="Overnight gap fill trading on ETFs")
        BaseAgent.__init__(self, name="GapFill_Strategy", role="Gap fill trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule: Large overnight gap (approximated by high Volatility_20_Z) + mean-reverting bias.
        News sentiment filter: if gap is on positive/negative news, skip (fundamental gap).
        If gap on no news (noise), expect fill.
        """
        volatility_z = indicators.get("Volatility_20_Z", 0.0)
        sentiment = alt_data.get("sentiment", 0.0)
        news_volume = alt_data.get("news_volume", 0)

        # ETF gap fill strategy works best on index ETFs (NIFTYBEES, etc)
        is_etf = "BEES" in ticker.upper()

        if is_etf and volatility_z > 1.0:
            # Large vol spike (gap detected)
            if news_volume == 0:
                # Gap on no news = noise, expect fill
                action = "LONG" if sentiment < 0 else "SHORT"
                confidence = min(0.7, 0.5 + abs(volatility_z) * 0.1)
                rationale = f"Gap on low news volume, expect fill. Vol={volatility_z:.2f}"
            else:
                # Gap on news = maybe fundamental, be cautious
                action = "HOLD"
                confidence = 0.3
                rationale = f"Gap on news volume={news_volume}; fundamental gap, skip"
        else:
            action = "HOLD"
            confidence = 0.2
            rationale = "Gap fill strategy targets index ETFs; low volatility environment"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
