"""
Factor Investing Strategy
=========================
Long low-P/E, high-ROE NIFTY 50 stocks.
LLM re-screens fundamentals quarterly using earnings call sentiment.
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class FactorInvestingStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="factor_investing", description="Low-P/E + high-ROE factor strategy")
        BaseAgent.__init__(self, name="FactorInvesting_Strategy", role="Factor analyst")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule: Use market_data yfinance fundamental info (if available).
        For now, simple rule: If ticker is in NIFTY 50 and sentiment >= 0, consider LONG.
        Actual quarterly fundamental re-screening would require LLM call on earnings transcripts.
        """
        sentiment = alt_data.get("sentiment", 0.0)
        news_volume = alt_data.get("news_volume", 0)

        nifty_50_stocks = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "INFY.NS", "HINDUNILVR.NS", "BHARTIARTL.NS", "ITC.NS",
            "KOTAKBANK.NS", "LT.NS"
        ]

        is_nifty_stock = ticker in nifty_50_stocks

        if is_nifty_stock:
            # NIFTY 50 stock with positive sentiment = quality factor signal
            if sentiment >= 0.1 and news_volume > 0:
                action = "LONG"
                confidence = min(0.75, 0.5 + sentiment * 0.25)
                rationale = f"NIFTY 50 quality stock, positive sentiment ({sentiment:.2f})"
            else:
                action = "HOLD"
                confidence = 0.5
                rationale = "NIFTY 50 stock, but sentiment neutral/negative"
        else:
            action = "HOLD"
            confidence = 0.3
            rationale = "Not a NIFTY 50 stock; strategy focuses on blue-chips"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
