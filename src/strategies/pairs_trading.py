"""
Pairs Trading Strategy
======================
Long/short correlated pairs (e.g., HDFCBANK.NS vs ICICIBANK.NS).
RL determines hedge ratio dynamically.
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class PairsTradingStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="pairs_trading", description="Long/short pairs trading (banks)")
        BaseAgent.__init__(self, name="PairsTrading_Strategy", role="Pairs trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Strategy: Trade HDFCBANK vs ICICIBANK (correlated bank stocks).
        If RSI diverges (one oversold, other overbought), we have a reversion opportunity.
        For simplicity, if this ticker is HDFCBANK and sentiment is positive relative to ICICI, LONG.
        """
        ticker_sym = ticker.split(".")[0] if "." in ticker else ticker
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        sentiment = alt_data.get("sentiment", 0.0)

        # Bank pair logic (simplified)
        if ticker_sym == "HDFCBANK":
            # If HDFCBANK RSI oversold and has good sentiment, long it (expecting reversion)
            if rsi_z < -0.5 and sentiment > -0.1:
                action = "LONG"
                confidence = min(0.7, 0.5 + abs(rsi_z) * 0.15)
                rationale = f"HDFCBANK oversold relative to sector, RSI={rsi_z:.2f}"
            elif rsi_z > 1.0 and sentiment < 0.1:
                action = "SHORT"
                confidence = min(0.6, 0.4 + rsi_z * 0.1)
                rationale = f"HDFCBANK overbought, RSI={rsi_z:.2f}"
            else:
                action = "HOLD"
                confidence = 0.4
                rationale = "HDFCBANK in neutral zone for pair trade"

        elif ticker_sym == "ICICIBANK":
            # Inverse logic for ICICI
            if rsi_z > 1.0 and sentiment > -0.1:
                action = "LONG"
                confidence = min(0.7, 0.5 + rsi_z * 0.1)
                rationale = f"ICICIBANK overbought, potential mean reversion pair"
            elif rsi_z < -0.5 and sentiment < 0.1:
                action = "SHORT"
                confidence = min(0.6, 0.4 + abs(rsi_z) * 0.15)
                rationale = f"ICICIBANK oversold relative to HDFCBANK"
            else:
                action = "HOLD"
                confidence = 0.4
                rationale = "ICICIBANK in neutral zone"

        else:
            action = "HOLD"
            confidence = 0.2
            rationale = "Pairs strategy focuses on bank pair: HDFCBANK vs ICICIBANK"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
