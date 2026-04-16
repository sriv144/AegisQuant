"""
Sector Rotation Strategy
=======================
Rotate into outperforming sectors monthly.
LLM reads macro signals (RBI policy, monsoon, oil prices).
"""

import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

class SectorRotationStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="sector_rotation", description="Macro-driven sector rotation")
        BaseAgent.__init__(self, name="SectorRotation_Strategy", role="Sector analyst")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule: Assign stocks to sectors, rotate based on macro environment.
        Simplified logic:
        - Tech/Bank stocks (INFY, TCS, HDFCBANK): rotate in when inflation is low (opposite of VIX)
        - Auto/Infra (LT): rotate in when economic growth outlook is positive
        - FMCG (HINDUNILVR, ITC): defensive, rotate in when VIX high (risk-off)
        """
        ticker_sym = ticker.split(".")[0] if "." in ticker else ticker
        vix_raw = portfolio_state.get("vix_raw", 20.0)
        sentiment = alt_data.get("sentiment", 0.0)

        # Sector mapping
        tech_stocks = ["INFY", "TCS"]
        bank_stocks = ["HDFCBANK", "ICICIBANK", "KOTAKBANK"]
        auto_infra_stocks = ["LT"]  # Larsen & Toubro
        fmcg_stocks = ["HINDUNILVR", "ITC"]

        # Sector rotation logic
        if ticker_sym in tech_stocks + bank_stocks:
            # Growth sectors: like low VIX + positive sentiment
            if vix_raw < 18.0 and sentiment > 0.0:
                action = "LONG"
                confidence = min(0.8, 0.5 + (20 - vix_raw) * 0.05)
                rationale = f"Growth sector in low-VIX ({vix_raw:.1f}) environment"
            else:
                action = "HOLD"
                confidence = 0.4
                rationale = "Growth sector; waiting for lower volatility"

        elif ticker_sym in auto_infra_stocks:
            # Cyclical: rotate in on positive macro
            if sentiment > 0.1 and vix_raw < 25.0:
                action = "LONG"
                confidence = min(0.75, 0.5 + sentiment * 0.25)
                rationale = f"Cyclical sector, positive macro outlook (sentiment={sentiment:.2f})"
            else:
                action = "HOLD"
                confidence = 0.3
                rationale = "Cyclical; waiting for clearer macro signals"

        elif ticker_sym in fmcg_stocks:
            # Defensive: rotate in on high VIX (risk-off)
            if vix_raw > 20.0 or sentiment < 0.0:
                action = "LONG"
                confidence = min(0.7, 0.5 + (vix_raw - 20) * 0.03)
                rationale = f"Defensive sector in risk-off mode (VIX={vix_raw:.1f})"
            else:
                action = "HOLD"
                confidence = 0.3
                rationale = "Defensive; taking risk-on posture"

        else:
            action = "HOLD"
            confidence = 0.3
            rationale = "Sector rotation strategy target not identified"

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": rationale,
            "strategy": self.name,
        }
