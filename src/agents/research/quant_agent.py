"""
Quant Agent
===========
Replaces dummy stubs with real logical processing for the RL environment to consume.
Extracts 'quant_confidence' from `feature_engineering.py`.
"""
import pandas as pd
from typing import Dict, Any


def _score_to_action(score: float) -> str:
    if score >= 0.15:
        return "PROPOSE_LONG"
    if score <= -0.15:
        return "PROPOSE_SHORT"
    return "HOLD"

class QuantAgent:
    def __init__(self):
        self.name = "Quant_Research_Agent"
        
    def analyze(self, current_features: pd.Series) -> Dict[str, Any]:
        """
        Takes the current day's feature vector and outputs a directional confidence score.
        Confidence is mapped to [-1, 1].
        """
        # We look at moving averages, RSI divergence, and Volatility positioning.
        
        rsi_z = current_features.get("RSI_14_Z", 0.0)
        macd_z = current_features.get("MACD_Z", 0.0)
        mom_z = current_features.get("mom_12m_Z", 0.0)
        
        # Simple logical rules to represent an algorithmic "agent"
        score = 0.0
        
        # Momentum
        if mom_z > 1.0:
            score += 0.3
        elif mom_z < -1.0:
            score -= 0.3
            
        # Mean Reversion (RSI heavily overbought/oversold limits)
        if rsi_z > 2.0:
            score -= 0.4
        elif rsi_z < -2.0:
            score += 0.4
            
        # MACD alignment
        if macd_z > 0.5:
            score += 0.2
        elif macd_z < -0.5:
            score -= 0.2
            
        # Clip to boundaries
        score = max(-1.0, min(1.0, score))
        
        return {
            "confidence": score,
            "agent_name": self.name,
            "rationale": f"RSI_z={rsi_z:.2f}, MACD_z={macd_z:.2f}, Mom_z={mom_z:.2f}"
        }

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_features = pd.Series(state.get("technical_indicators", {}), dtype="float64")
        decision = self.analyze(current_features)
        signed_confidence = float(decision["confidence"])

        return {
            "research_signals": [{
                "agent_name": self.name,
                "action": _score_to_action(signed_confidence),
                "confidence": round(abs(signed_confidence), 4),
                "rationale": decision["rationale"],
            }]
        }


quant_agent = QuantAgent()
