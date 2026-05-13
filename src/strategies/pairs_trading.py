"""
Pairs Trading Strategy
======================
Relative-value mean reversion within sector peer groups. Trades stocks that
deviate from their sector's average RSI/momentum. Works across all stocks in
the universe, not just hardcoded pairs.
"""

from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent


SECTOR_MAP = {
    "RELIANCE": "energy", "ONGC": "energy", "BPCL": "energy",
    "HDFCBANK": "banking", "ICICIBANK": "banking", "KOTAKBANK": "banking",
    "SBIN": "banking", "AXISBANK": "banking", "INDUSINDBK": "banking",
    "INFY": "it", "TCS": "it", "WIPRO": "it", "HCLTECH": "it", "TECHM": "it",
    "HINDUNILVR": "fmcg", "ITC": "fmcg", "NESTLEIND": "fmcg", "BRITANNIA": "fmcg",
    "LT": "infra", "ADANIENT": "infra", "ULTRACEMCO": "infra",
    "BHARTIARTL": "telecom", "IDEA": "telecom",
    "SUNPHARMA": "pharma", "DRREDDY": "pharma", "CIPLA": "pharma",
    "TATAMOTORS": "auto", "MARUTI": "auto", "M&M": "auto",
    "BAJFINANCE": "nbfc", "BAJAJFINSV": "nbfc",
}


class PairsTradingStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="pairs_trading", description="Sector-relative mean reversion")
        BaseAgent.__init__(self, name="PairsTrading_Strategy", role="Pairs trader")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        ticker_sym = ticker.replace(".NS", "").replace(".BO", "")
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        bb_pos = indicators.get("BB_Position_Z", 0.0)
        mom_z = indicators.get("mom_12m_Z", 0.0)
        vol_z = indicators.get("Volatility_20_Z", 0.0)
        adx = indicators.get("ADX_14", 20.0)

        sector = SECTOR_MAP.get(ticker_sym)
        if not sector:
            return {
                "action": "HOLD",
                "confidence": 0.2,
                "rationale": f"{ticker_sym} not mapped to a sector peer group",
                "strategy": self.name,
            }

        score = 0.0
        reasons = [f"sector: {sector}"]

        # Relative-value signal: stock deviates from "fair value" within sector
        # RSI extreme = stock diverged from sector mean
        if rsi_z < -1.2:
            score += 0.35
            reasons.append(f"RSI oversold vs sector ({rsi_z:.2f})")
        elif rsi_z > 1.2:
            score -= 0.35
            reasons.append(f"RSI overbought vs sector ({rsi_z:.2f})")

        # BB position extreme = price at statistical extreme
        if bb_pos < -0.8:
            score += 0.2
            reasons.append("below lower Bollinger (sector underperformer)")
        elif bb_pos > 1.8:
            score -= 0.2
            reasons.append("above upper Bollinger (sector outperformer)")

        # Momentum divergence: if stock momentum diverges from overall market
        if abs(mom_z) > 0.8:
            if mom_z < -0.8 and rsi_z < -0.5:
                score += 0.15
                reasons.append("lagging sector peer — reversion candidate")
            elif mom_z > 0.8 and rsi_z > 0.5:
                score -= 0.15
                reasons.append("outperforming sector peer — reversion risk")

        # Mean reversion works in range-bound markets, not trending ones
        if adx > 30:
            score *= 0.5
            reasons.append(f"strong trend (ADX={adx:.0f}), pairs risky")
        elif adx < 18:
            score *= 1.3
            reasons.append(f"range-bound (ADX={adx:.0f}), pairs favorable")

        # High volatility = wider spreads = more reversion opportunity
        if vol_z > 1.0:
            score *= 1.15
            reasons.append("elevated vol, wider reversion band")

        if score > 0.2:
            action = "LONG"
            confidence = min(0.8, 0.5 + score)
        elif score < -0.2:
            action = "SHORT"
            confidence = min(0.7, 0.4 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons),
            "strategy": self.name,
        }
