"""
Sector Rotation Strategy
========================
Rotates into sectors favored by the current macro regime (VIX level + momentum).
Growth sectors in low-VIX; defensive sectors in high-VIX. Uses per-stock
momentum + sentiment to pick the best within each sector.

Supports both US and India markets via MARKET env var.
"""

import os
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
from src.agents.base_agent import BaseAgent

_MARKET = os.getenv("MARKET", "US").upper()

# ── US Sector Regime ─────────────────────────────────────────────────────────
SECTOR_REGIME_US = {
    "growth": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "CRM",
                     "ADBE", "AMD", "NOW", "PANW", "SNPS"],
        "favored_when": "low_vix",
    },
    "cyclical": {
        "tickers": ["CAT", "BA", "HON", "DE", "GE", "UNP", "TSLA"],
        "favored_when": "low_vix",
    },
    "defensive": {
        "tickers": ["PG", "KO", "PEP", "JNJ", "WMT", "MCD", "CL",
                     "MRK", "ABBV", "UNH"],
        "favored_when": "high_vix",
    },
    "energy": {
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "favored_when": "neutral",
    },
    "finance": {
        "tickers": ["JPM", "BAC", "GS", "MS", "BRK-B", "V", "MA"],
        "favored_when": "neutral",
    },
}

# ── India Sector Regime ──────────────────────────────────────────────────────
SECTOR_REGIME_IN = {
    "growth": {
        "tickers": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM",
                     "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK",
                     "BAJFINANCE", "BAJAJFINSV"],
        "favored_when": "low_vix",
    },
    "cyclical": {
        "tickers": ["LT", "ADANIENT", "ULTRACEMCO", "TATAMOTORS", "MARUTI", "M&M"],
        "favored_when": "low_vix",
    },
    "defensive": {
        "tickers": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
                     "SUNPHARMA", "DRREDDY", "CIPLA"],
        "favored_when": "high_vix",
    },
    "energy": {
        "tickers": ["RELIANCE", "ONGC", "BPCL"],
        "favored_when": "neutral",
    },
    "telecom": {
        "tickers": ["BHARTIARTL"],
        "favored_when": "neutral",
    },
}

SECTOR_REGIME = SECTOR_REGIME_US if _MARKET == "US" else SECTOR_REGIME_IN


def _get_sector(ticker_sym: str):
    for sector, info in SECTOR_REGIME.items():
        if ticker_sym in info["tickers"]:
            return sector, info["favored_when"]
    return None, None


class SectorRotationStrategy(BaseStrategy, BaseAgent):
    def __init__(self):
        BaseStrategy.__init__(self, name="sector_rotation", description="Macro-regime sector rotation")
        BaseAgent.__init__(self, name="SectorRotation_Strategy", role="Sector analyst")

    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        ticker_sym = ticker.replace(".NS", "").replace(".BO", "").replace(".US", "")
        vix_raw = portfolio_state.get("vix_raw", 20.0)
        sentiment = alt_data.get("sentiment", 0.0)
        mom_z = indicators.get("mom_12m_Z", 0.0)
        rsi_z = indicators.get("RSI_14_Z", 0.0)
        vol_z = indicators.get("Volatility_20_Z", 0.0)
        macd_z = indicators.get("MACD_Z", 0.0)

        sector, favored_when = _get_sector(ticker_sym)
        if not sector:
            return {
                "action": "HOLD",
                "confidence": 0.2,
                "rationale": f"{ticker_sym} not in sector rotation universe",
                "strategy": self.name,
            }

        # Determine macro regime from VIX
        if vix_raw < 16:
            regime = "low_vix"
            regime_label = f"risk-on (VIX={vix_raw:.0f})"
        elif vix_raw > 22:
            regime = "high_vix"
            regime_label = f"risk-off (VIX={vix_raw:.0f})"
        else:
            regime = "neutral"
            regime_label = f"neutral (VIX={vix_raw:.0f})"

        score = 0.0
        reasons = [f"sector={sector}", regime_label]

        # Sector-regime alignment
        if favored_when == regime:
            score += 0.3
            reasons.append(f"{sector} favored in {regime} regime")
        elif favored_when == "neutral":
            score += 0.1
            reasons.append(f"{sector} regime-neutral")
        else:
            score -= 0.2
            reasons.append(f"{sector} not favored in {regime} regime")

        # Within-sector stock selection: momentum + sentiment + RSI
        if mom_z > 0.3:
            score += 0.2
            reasons.append(f"strong relative momentum ({mom_z:.2f})")
        elif mom_z < -0.3:
            score -= 0.15
            reasons.append(f"weak relative momentum ({mom_z:.2f})")

        if sentiment > 0.15:
            score += 0.1
            reasons.append(f"positive sentiment ({sentiment:.2f})")
        elif sentiment < -0.15:
            score -= 0.1
            reasons.append(f"negative sentiment ({sentiment:.2f})")

        # MACD trend confirmation
        if macd_z > 0.3 and score > 0:
            score += 0.1
            reasons.append("MACD confirms rotation")

        # RSI extreme filter: don't rotate into overbought or out of oversold
        if rsi_z > 1.5 and score > 0:
            score *= 0.6
            reasons.append(f"RSI stretched ({rsi_z:.2f}), late rotation")
        elif rsi_z < -1.0 and score < 0:
            score *= 0.7
            reasons.append(f"RSI oversold ({rsi_z:.2f}), may be value")

        # Low-vol stocks get a quality premium in rotation decisions
        if vol_z < -0.5:
            score += 0.1
            reasons.append("low-vol quality premium")

        if score > 0.2:
            action = "LONG"
            confidence = min(0.8, 0.5 + score)
        elif score < -0.15:
            action = "SHORT"
            confidence = min(0.65, 0.35 + abs(score))
        else:
            action = "HOLD"
            confidence = 0.3

        return {
            "action": action,
            "confidence": float(round(confidence, 4)),
            "rationale": " | ".join(reasons),
            "strategy": self.name,
        }
