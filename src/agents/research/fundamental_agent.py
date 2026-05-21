import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState


def _fetch_etf_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch real ETF/equity characteristics from yfinance.
    For broad-market ETFs (SPY, QQQ, TLT, GLD) P/E-style ratios are less
    meaningful; we instead pull:
      - trailing_pe         (equities / equity ETFs)
      - price_to_book       (equities / equity ETFs)
      - 52w_high_pct        how far below 52-week high (momentum health)
      - avg_volume_ratio    today vol / 3-month avg vol (institutional interest)
      - ytd_return          year-to-date price return
    All values fall back to None if unavailable so the LLM can note missing data.
    """
    try:
        import yfinance as yf
        import pandas as pd

        t = yf.Ticker(ticker)
        info = t.fast_info          # lightweight; avoids heavy scraping

        hist = t.history(period="1y", auto_adjust=True)
        if hist.empty:
            return {"ticker": ticker, "error": "no_price_history"}

        close = hist["Close"]
        volume = hist["Volume"]

        week52_high = float(info.get("year_high") or close.max())
        current = float(close.iloc[-1])
        pct_from_high = round((current - week52_high) / week52_high * 100, 2)

        vol_today = float(volume.iloc[-1])
        vol_3m_avg = float(volume.rolling(63).mean().iloc[-1]) if len(volume) >= 63 else float(volume.mean())
        vol_ratio = round(vol_today / vol_3m_avg, 3) if vol_3m_avg > 0 else None

        ytd_start = close[close.index.year == close.index[-1].year].iloc[0] if not close.empty else current
        ytd_return = round((current - float(ytd_start)) / float(ytd_start) * 100, 2)

        result: Dict[str, Any] = {
            "ticker": ticker,
            "current_price": round(current, 2),
            "52w_high": round(week52_high, 2),
            "pct_from_52w_high": pct_from_high,
            "volume_ratio_vs_3m_avg": vol_ratio,
            "ytd_return_pct": ytd_return,
        }

        # Append equity-specific ratios if available
        pe = info.get("pe_ratio") or info.get("trailing_pe")
        pb = info.get("price_to_book")
        if pe:
            result["trailing_pe"] = round(float(pe), 2)
        if pb:
            result["price_to_book"] = round(float(pb), 2)

        return result

    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


class FundamentalResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Fundamental_Research_Agent",
            role=(
                "Fundamental Analyst focusing on valuation, volume dynamics, and relative strength. "
                "For ETFs you focus on price health vs 52-week highs, institutional volume surges, "
                "and YTD return trend rather than traditional P&L metrics."
            ),
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        ticker = state["current_asset"]
        print(f"[{self.name}] Fetching fundamentals for {ticker}...")

        data = _fetch_etf_fundamentals(ticker)

        prompt = f"""
Analyze the following fundamental/market-structure data for {ticker}:
{json.dumps(data, indent=2)}

Interpretation guidelines:
- pct_from_52w_high < -15% and volume_ratio > 1.5 → potential capitulation / contrarian long
- pct_from_52w_high > -5% with rising volume_ratio → breakout / momentum long
- ytd_return < -10% and no volume confirmation → caution / short bias
- For equity ETFs: trailing_pe > 25 in rising rate environment → overvalued

Produce a JSON output matching this schema exactly:
{{
    "agent_name": "Fundamental_Research_Agent",
    "action": "PROPOSE_LONG" | "HOLD",
    "confidence": <float between 0.0 and 1.0>,
    "rationale": "<concise explanation referencing the actual data values>"
}}
"""

        pct_from_high = data.get("pct_from_52w_high")
        volume_ratio = data.get("volume_ratio_vs_3m_avg")
        ytd_return = data.get("ytd_return_pct")
        action = "HOLD"
        confidence = 0.25
        rationale = "No clear fundamental signal."
        if data.get("error"):
            rationale = f"Data fetch error ({data['error']}) — neutral."
        elif isinstance(pct_from_high, (int, float)) and isinstance(volume_ratio, (int, float)):
            if pct_from_high > -12 and volume_ratio >= 0.8:
                # Within 12% of 52w high with normal volume — uptrend intact
                action = "PROPOSE_LONG"
                confidence = 0.60
                rationale = f"Price within 12% of 52w high ({pct_from_high:.1f}%) with adequate volume ({volume_ratio:.2f}x). Uptrend intact."
            elif isinstance(ytd_return, (int, float)) and ytd_return > 8 and volume_ratio >= 0.8:
                # Strong YTD return + steady volume — sustained positive trend
                action = "PROPOSE_LONG"
                confidence = 0.52
                rationale = f"YTD return={ytd_return:.1f}% with steady volume ({volume_ratio:.2f}x). Sustained uptrend."
            elif pct_from_high < -15 and volume_ratio >= 1.3:
                # Deep pullback with above-average volume — capitulation / contrarian long
                action = "PROPOSE_LONG"
                confidence = 0.55
                rationale = f"Deep pullback ({pct_from_high:.1f}% from high) with volume surge ({volume_ratio:.2f}x). Potential capitulation."
            elif isinstance(ytd_return, (int, float)) and ytd_return < -10 and volume_ratio < 1.0:
                action = "HOLD"
                confidence = 0.30
                rationale = f"Weak YTD return ({ytd_return:.1f}%) without volume confirmation — staying flat."

        fallback = {
            "agent_name": self.name,
            "action": action,
            "confidence": confidence,
            "rationale": rationale,
        }

        decision = self._invoke_llm_json(prompt, fallback)

        return {"research_signals": [decision]}


fundamental_agent = FundamentalResearchAgent()
