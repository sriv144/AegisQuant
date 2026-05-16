import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState


def _fetch_macro_snapshot() -> Dict[str, Any]:
    """
    Pull live macro indicators from yfinance.
    Tickers:
      ^VIX   — CBOE Volatility Index (fear gauge)
      ^TNX   — 10-year Treasury yield
      ^FVX   — 5-year Treasury yield (slope proxy: ^TNX - ^FVX ≈ term spread)
      DX-Y.NYB — US Dollar Index
    Returns the most-recent closing values.
    Includes timeout protection to prevent hanging on slow/unavailable feeds.
    """
    try:
        import yfinance as yf
        import pandas as pd
        from threading import Thread
        import queue as queue_module

        result_queue = queue_module.Queue()

        def fetch_data():
            try:
                tickers = {"^VIX": "vix", "^TNX": "10y_treasury_yield", "^FVX": "5y_treasury_yield", "DX-Y.NYB": "usd_index"}
                raw = yf.download(list(tickers.keys()), period="5d", auto_adjust=True, progress=False)
                result_queue.put(raw)
            except Exception as e:
                result_queue.put(None)

        thread = Thread(target=fetch_data, daemon=True)
        thread.start()
        thread.join(timeout=5)  # 5 second timeout

        try:
            raw = result_queue.get_nowait()
        except queue_module.Empty:
            return {
                "vix": None,
                "10y_treasury_yield": None,
                "5y_treasury_yield": None,
                "usd_index": None,
                "yield_curve_slope_bps": None,
                "fetch_error": "yfinance timeout",
            }

        if raw is None:
            return {
                "vix": None,
                "10y_treasury_yield": None,
                "5y_treasury_yield": None,
                "usd_index": None,
                "yield_curve_slope_bps": None,
                "fetch_error": "yfinance returned None",
            }

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        latest = close.ffill().iloc[-1]

        tickers = {"^VIX": "vix", "^TNX": "10y_treasury_yield", "^FVX": "5y_treasury_yield", "DX-Y.NYB": "usd_index"}
        snapshot: Dict[str, Any] = {}
        for yf_ticker, label in tickers.items():
            val = latest.get(yf_ticker)
            snapshot[label] = round(float(val), 4) if val is not None and not pd.isna(val) else None

        # Derived: yield curve slope (term spread)
        if snapshot.get("10y_treasury_yield") and snapshot.get("5y_treasury_yield"):
            snapshot["yield_curve_slope_bps"] = round(
                (snapshot["10y_treasury_yield"] - snapshot["5y_treasury_yield"]) * 100, 1
            )

        return snapshot

    except Exception as exc:
        # Fallback so the agent doesn't crash if yfinance is unavailable
        return {
            "vix": None,
            "10y_treasury_yield": None,
            "5y_treasury_yield": None,
            "usd_index": None,
            "yield_curve_slope_bps": None,
            "fetch_error": str(exc),
        }


class MacroAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Macro_Research_Agent",
            role=(
                "Macro-economic Analyst focused on interest rates, inflation, and global liquidity. "
                "You identify systemic risks and macro trends that override individual asset fundamentals."
            ),
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Analyzing macro environment for {state['current_asset']}...")

        macro_data = _fetch_macro_snapshot()

        prompt = f"""
Analyze the following live macro-economic indicators for the global market involving {state['current_asset']}:
{json.dumps(macro_data, indent=2)}

Interpret:
- VIX > 25 signals elevated fear / risk-off regime
- 10y yield rising fast → tightening financial conditions
- Negative yield curve slope (inverted) → recession risk
- Rising USD index → pressure on international assets

Produce a JSON output matching this schema exactly:
{{
    "agent_name": "Macro_Research_Agent",
    "action": "PROPOSE_LONG" | "HOLD",
    "confidence": <float between 0.0 and 1.0>,
    "rationale": "<concise explanation referencing the actual indicator values>"
}}
"""

        vix = macro_data.get("vix")
        slope = macro_data.get("yield_curve_slope_bps")
        action = "HOLD"
        confidence = 0.25
        rationale = "Fallback macro signal remains neutral."
        if isinstance(vix, (int, float)) and vix >= 25:
            action = "HOLD"
            confidence = 0.30
            rationale = "Elevated VIX indicates a risk-off regime — staying flat (long-only mode)."
        elif isinstance(slope, (int, float)) and slope < 0:
            action = "HOLD"
            confidence = 0.25
            rationale = "An inverted yield curve suggests macro stress — staying flat (long-only mode)."

        fallback = {
            "agent_name": self.name,
            "action": action,
            "confidence": confidence,
            "rationale": rationale,
        }

        decision = self._invoke_llm_json(prompt, fallback)

        return {"research_signals": [decision]}


macro_agent = MacroAgent()
