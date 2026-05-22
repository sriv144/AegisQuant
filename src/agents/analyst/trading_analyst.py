"""
Trading Analyst Agent
======================
LLM-powered agent that reasons about each stock autonomously.

Receives ALL context for a ticker:
  - Technical indicators (RSI, MACD, BB, ADX, momentum, volume, etc.)
  - Research agent signals (4 agents: quant, fundamental, macro, sentiment)
  - Strategy signals (9 strategies: momentum, mean reversion, trend, etc.)
  - Current portfolio state (positions, drawdown, VIX, cash)
  - Whether the stock is currently held

The LLM then:
  1. Analyzes the full picture
  2. Picks which signals/strategies are most relevant for THIS stock
  3. Decides: BUY (with confidence), HOLD, or EXIT (with reason)
  4. Explains its reasoning in detail

When no LLM is available (OPENAI_API_KEY not set), falls back to a
soft consensus heuristic (no rigid fixed rules).
"""

import json
import logging
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TradingAnalystAgent(BaseAgent):
    """
    Autonomous trading analyst — one LLM call per ticker, full reasoning.
    """

    def __init__(self):
        super().__init__(
            name="Trading_Analyst",
            role=(
                "You are a long-term value investor in the style of Warren Buffett. "
                "You invest in WONDERFUL companies at FAIR prices and hold them for years. "
                "You think in business terms, not price terms. Your goal is to compound "
                "capital over the long run — you are NOT a trader chasing short-term moves. "
                "You only EXIT a position when: (1) the business thesis is broken, "
                "(2) the price is dangerously overvalued, or (3) a better opportunity "
                "requires the capital. You NEVER exit because of a bad week or short-term "
                "price noise. If a quality company you hold dips 5-10%, that is a BUYING "
                "opportunity, not a reason to sell — unless the underlying business has changed. "
                "You are long-only and patient. Favor HOLD over EXIT in ambiguous situations. "
                "Be greedy when others are fearful (VIX spikes) and patient when others are greedy."
            ),
        )

    def analyze_ticker(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        agent_signals: List[Dict[str, Any]],
        strategy_signals: List[Dict[str, Any]],
        portfolio_state: Dict[str, Any],
        is_held: bool,
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Analyze a single ticker and decide BUY / HOLD / EXIT.

        Returns:
            {
                "action": "BUY" | "HOLD" | "EXIT",
                "confidence": float (0.0 to 1.0),
                "allocation_pct": float (suggested % of portfolio, 0.0 to 0.10),
                "reasoning": str (detailed explanation),
                "strategy_used": str (which approach the agent chose),
            }
        """
        # Format the data for the LLM
        key_indicators = self._format_indicators(indicators)
        agent_summary = self._format_agent_signals(agent_signals)
        strategy_summary = self._format_strategy_signals(strategy_signals)

        position_status = (
            "HELD (core long-term position — only EXIT if the business thesis is broken)"
            if is_held else "NEW candidate (not currently held)"
        )
        vix = portfolio_state.get("vix_raw", 20)
        vix_context = (
            "HIGH — be greedy when others are fearful, look for buying opportunities"
            if vix > 25 else "Normal"
        )

        prompt = f"""Analyze {ticker} and decide whether to BUY, HOLD, or EXIT as a long-term value investor.

## Portfolio Context
- Portfolio value: ${portfolio_state.get('portfolio_value', 100000):,.0f}
- Current drawdown: {portfolio_state.get('current_drawdown', 0):.2%}
- VIX (fear gauge): {vix:.1f} — {vix_context}
- Position status: {position_status}
- Current price: ${current_price:.2f}

## Technical Indicators
{key_indicators}

## Research Agent Opinions (4 agents)
{agent_summary}

## Strategy Signals (9 strategies)
{strategy_summary}

## Your Task (Buffett-style investor)
You are LONG-ONLY. Think in YEARS, not days. Evaluate whether this is a wonderful business at a fair price.

Decision rules:
- BUY: The business is high-quality, reasonably priced, and agents/strategies show conviction.
  Set confidence >= 0.55 for a core holding (10% allocation). 0.40-0.54 for a smaller tactical bet.
- HOLD: The thesis is intact. Short-term noise doesn't change the long-term story. Default for held positions.
- EXIT: ONLY if (1) thesis is broken, (2) price is dangerously overvalued, or (3) capital needed elsewhere.
  Do NOT exit because of a -5% to -10% drawdown on a quality company — that's normal volatility.

Respond with ONLY a JSON object:
{{
    "action": "BUY" | "HOLD" | "EXIT",
    "confidence": <float 0.0 to 1.0>,
    "allocation_pct": <float 0.01 to 0.10, only if BUY>,
    "reasoning": "<2-3 sentences: what the business fundamentals and signals show, why you made this decision>",
    "strategy_used": "<value, quality_growth, momentum, contrarian, or mixed>"
}}"""

        # Compute deterministic fallback
        fallback = self._compute_fallback(
            ticker, indicators, agent_signals, strategy_signals, is_held
        )

        # Call LLM (or use fallback if no API key)
        decision = self._invoke_llm_json(prompt, fallback)

        # Safety: enforce long-only, cap allocation
        action = decision.get("action", "HOLD").upper()
        if action not in ("BUY", "HOLD", "EXIT"):
            action = "HOLD"
        if action == "EXIT" and not is_held:
            action = "HOLD"  # Can't exit what you don't hold

        confidence = min(1.0, max(0.0, float(decision.get("confidence", 0.3))))
        allocation = min(0.10, max(0.0, float(decision.get("allocation_pct", 0.0))))

        result = {
            "action": action,
            "confidence": round(confidence, 4),
            "allocation_pct": round(allocation, 4) if action == "BUY" else 0.0,
            "reasoning": decision.get("reasoning", fallback.get("reasoning", "")),
            "strategy_used": decision.get("strategy_used", "mixed"),
            "used_llm": self.llm is not None,
        }

        # Log the decision
        llm_tag = "LLM" if result["used_llm"] else "FALLBACK"
        if action != "HOLD":
            print(
                f"  [{ticker}] {llm_tag} -> {action} "
                f"(conf={confidence:.2f}, alloc={allocation:.1%}) "
                f"| {result['reasoning'][:80]}..."
            )

        return result

    def _format_indicators(self, ind: Dict[str, Any]) -> str:
        """Format key indicators into readable text for the LLM."""
        lines = []
        mapping = {
            "RSI_14": ("RSI (14-day)", "below 30 = oversold, above 70 = overbought"),
            "RSI_14_Z": ("RSI Z-score", "below -2 = deeply oversold, above 2 = deeply overbought"),
            "MACD_Z": ("MACD Z-score", "positive = bullish momentum, negative = bearish"),
            "BB_Position": ("Bollinger Band Position", "0 = at lower band, 0.5 = middle, 1.0 = at upper band, >1.0 = breakout above"),
            "ADX_14": ("ADX (trend strength)", "below 20 = no trend, above 25 = trending, above 40 = strong trend"),
            "mom_12m_Z": ("12-month Momentum Z", "positive = outperforming, negative = underperforming"),
            "Volatility_20_Z": ("20-day Volatility Z", "high = more volatile than usual"),
            "Volume_Z": ("Volume Z-score", "above 1.5 = unusual volume surge"),
        }
        for key, (label, hint) in mapping.items():
            val = ind.get(key)
            if val is not None:
                lines.append(f"- {label}: {val:.3f} ({hint})")
        return "\n".join(lines) if lines else "- No indicators available"

    def _format_agent_signals(self, signals: List[Dict[str, Any]]) -> str:
        """Format research agent signals into readable text."""
        if not signals:
            return "- No research signals available"
        lines = []
        for s in signals:
            name = s.get("agent_name", "Unknown")
            action = s.get("action", "?")
            conf = s.get("confidence", 0)
            rationale = s.get("rationale", "")
            lines.append(f"- {name}: {action} (confidence: {conf:.2f}) — {rationale}")
        return "\n".join(lines)

    def _format_strategy_signals(self, signals: List[Dict[str, Any]]) -> str:
        """Format strategy signals into readable text."""
        if not signals:
            return "- No strategy signals available"
        lines = []
        for s in signals:
            name = s.get("strategy", "Unknown")
            action = s.get("action", "?")
            conf = s.get("confidence", 0)
            rationale = s.get("rationale", "")[:100]
            lines.append(f"- {name}: {action} (confidence: {conf:.2f}) — {rationale}")
        return "\n".join(lines)

    def _compute_fallback(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        agent_signals: List[Dict[str, Any]],
        strategy_signals: List[Dict[str, Any]],
        is_held: bool,
    ) -> Dict[str, Any]:
        """
        Soft consensus fallback when LLM is unavailable.
        No rigid rules — just counts agreement across agents and strategies.
        """
        # Count LONG/BUY signals
        agent_long = sum(
            1 for s in agent_signals
            if s.get("action") in ("PROPOSE_LONG",)
        )
        agent_total = len(agent_signals) or 1

        strategy_long = sum(
            1 for s in strategy_signals
            if s.get("action") in ("LONG",)
        )
        strategy_total = len(strategy_signals) or 1

        # Weighted consensus: how many signals agree on LONG?
        agent_pct = agent_long / agent_total
        strategy_pct = strategy_long / strategy_total
        consensus = 0.6 * agent_pct + 0.4 * strategy_pct

        # Average confidence of LONG signals
        long_confs = [
            float(s.get("confidence", 0.3))
            for s in agent_signals + strategy_signals
            if s.get("action") in ("PROPOSE_LONG", "LONG")
        ]
        avg_conf = sum(long_confs) / len(long_confs) if long_confs else 0.2

        # Collect reasons from LONG signals
        reasons = []
        for s in agent_signals:
            if s.get("action") == "PROPOSE_LONG":
                reasons.append(f"{s.get('agent_name', '?')}: {s.get('rationale', '')[:60]}")
        for s in strategy_signals:
            if s.get("action") == "LONG":
                reasons.append(f"{s.get('strategy', '?')}: {s.get('rationale', '')[:60]}")

        # EXIT logic — Buffett-style: only on genuinely broken thesis
        # A simple -5% or -10% dip is NOT a reason to exit a quality company
        if is_held:
            rsi_z = indicators.get("RSI_14_Z", 0.0)
            mom_z = indicators.get("mom_12m_Z", 0.0)
            bb_pos = indicators.get("BB_Position", 0.5)

            # Only exit on severe simultaneous deterioration (all 3 signals agree)
            if mom_z < -1.5 and rsi_z > 1.5 and bb_pos < 0.15:
                return {
                    "action": "EXIT",
                    "confidence": 0.65,
                    "allocation_pct": 0.0,
                    "reasoning": (
                        f"Multiple severe deterioration signals: momentum={mom_z:.2f}, "
                        f"RSI={rsi_z:.2f}, price at extreme low (BB={bb_pos:.3f}). "
                        "Business thesis appears broken — exiting."
                    ),
                    "strategy_used": "risk_management",
                }

        # BUY decision thresholds — higher bar than before to reduce churn
        # Strong conviction: consensus >= 45% AND avg confidence >= 45%
        if consensus >= 0.45 and avg_conf >= 0.45:
            allocation = min(0.10, 0.04 + consensus * 0.06)
            return {
                "action": "BUY",
                "confidence": round(min(0.9, avg_conf), 4),
                "allocation_pct": round(allocation, 4),
                "reasoning": (
                    f"Strong consensus={consensus:.0%} ({agent_long} agents, {strategy_long} strategies). "
                    + "; ".join(reasons[:2])
                ),
                "strategy_used": "quality_consensus",
            }
        # Moderate consensus but very high individual conviction
        elif consensus >= 0.30 and avg_conf >= 0.55:
            allocation = min(0.05, 0.02 + consensus * 0.03)
            return {
                "action": "BUY",
                "confidence": round(min(0.70, avg_conf * 0.85), 4),
                "allocation_pct": round(allocation, 4),
                "reasoning": (
                    f"Moderate consensus={consensus:.0%} but high-conviction signals (avg_conf={avg_conf:.2f}). "
                    + "; ".join(reasons[:2])
                ),
                "strategy_used": "selective_quality",
            }
        else:
            # Default: HOLD — patience is a Buffett virtue
            action = "HOLD"
            hold_reason = (
                f"Consensus too low ({consensus:.0%}) or conviction insufficient "
                f"({agent_long}/{agent_total} agents, {strategy_long}/{strategy_total} strategies favor LONG). "
                "Holding cash; waiting for better opportunity."
            )
            if is_held:
                hold_reason = (
                    f"Currently held — maintaining position. Consensus={consensus:.0%} with "
                    f"{agent_long}/{agent_total} agents bullish. No compelling reason to exit."
                )
            return {
                "action": action,
                "confidence": 0.35,
                "allocation_pct": 0.0,
                "reasoning": hold_reason,
                "strategy_used": "none",
            }


# Module-level singleton
trading_analyst = TradingAnalystAgent()
