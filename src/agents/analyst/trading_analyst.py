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
                "You are the Chief Trading Analyst at an AI hedge fund. "
                "You make the FINAL buy/hold/sell decision for each stock. "
                "You analyze technical indicators, research agent opinions, "
                "strategy signals, and market conditions to form your own view. "
                "You explain your reasoning clearly. You are long-only — you "
                "never short. You think like the best trader in the world."
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

        prompt = f"""Analyze {ticker} (current price: ${current_price:.2f}) and make a trading decision.

## Your Portfolio
- Portfolio value: ${portfolio_state.get('portfolio_value', 100000):,.0f}
- Current drawdown: {portfolio_state.get('current_drawdown', 0):.2%}
- VIX (market fear): {portfolio_state.get('vix_raw', 20):.1f}
- Currently holding {ticker}: {"YES" if is_held else "NO"}

## Technical Indicators for {ticker}
{key_indicators}

## Research Agent Opinions (4 agents analyzed this stock)
{agent_summary}

## Strategy Signals (9 strategies scored this stock)
{strategy_summary}

## Your Task
You are LONG-ONLY. You can BUY, HOLD (do nothing), or EXIT (sell existing position).

Think step by step:
1. What is the overall trend and momentum telling you?
2. Is the stock at an attractive entry point or overextended?
3. What do the research agents agree/disagree on?
4. Which trading strategy best fits this stock's current situation?
5. What is the risk/reward here?

Then decide:
- BUY: You see a good opportunity. State your confidence (0.0-1.0) and suggested allocation (1-10% of portfolio).
- HOLD: Not compelling enough to act. Already held positions stay.
- EXIT: Only if currently held AND the position should be closed (e.g., momentum lost, stop-loss level, thesis broken).

Respond with ONLY a JSON object:
{{
    "action": "BUY" | "HOLD" | "EXIT",
    "confidence": <float 0.0 to 1.0>,
    "allocation_pct": <float 0.01 to 0.10, only if BUY>,
    "reasoning": "<2-3 sentences explaining your decision — what you see, why you decided this, what strategy approach you're using>",
    "strategy_used": "<which approach: momentum, mean_reversion, trend_following, breakout, value, contrarian, or mixed>"
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

        # Check for EXIT signals on held positions
        if is_held:
            # Check momentum and RSI for deterioration
            rsi_z = indicators.get("RSI_14_Z", 0.0)
            mom_z = indicators.get("mom_12m_Z", 0.0)
            bb_pos = indicators.get("BB_Position", 0.5)

            if mom_z < -1.0 and rsi_z > 1.5:
                return {
                    "action": "EXIT",
                    "confidence": 0.6,
                    "allocation_pct": 0.0,
                    "reasoning": f"Momentum deteriorating (mom_z={mom_z:.2f}) while RSI overbought (rsi_z={rsi_z:.2f}). Closing position.",
                    "strategy_used": "risk_management",
                }
            if bb_pos < 0.05 and mom_z < -0.5:
                return {
                    "action": "EXIT",
                    "confidence": 0.55,
                    "allocation_pct": 0.0,
                    "reasoning": f"Price near lower Bollinger Band (BB={bb_pos:.3f}) with negative momentum. Protecting capital.",
                    "strategy_used": "risk_management",
                }

        # Decision thresholds (soft, not rigid)
        if consensus >= 0.35 and avg_conf >= 0.4:
            # Strong agreement — BUY
            allocation = min(0.10, 0.03 + consensus * 0.07)
            return {
                "action": "BUY",
                "confidence": round(min(0.9, avg_conf), 4),
                "allocation_pct": round(allocation, 4),
                "reasoning": f"Consensus={consensus:.0%} across {agent_long} agents + {strategy_long} strategies. " + "; ".join(reasons[:2]),
                "strategy_used": "consensus",
            }
        elif consensus >= 0.20 and avg_conf >= 0.5:
            # Moderate agreement with high conviction — smaller BUY
            allocation = min(0.06, 0.02 + consensus * 0.04)
            return {
                "action": "BUY",
                "confidence": round(min(0.7, avg_conf * 0.8), 4),
                "allocation_pct": round(allocation, 4),
                "reasoning": f"Moderate consensus={consensus:.0%} but high conviction signals. " + "; ".join(reasons[:2]),
                "strategy_used": "selective",
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.3,
                "allocation_pct": 0.0,
                "reasoning": f"Insufficient consensus ({consensus:.0%}) — {agent_long}/{agent_total} agents and {strategy_long}/{strategy_total} strategies favor LONG.",
                "strategy_used": "none",
            }


# Module-level singleton
trading_analyst = TradingAnalystAgent()
