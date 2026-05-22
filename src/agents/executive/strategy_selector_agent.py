"""
Strategy Selector Agent
======================
New crew member that reads weekly performance + macro regime.
Nominates 2-3 active strategies per cycle.
"""

import json
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.agents.state import AgentState

class StrategySelectorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Strategy_Selector",
            role="Chief Strategy Selector. You receive weekly strategy performance scores and current market regime signals. You nominate the top 2-3 strategies most likely to outperform in this environment."
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        """
        Select active strategies based on:
        1. Weekly Sharpe ratios from state["strategy_scores"]
        2. Current market regime (VIX level)
        3. LLM synthesis for final ranking
        """
        print(f"[{self.name}] Selecting active strategies for {state['current_asset']}...")

        strategy_scores = state.get("strategy_scores", {})
        vix_raw = state.get("portfolio_state", {}).get("vix_raw", 20.0)
        drawdown = state.get("portfolio_state", {}).get("current_drawdown", 0.0)

        # Regime-based fallback if no historical scores yet
        if not strategy_scores:
            active_strategies = self._regime_based_fallback(vix_raw, drawdown)
        else:
            active_strategies = sorted(
                strategy_scores.keys(),
                key=lambda s: strategy_scores[s],
                reverse=True
            )[:3]

        # Select current strategy (top performer or first if no scores)
        current_strategy = active_strategies[0] if active_strategies else "momentum"
        memory_context = self._format_memory_context(state)

        prompt = f"""
        Current market regime:
        - India VIX: {vix_raw:.1f}
        - Current Drawdown: {drawdown:.1%}
        - Asset: {state['current_asset']}

        Weekly strategy performance (Sharpe ratios):
        {json.dumps(strategy_scores, indent=2)}

        Active strategy candidates: {active_strategies}
        Current primary strategy: {current_strategy}

        {memory_context}

        Confirm or adjust the strategy selection based on the regime and performance.
        Produce a JSON output:
        {{
            "active_strategies": ["strategy1", "strategy2", "strategy3"],
            "current_strategy": "strategy_name",
            "rationale": "Why these strategies are selected for this regime"
        }}
        """

        fallback = {
            "active_strategies": active_strategies,
            "current_strategy": current_strategy,
            "rationale": f"Regime-based selection: VIX={vix_raw:.1f}, Drawdown={drawdown:.2%}"
        }

        decision = self._invoke_llm_json(prompt, fallback)

        return {
            "active_strategies": decision.get("active_strategies", active_strategies),
            "current_strategy": decision.get("current_strategy", current_strategy),
        }

    def _regime_based_fallback(self, vix_raw: float, drawdown: float) -> List[str]:
        """
        Fallback strategy selection based on market regime (no LLM).
        """
        if vix_raw < 15.0:
            # Low volatility = trend following + momentum work well
            return ["trend_following", "momentum", "factor_investing"]
        elif vix_raw < 25.0:
            # Normal volatility = balanced approach
            return ["mean_reversion", "trend_following", "sector_rotation"]
        else:
            # High volatility = mean reversion + breakout + defensive
            return ["volatility_breakout", "gap_fill", "mean_reversion"]

strategy_selector_agent = StrategySelectorAgent()
