import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

class ExecutionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Trade_Execution_Agent",
            role="Execution Trader. You handle routing the approved allocation to the broker or simulation engine. You implement optimal TWAP / VWAP execution slicing to minimize slippage. You also determine whether to trade intraday (MIS) or delivery (CNC) based on strategy type and capital availability."
        )

    def _determine_trade_type(self, state: AgentState) -> str:
        """
        Determine MIS vs CNC based on strategy and conviction.

        CNC (delivery) if:
          - committee confidence >= 0.6
          - strategy in [momentum, trend_following, factor_investing, sector_rotation, earnings_momentum]
          - delivery_budget > 0

        MIS (intraday) if:
          - committee confidence >= 0.5
          - strategy in [gap_fill, volatility_breakout, mean_reversion, pairs_trading]
          - intraday_budget > 0
          - High news_volume or VIX spike today

        SKIP otherwise
        """
        committee = state.get("committee_decision", {})
        confidence = committee.get("confidence", 0.0)
        strategy = state.get("current_strategy", "momentum")
        alt_data = state.get("alternative_data", {})
        portfolio = state.get("portfolio_state", {})

        delivery_strategies = {
            "momentum", "trend_following", "factor_investing",
            "sector_rotation", "earnings_momentum"
        }
        intraday_strategies = {
            "gap_fill", "volatility_breakout", "mean_reversion", "pairs_trading"
        }

        # CNC priority
        if confidence >= 0.6 and strategy in delivery_strategies:
            if state.get("delivery_budget", 0) > 0:
                return "CNC"

        # MIS if intraday strategy + high conviction
        if confidence >= 0.5 and strategy in intraday_strategies:
            news_volume = alt_data.get("news_volume", 0)
            vix = portfolio.get("vix_raw", 20)
            high_catalyst = news_volume > 2 or vix > 25
            if state.get("intraday_budget", 0) > 0 and high_catalyst:
                return "MIS"

        return "SKIP"

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Preparing execution strategy for {state['current_asset']}...")

        # Determine trade type first
        trade_type = self._determine_trade_type(state)

        # If SKIP, don't execute
        if trade_type == "SKIP":
            print(f"[{self.name}] SKIP — no suitable MIS/CNC execution")
            return {
                "execution_result": {
                    "action": "SKIP",
                    "rationale": "Insufficient conviction or budget for MIS/CNC",
                },
                "trade_type": "SKIP",
            }

        risk_approval = state.get("risk_approval", {})
        allocation_request = state.get("allocation_request", {})

        prompt = f"""
        Risk Management has APPROVED the trade for {state['current_asset']}:
        Trade Type: {trade_type}
        {json.dumps(risk_approval, indent=2)}
        {json.dumps(allocation_request, indent=2)}

        Determine the execution algorithm to minimize slippage on the requested size.
        Produce a JSON output matching this schema:
        {{
            "agent_name": "Trade_Execution_Agent",
            "action": "EXECUTE",
            "execution_algo": "TWAP" | "VWAP" | "MARKET" | "LIMIT",
            "limit_price": float (Optional limit price target, based on requested entry),
            "rationale": "Execution logic explanation."
        }}
        """

        fallback = {
            "agent_name": self.name,
            "action": "EXECUTE",
            "execution_algo": "MARKET",
            "limit_price": None,
            "rationale": "Deterministic fallback uses market execution after risk approval.",
        }

        decision = self._invoke_llm_json(prompt, fallback)

        # Return both execution_result and trade_type
        return {
            "execution_result": decision,
            "trade_type": trade_type,
        }

execution_agent = ExecutionAgent()
