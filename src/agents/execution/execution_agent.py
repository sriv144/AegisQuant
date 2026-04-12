import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

class ExecutionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Trade_Execution_Agent",
            role="Execution Trader. You handle routing the approved allocation to the broker or simulation engine. You implement optimal TWAP / VWAP execution slicing to minimize slippage."
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Preparing execution strategy for {state['current_asset']}...")
        
        risk_approval = state.get("risk_approval", {})
        allocation_request = state.get("allocation_request", {})
        
        prompt = f"""
        Risk Management has APPROVED the trade for {state['current_asset']}:
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
        return {"execution_result": decision}

execution_agent = ExecutionAgent()
