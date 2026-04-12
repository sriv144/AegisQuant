import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

class RiskOfficerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Risk_Officer",
            role="Chief Risk Officer. You have absolute veto power over all trades. You evaluate current portfolio drawdown, exposure limits, and single-asset position limits. You output APPROVE or REJECT for the PM's allocation request."
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Evaluating risk limits for {state['current_asset']}...")
        
        allocation_request = state.get("allocation_request", {})
        portfolio_state = state.get("portfolio_state", {})
        
        prompt = f"""
        The Portfolio Manager has requested the following adjusted allocation for {state['current_asset']}:
        {json.dumps(allocation_request, indent=2)}
        
        Current portfolio global risk constraints:
        {json.dumps(portfolio_state.get('risk_limits', {}), indent=2)}
        
        Assess if this trade violates any maximum drawdown limits, volatility ceilings, or exposure rules.
        Produce a JSON output matching this schema:
        {{
            "agent_name": "Risk_Officer",
            "action": "APPROVE" | "REJECT",
            "confidence": float (0.0 to 1.0),
            "rationale": "Explanation for the approval or rejection. If rejected, cite the violated rule.",
            "max_volume_allowed": float # Optionally scale down the requested size
        }}
        """

        risk_limits = portfolio_state.get("risk_limits", {})
        adjusted_exposure = float(allocation_request.get("adjusted_exposure_pct", allocation_request.get("target_exposure_pct", 0.0)) or 0.0)
        max_position = float(risk_limits.get("max_position_pct", 0.25) or 0.25)
        current_drawdown = float(portfolio_state.get("current_drawdown", 0.0) or 0.0)
        max_drawdown = float(risk_limits.get("max_drawdown_limit", 0.2) or 0.2)
        approved = adjusted_exposure <= max_position and current_drawdown <= max_drawdown
        fallback = {
            "agent_name": self.name,
            "action": "APPROVE" if approved else "REJECT",
            "confidence": float(allocation_request.get("confidence", 0.0) or 0.0),
            "rationale": "Deterministic fallback based on configured exposure and drawdown limits.",
            "max_volume_allowed": round(min(adjusted_exposure, max_position), 4),
        }

        decision = self._invoke_llm_json(prompt, fallback)
        return {"risk_approval": decision}

risk_officer_agent = RiskOfficerAgent()
