import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

class AssetAllocationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Asset_Allocation_Agent",
            role="Asset Allocator focusing on diversification. You analyze the current correlation matrix and adjust the PM's target exposure to ensure the portfolio is not overly balanced in a single asset class or sector."
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Checking portfolio correlations for {state['current_asset']}...")
        
        pm_request = state.get("allocation_request", {})
        portfolio_state = state.get("portfolio_state", {})
        
        prompt = f"""
        The Portfolio Manager has requested the following allocation for {state['current_asset']}:
        {json.dumps(pm_request, indent=2)}
        
        Current portfolio exposure map:
        {json.dumps(portfolio_state.get('asset_exposures', {}), indent=2)}
        
        Adjust the requested allocation for correlation risks and output the final target exposure.
        Produce a JSON output matching this schema:
        {{
            "agent_name": "Asset_Allocation_Agent",
            "action": "ADJUST_ALLOCATION",
            "confidence": float (0.0 to 1.0),
            "rationale": "Explanation for preserving or scaling down the PM's requested size due to correlations.",
            "adjusted_exposure_pct": float (0.0 to 1.0)
        }}
        """

        current_exposure = float(portfolio_state.get("asset_exposures", {}).get(state["current_asset"], 0.0) or 0.0)
        requested = float(pm_request.get("target_exposure_pct", 0.0) or 0.0)
        adjusted = requested * 0.75 if current_exposure > 0.2 else requested
        fallback = {
            "agent_name": self.name,
            "action": "ADJUST_ALLOCATION",
            "confidence": float(pm_request.get("confidence", 0.0) or 0.0),
            "rationale": "Deterministic fallback preserving PM sizing unless current exposure is already elevated.",
            "adjusted_exposure_pct": round(max(0.0, min(1.0, adjusted)), 4),
        }

        decision = self._invoke_llm_json(prompt, fallback)
        
        # We overwrite the allocation request with the adjusted one before it hits Risk.
        # LangGraph state updates dictionaries by default.
        return {"allocation_request": decision}

asset_allocation_agent = AssetAllocationAgent()
