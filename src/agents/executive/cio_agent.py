import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

class CIOAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Chief_Investment_Officer",
            role="Chief Investment Officer. You oversee the overall fund strategy, review historical performance, and adjust broad asset allocation targets or risk limits dynamically."
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        """
        The CIO is an asynchronous oversight agent rather than a per-tick router.
        It evaluates portfolio snapshots and adjusts global parameters for PMs and Risk Officers.
        """
        print(f"[{self.name}] Conducting periodic review of portfolio metrics...")
        
        portfolio_state = state.get("portfolio_state", {})
        
        prompt = f"""
        Review the current overall portfolio status and historical analytics:
        {json.dumps(portfolio_state, indent=2)}
        
        Issue new directives to the PM and Risk teams. Are we too exposed? Do we need to scale down volatility?
        Produce a JSON output matching this schema:
        {{
            "agent_name": "Chief_Investment_Officer",
            "action": "ADJUST_LIMITS",
            "confidence": float,
            "rationale": "High-level strategic rationale.",
            "metadata": {{
                "target_volatility": float,
                "max_drawdown_limit": float
            }}
        }}
        """

        fallback = {
            "agent_name": self.name,
            "action": "ADJUST_LIMITS",
            "confidence": 0.25,
            "rationale": "Deterministic fallback keeps current portfolio risk limits unchanged.",
            "metadata": {
                "target_volatility": float(portfolio_state.get("target_volatility", 0.12) or 0.12),
                "max_drawdown_limit": float(portfolio_state.get("max_drawdown_limit", 0.15) or 0.15),
            },
        }

        decision = self._invoke_llm_json(prompt, fallback)
        # Note: CIO updates the global state limits, not the immediate tick per-asset state.
        # This is a simplification for the current graph.
        return {"cio_directive": decision}

cio_agent = CIOAgent()
