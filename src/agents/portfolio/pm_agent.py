import json
import os
import numpy as np
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

MODEL_PATH = "ppo_portfolio_manager.zip"

class PMAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Portfolio_Manager",
            role="Portfolio Manager. You receive approved signals from the Strategy Committee and output the initial sizing/allocation request before submitting it to Risk Management. You prioritize Risk-Adjusted Returns."
        )
        self.rl_model = None
        if RL_AVAILABLE and os.path.exists(MODEL_PATH):
            print(f"[{self.name}] Loading RL Optimization Model...")
            try:
                self.rl_model = PPO.load(MODEL_PATH)
            except Exception as exc:
                print(f"[{self.name}] Failed to load RL model '{MODEL_PATH}': {exc}")
                self.rl_model = None

    def _extract_rl_state(self, state: AgentState) -> np.ndarray:
        """
        Builds the 6-dimension state array for the RL model exactly matching the Env.
        [Volatility, Quant, Fund, Macro, Sentiment, Drawdown]
        """
        vol = 0.1 # Default or extract from technicals if piped
        
        # Simplified mapping from text actions to [-1.0, 1.0] signals
        def _map_act(acts: list) -> float:
            if not acts: return 0.0
            score = sum(1.0 if "LONG" in a or "BUY" in a else -1.0 if "SHORT" in a or "SELL" in a else 0.0 for a in acts)
            return score / len(acts)

        signals = state.get("research_signals", [])
        
        q_act = [s.get("action") for s in signals if "Quant" in s.get("agent_name", "")]
        f_act = [s.get("action") for s in signals if "Fundamental" in s.get("agent_name", "")]
        m_act = [s.get("action") for s in signals if "Macro" in s.get("agent_name", "")]
        s_act = [s.get("action") for s in signals if "Sentiment" in s.get("agent_name", "")]

        drawdown = state.get("portfolio_state", {}).get("current_drawdown", 0.0)

        # Normalization clipping just in case
        return np.array([
            np.clip(vol, 0.0, 1.0),
            np.clip(_map_act(q_act), -1.0, 1.0),
            np.clip(_map_act(f_act), -1.0, 1.0),
            np.clip(_map_act(m_act), -1.0, 1.0),
            np.clip(_map_act(s_act), -1.0, 1.0),
            np.clip(drawdown, 0.0, 1.0)
        ], dtype=np.float32)

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Sizing allocation for {state['current_asset']}...")
        
        committee_decision = state.get("committee_decision", {})
        portfolio_state = state.get("portfolio_state", {})
        
        rl_weight_suggestion = None
        if self.rl_model:
            obs = self._extract_rl_state(state)
            action, _states = self.rl_model.predict(obs, deterministic=True)
            rl_weight_suggestion = action[0]
            print(f"[{self.name}] RL Optimizer suggests target weight: {rl_weight_suggestion:.4f}")
        
        prompt = f"""
        The Strategy Committee has proposed the following trade for {state['current_asset']}:
        {json.dumps(committee_decision, indent=2)}
        
        Current portfolio constraints:
        {json.dumps(portfolio_state, indent=2)}
        """
        
        if rl_weight_suggestion is not None:
            prompt += f"\n\nCRITICAL: The RL Strategy Optimizer algorithm has mathematically determined the optimal target exposure weight for this asset is {rl_weight_suggestion:.4f} (where negative means short, positive means long). You MUST adopt this exposure weight. Provide the rationale justifying this mathematical model choice."
        else:
            prompt += "\n\nDetermine the sizing or allocation percentage based on your LLM intuition."
            
        prompt += """
        \nProduce a JSON output matching this schema:
        {{
            "agent_name": "Portfolio_Manager",
            "action": "REQUEST_ALLOCATION",
            "confidence": float (0.0 to 1.0),
            "rationale": "Explanation of the sizing strategy used.",
            "target_exposure_pct": float (0.0 to 1.0, representing absolute % of total portfolio AUM),
            "stop_loss_pct": float (0.0 to 1.0, relative distance from entry)
        }}
        """

        committee_confidence = float(committee_decision.get("confidence", 0.0) or 0.0)
        base_exposure = abs(float(rl_weight_suggestion)) if rl_weight_suggestion is not None else min(0.25, 0.05 + committee_confidence * 0.2)
        fallback = {
            "agent_name": self.name,
            "action": "REQUEST_ALLOCATION",
            "confidence": round(max(0.1, committee_confidence), 4),
            "rationale": "Deterministic fallback sizing based on committee confidence and RL signal when available.",
            "target_exposure_pct": round(base_exposure, 4),
            "stop_loss_pct": 0.05,
        }

        decision = self._invoke_llm_json(prompt, fallback)
        
        # Override target exposure physically to guarantee the RL agent runs the show if available
        if rl_weight_suggestion is not None:
            decision["target_exposure_pct"] = round(abs(float(rl_weight_suggestion)), 4)
            # The direction is technically derived from the sign, but the Committee sets Direction 
            # in our pipeline. We can pass the sign into the metadata to ensure overrides.
            decision["rl_direction"] = "LONG" if rl_weight_suggestion >= 0 else "SHORT"
            
        return {"allocation_request": decision}

pm_agent = PMAgent()
