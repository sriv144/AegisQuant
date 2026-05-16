import json
import os
import sys
import numpy as np
from typing import Dict, Any
from pathlib import Path
from gymnasium.spaces import Box
from ..base_agent import BaseAgent
from ..state import AgentState

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

MODEL_PATH = "ppo_portfolio_manager.zip"
FALLBACK_MODEL_GLOB = "models/*.zip"

class PMAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Portfolio_Manager",
            role="Portfolio Manager. You receive approved signals from the Strategy Committee and output the initial sizing/allocation request before submitting it to Risk Management. You prioritize Risk-Adjusted Returns."
        )
        self.rl_model = None
        if RL_AVAILABLE:
            self.rl_model = self._load_rl_model()

    def _load_rl_model(self):
        from src.models.registry import ModelRegistry
        registry = ModelRegistry()
        prod_path = registry.get_production_model()

        # Priority: registry production → curriculum model → walk-forward latest → legacy
        candidates = []
        if prod_path:
            candidates.append(Path(str(prod_path)))
        curriculum_nsei = Path("ppo_curriculum_^NSEI.zip")
        if curriculum_nsei.exists():
            candidates.append(curriculum_nsei)
        wf_models = sorted(Path().glob("models/wf_w*_ppo.zip"))
        if wf_models:
            candidates.append(wf_models[-1])
        candidates.append(Path(MODEL_PATH))

        action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        import numpy.core
        import numpy.core.numeric
        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)

        for model_path in candidates:
            if not model_path.exists():
                continue

            print(f"[{self.name}] Loading RL model from {model_path}...")
            try:
                model = PPO.load(str(model_path))
                obs_dim = model.observation_space.shape[0]
                self._obs_dim = obs_dim
                print(f"[{self.name}] Loaded OK — obs_dim={obs_dim}")
                return model
            except Exception:
                pass

            for obs_dim in (14, 6):
                obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                try:
                    model = PPO.load(
                        str(model_path),
                        custom_objects={
                            "observation_space": obs_space,
                            "action_space": action_space,
                        },
                    )
                    self._obs_dim = obs_dim
                    print(f"[{self.name}] Loaded OK (forced obs_dim={obs_dim})")
                    return model
                except Exception as exc:
                    print(f"[{self.name}] Failed obs_dim={obs_dim} for '{model_path}': {exc}")

        return None

    def _extract_rl_state(self, state: AgentState) -> np.ndarray:
        """
        Build observation vector matching the loaded model's expected dimension.
        14-D: matches HistoricalHedgeFundEnv (walk-forward / curriculum models).
        6-D:  matches legacy HedgeFundEnv (random-noise models).
        """
        obs_dim = getattr(self, "_obs_dim", 6)
        ti = state.get("technical_indicators", {})
        pf = state.get("portfolio_state", {})
        drawdown = pf.get("current_drawdown", 0.0)

        if obs_dim == 14:
            vix_raw = pf.get("vix_raw", 20.0)
            return np.array([
                np.clip(ti.get("Volatility_20_Z", 0.0), -1.0, 1.0),
                np.clip(ti.get("RSI_14_Z", 0.0), -1.0, 1.0),
                np.clip(ti.get("MACD_Z", 0.0), -1.0, 1.0),
                np.clip(ti.get("BB_Position_Z", 0.0), -1.0, 1.0),
                np.clip(ti.get("mom_12m_Z", 0.0), -1.0, 1.0),
                0.0,  # current_weight — 0 for new position
                np.clip(drawdown, 0.0, 1.0),
                1.0, 0.0, 0.0, 0.0,  # regime one-hot default: Bull Quiet
                0.0,  # portfolio_return_5d
                np.clip((vix_raw - 20.0) / 10.0, -1.0, 1.0),  # vix_z approx
                0.0,  # yield_curve_slope
            ], dtype=np.float32)

        def _map_act(acts: list) -> float:
            if not acts:
                return 0.0
            score = sum(1.0 if "LONG" in a or "BUY" in a else -1.0 if "SHORT" in a or "SELL" in a else 0.0 for a in acts)
            return score / len(acts)

        signals = state.get("research_signals", [])
        q_act = [s.get("action") for s in signals if "Quant" in s.get("agent_name", "")]
        f_act = [s.get("action") for s in signals if "Fundamental" in s.get("agent_name", "")]
        m_act = [s.get("action") for s in signals if "Macro" in s.get("agent_name", "")]
        s_act = [s.get("action") for s in signals if "Sentiment" in s.get("agent_name", "")]
        vol = np.clip(ti.get("Volatility_20_Z", 0.1), 0.0, 1.0)

        return np.array([
            vol,
            np.clip(_map_act(q_act), -1.0, 1.0),
            np.clip(_map_act(f_act), -1.0, 1.0),
            np.clip(_map_act(m_act), -1.0, 1.0),
            np.clip(_map_act(s_act), -1.0, 1.0),
            np.clip(drawdown, 0.0, 1.0),
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
            prompt += f"\n\nAdvisory: The RL Strategy Optimizer suggests a target exposure weight of {rl_weight_suggestion:.4f}. Consider this as one input among many, but make your own sizing decision based on the committee signal strength and risk constraints."
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

        # RL override: only if USE_RL_OVERRIDE=true (default: disabled)
        # When disabled, RL weight is advisory-only metadata — consensus engine drives allocation
        use_rl_override = os.getenv("USE_RL_OVERRIDE", "false").lower() == "true"
        if rl_weight_suggestion is not None and use_rl_override:
            decision["target_exposure_pct"] = round(abs(float(rl_weight_suggestion)), 4)
            decision["rl_direction"] = "LONG" if rl_weight_suggestion >= 0 else "SHORT"
            print(f"[{self.name}] RL OVERRIDE active — forcing exposure to {decision['target_exposure_pct']:.4f}")
        elif rl_weight_suggestion is not None:
            # Store RL suggestion as metadata only
            decision["rl_advisory_weight"] = round(float(rl_weight_suggestion), 4)

        return {"allocation_request": decision}

pm_agent = PMAgent()
