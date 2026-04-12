"""
Agent Attribution & Sensitivity Analysis
======================================
Profiles trained RL models to determine exact attribution of which inputs 
(Macro Agents, Sentiment, VIX, etc.) drove the trading decisions in 
different HMM Regimes.
"""

import numpy as np
import pandas as pd
import shap
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "Volatility_20", "RSI_14", "MACD", "BB_Position", "Mom_12m", 
    "Current_Weight", "Drawdown", 
    "Regime_Bull_Quiet", "Regime_Bull_Vol", "Regime_Bear_Quiet", "Regime_Bear_Vol",
    "Port_Return_5d", "VIX_Z", "Yield_Curve_Slope"
]

class AgentAttributionEngine:
    def __init__(self, model_path: str, save_dir: str = "backtest_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load the stable-baselines3 model
        try:
            from stable_baselines3 import PPO, SAC, TD3
            import zipfile
            
            # Auto-detect algorithm from name if possible
            if "sac" in model_path.lower():
                self.model = SAC.load(model_path)
            elif "td3" in model_path.lower():
                self.model = TD3.load(model_path)
            else:
                self.model = PPO.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model = None

    def compute_shap_importance(self, historical_states: np.ndarray, num_samples: int = 100):
        """
        Uses SHAP to compute the global feature importance of the observation vector.
        Requires a batch of historical states (e.g., from the validation set).
        """
        if self.model is None or historical_states is None or len(historical_states) == 0:
            return {}
            
        print("[Attribution] Running SHAP TreeExplainer over agent policy...")
        
        # We sample a background dataset to compute expected values
        sample_size = min(len(historical_states), num_samples)
        background = historical_states[np.random.choice(len(historical_states), sample_size, replace=False)]
        
        # Define a wrapper for SHAP that outputs the deterministic action
        def model_predict(obs):
            # predict returns (actions, states), we just want actions
            actions = [self.model.predict(o, deterministic=True)[0] for o in obs]
            return np.array(actions)
            
        try:
            # Mask explainer for continuous / black-box models
            explainer = shap.Explainer(model_predict, background)
            shap_values = explainer(background)
            
            # Compute mean absolute SHAP value per feature
            # SHAP values shape depends on action dim, we take mean across actions
            if len(shap_values.shape) > 2:
                mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
            else:
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                
            # If the state space is N_assets * FEATURES_PER_ASSET
            # we average the importance across all assets for the base features
            num_features = len(FEATURE_NAMES)
            if len(mean_abs_shap) > num_features:
                reshaped = mean_abs_shap.reshape(-1, num_features)
                aggregated_shap = reshaped.mean(axis=0)
            else:
                aggregated_shap = mean_abs_shap
                
            importance_dict = {feat: float(val) for feat, val in zip(FEATURE_NAMES, aggregated_shap)}
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
            
            # Save to JSON
            out_path = self.save_dir / "shap_feature_importance.json"
            with open(out_path, "w") as f:
                json.dump(importance_dict, f, indent=4)
                
            self._plot_feature_importance(importance_dict)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"SHAP Explainer computation failed: {e}")
            return {}
            
    def _plot_feature_importance(self, importance_dict: dict):
        plt.figure(figsize=(10, 8))
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        # Reverse to have highest on top
        features.reverse()
        values.reverse()
        
        plt.barh(features, values, color="royalblue")
        plt.xlabel("Mean Absolute SHAP Value (Impact on Target Weight)")
        plt.title("AegisQuant Agent Attribution (Global State Importance)")
        plt.tight_layout()
        plt.savefig(self.save_dir / "shap_attribution_plot.png", dpi=300)
        plt.close()
