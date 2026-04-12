"""
MultiAssetEnv
=============
Gymnasium environment for managing a multi-asset portfolio natively using PPO.

Action space: `[-1, 1]` for `N` assets. Softmax or normalization will bound 
the total gross exposure to a realistic level (e.g. max 150%).

Features implemented:
- Multi-dimensional observation space flattening multiple tickers' technicals.
- Average pairwise correlation penalty.
- Market impact / transaction cost matrix implementation. 
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Any
from src.engine.cost_model import cost_model

class MultiAssetEnv(gym.Env):
    metadata = {"render.modes": ["console"]}
    
    # 14 features per asset (12 technical/regime + 2 macro)
    FEATURES_PER_ASSET = 14
    
    def __init__(
        self, 
        df_dict: Dict[str, pd.DataFrame], 
        tickers: List[str],
        initial_balance: float = 1_000_000.0,
        regime_detectors: Optional[Dict[str, Any]] = None,
        max_gross_exposure: float = 1.5
    ):
        super().__init__()
        
        self.df_dict = df_dict
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.initial_balance = initial_balance
        self.max_gross_exposure = max_gross_exposure
        self.regime_detectors = regime_detectors or {}
        
        # We align all dataframes to a common DateTime index (inner join) to step simultaneously
        self._align_data()
        self.n_steps = len(self.common_index)
        
        # State space expansion: 12 features * N assets
        self.obs_dim = self.FEATURES_PER_ASSET * self.num_assets
        
        # Action space: [-1.0, 1.0] weight for each asset in the universe
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(self.obs_dim,), dtype=np.float32)
        
        # Pre-process regimes for speed
        self._precompute_regimes()
        
        self.reset()
        
    def _align_data(self):
        """Align all asset features strictly by date."""
        dfs = []
        for tick in self.tickers:
            df = self.df_dict[tick].copy()
            # Ensure DateTime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
            dfs.append(df)
            
        # Intersect indexes
        common_idx = dfs[0].index
        for d in dfs[1:]:
            common_idx = common_idx.intersection(d.index)
            
        self.common_index = common_idx.sort_values()
        
        # Slice all
        self.features_dict = {}
        for tick, d in zip(self.tickers, dfs):
            self.features_dict[tick] = d.loc[self.common_index]
            
    def _precompute_regimes(self):
        """Precompute the HMM regime class for each asset across all steps."""
        self.regimes_dict = {}
        for tick in self.tickers:
            det = self.regime_detectors.get(tick)
            if det is not None and getattr(det, "_is_fitted", False):
                self.regimes_dict[tick] = det.predict(self.features_dict[tick])
            else:
                self.regimes_dict[tick] = np.zeros(self.n_steps, dtype=int)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_step = 0
        self.drawdown = 0.0
        
        self.current_weights = np.zeros(self.num_assets, dtype=np.float32)
        self._daily_returns = []
        self._asset_returns_window = [] # For correlation penalty calculation
        
        return self._get_obs(), {}
        
    def _get_obs(self) -> np.ndarray:
        if self.current_step >= self.n_steps:
             return np.zeros(self.obs_dim, dtype=np.float32)
             
        obs_blocks = []
        for i, tick in enumerate(self.tickers):
            row = self.features_dict[tick].iloc[self.current_step]
            
            def safe(key: str, default: float = 0.0) -> float:
                val = row.get(key, default)
                return default if (pd.isna(val) or val is None) else float(val)
                
            vol = float(np.clip(safe("Volatility_20_Z"), -2.0, 2.0))
            rsi = float(np.clip(safe("RSI_14_Z"), -2.0, 2.0))
            macd = float(np.clip(safe("MACD_Z"), -2.0, 2.0))
            bb = float(np.clip(safe("BB_Position_Z"), -2.0, 2.0))
            mom = float(np.clip(safe("mom_12m_Z"), -2.0, 2.0))
            vix_z = float(np.clip(safe("vix_z"), -3.0, 3.0))
            yc_slope = float(np.clip(safe("yield_curve_slope"), -5.0, 5.0))
            
            # Asset specific state memory
            curr_w = float(np.clip(self.current_weights[i], -1.0, 1.0))
            dd_v = float(np.clip(self.drawdown, 0.0, 2.0))
            
            regime_id = int(self.regimes_dict[tick][self.current_step])
            regime_onehot = np.zeros(4, dtype=np.float32)
            regime_onehot[regime_id % 4] = 1.0
            
            if len(self._daily_returns) >= 5:
                port_ret_5d = float(np.clip(np.mean(self._daily_returns[-5:]), -2.0, 2.0))
            else:
                port_ret_5d = 0.0
                
            asset_obs = np.array([
                vol, rsi, macd, bb, mom, curr_w, dd_v,
                regime_onehot[0], regime_onehot[1], regime_onehot[2], regime_onehot[3],
                port_ret_5d, vix_z, yc_slope
            ], dtype=np.float32)
            
            obs_blocks.append(asset_obs)
            
        obs = np.concatenate(obs_blocks)
        return np.nan_to_num(obs, nan=0.0)

    def _compute_correlation_penalty(self) -> float:
        """
        Penalizes holding positions that are highly historically correlated in the same direction.
        Requires at least 20 days of track record for stability.
        """
        if len(self._asset_returns_window) < 20 or self.num_assets < 2:
            return 0.0
            
        ret_matrix = np.array(self._asset_returns_window[-60:]).T # Shape (num_assets, window)
        corr_matrix = np.corrcoef(ret_matrix)
        
        # Replace NaNs with 0 (e.g. constant return arrays)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Compute portfolio variance proxy `w.T * Cov * w`
        # Because we penalize total absolute correlation irrespective of beta:
        w = self.current_weights
        port_variance = w.T @ corr_matrix @ w
        
        # Normalize it by maximum possible variance sum(abs(weights))^2
        gross = np.sum(np.abs(w))
        if gross == 0:
            return 0.0
            
        normalized_variance = port_variance / (gross ** 2)
        
        # If variance > threshold (e.g., highly correlated Longs), penalize
        if normalized_variance > 0.5:
            return -0.2 * normalized_variance
        return 0.0

    def step(self, action):
        target_weights = np.clip(action, -1.0, 1.0)
        
        # Limit gross exposure (L1 Norm)
        gross_exposure = np.sum(np.abs(target_weights))
        if gross_exposure > self.max_gross_exposure:
            target_weights = target_weights * (self.max_gross_exposure / gross_exposure)
            
        # Determine actual returns for this step
        next_step = min(self.current_step + 1, self.n_steps - 1)
        actual_asset_returns = np.zeros(self.num_assets)
        total_costs = 0.0
        
        # Step-wise extraction
        for i, tick in enumerate(self.tickers):
            row = self.features_dict[tick].iloc[self.current_step]
            next_row = self.features_dict[tick].iloc[next_step]
            
            close_now = row.get("close", 1.0)
            close_now = close_now if close_now > 0 else 1.0
            
            close_next = next_row.get("close", 1.0)
            actual_asset_returns[i] = (close_next - close_now) / close_now
            
            # Costs
            w_diff = abs(target_weights[i] - self.current_weights[i])
            trade_notional = self.balance * w_diff
            qty = trade_notional / close_now
            
            adv = float(row.get("adv_20", 10_000_000.0))
            cost, _ = cost_model.compute_cost(price=float(close_now), quantity=qty, adv=max(adv, 1.0), algo="TWAP", ticker=tick)
            total_costs += cost
            
        self._asset_returns_window.append(actual_asset_returns)

        # Portfolio Update
        prev_balance = self.balance
        prev_weights = self.current_weights.copy()
        step_gross_return = np.dot(self.current_weights, actual_asset_returns)
        self.balance += (self.balance * step_gross_return) - total_costs
        self.balance = max(self.balance, 1.0)

        self.current_weights = target_weights
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.drawdown = (self.peak_balance - self.balance) / self.peak_balance
        
        step_return = (self.balance - prev_balance) / prev_balance
        self._daily_returns.append(step_return)
        
        # ── Multi-component reward ──────────────────────
        log_ret = float(np.log(max(self.balance, 1e-8) / prev_balance))

        dd_penalty = -2.0 * max(0.0, self.drawdown - 0.05)
        cost_penalty = -(total_costs / prev_balance) if prev_balance > 0 else 0.0

        concentration_penalty = -0.1 * np.sum(np.abs(target_weights) ** 2)
        correlation_penalty = self._compute_correlation_penalty()

        # Turnover penalty: directly penalises L1 weight change each step.
        # This is the primary lever to reduce excessive trading — the cost_penalty
        # alone is too small relative to the return signal to discourage churning.
        total_weight_change = float(np.sum(np.abs(target_weights - prev_weights)))
        turnover_penalty = -0.003 * total_weight_change

        if len(self._daily_returns) >= 5:
            vol_penalty = -0.5 * float(np.std(self._daily_returns[-5:]))
        else:
            vol_penalty = 0.0

        reward = log_ret + dd_penalty + cost_penalty + concentration_penalty + vol_penalty + correlation_penalty + turnover_penalty
        reward = float(np.clip(reward, -10.0, 10.0))
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        truncated = False
        
        if self.balance <= self.initial_balance * 0.5:
             done = True
             reward -= 100.0
             
        info = {
            "balance": self.balance,
            "step_return": step_return,
            "drawdown": self.drawdown,
            "transaction_cost": total_costs,
            "date": str(self.common_index[self.current_step] if self.current_step < len(self.common_index) else "")
        }
        
        return self._get_obs(), reward, done, truncated, info
