"""
HistoricalHedgeFundEnv
======================
A Gymnasium environment that replays real historical OHLCV data instead of
generating synthetic random observations.  Used by the walk-forward engine to
train and evaluate PPO on actual market data.

State vector (14-D):
    [0]  Volatility_20_Z     — normalised short-term volatility
    [1]  RSI_14_Z            — momentum oscillator z-score
    [2]  MACD_Z              — trend-following signal z-score
    [3]  BB_Position_Z       — mean-reversion signal z-score
    [4]  mom_12m_Z           — 12-month price momentum z-score
    [5]  current_weight      — portfolio state: current position in [-1, 1]
    [6]  drawdown            — current portfolio drawdown from peak
    [7]  regime_0            — Bull Quiet one-hot (from HMM)
    [8]  regime_1            — Bull Volatile one-hot (from HMM)
    [9]  regime_2            — Bear Quiet one-hot (from HMM)
    [10] regime_3            — Bear Volatile one-hot (from HMM)
    [11] portfolio_return_5d — 5-step rolling mean portfolio return
    [12] vix_z               — normalised VIX (rolling 63-day z-score); 0 if not provided
    [13] yield_curve_slope   — 10Y minus 3M Treasury spread; 0 if not provided

Action: target weight in [-1, 1].  Reward uses a 5-component formula with
regime-adaptive drawdown threshold.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.data.feature_engineering import feature_engineer
from src.engine.cost_model import cost_model


class HistoricalHedgeFundEnv(gym.Env):
    """
    Replays a historical OHLCV DataFrame step by step.

    Args:
        df:               DataFrame with columns [open, high, low, close, volume].
                          Must be sorted chronologically and have a DatetimeIndex.
        initial_balance:  Starting capital in dollars.
        ticker:           Ticker label (used by cost model for spread category).
        regime_detector:  Optional fitted RegimeDetector instance. If None,
                          regime one-hot defaults to [1,0,0,0] (Bull Quiet).
    """

    metadata = {"render.modes": ["console"]}

    FALLBACK_ADV = 10_000_000.0
    OBS_DIM = 14

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000.0,
        ticker: str = "SPY",
        regime_detector=None,
        macro_df: pd.DataFrame = None,
    ):
        super().__init__()

        self.ticker = ticker
        self.initial_balance = initial_balance

        # Pre-compute all features once
        price_records = self._df_to_records(df, ticker)
        self.features = feature_engineer.compute_technical_indicators(price_records)
        self.features = self.features.dropna(subset=["Volatility_20"]).reset_index()

        # Align macro features (VIX + yield curve) by date if provided
        if macro_df is not None and not macro_df.empty:
            macro_df.index = pd.to_datetime(macro_df.index)
            self.features["date_idx"] = pd.to_datetime(self.features.get("timestamp", self.features.index))
            merged = self.features.merge(
                macro_df[["vix_z", "yield_curve_slope"]],
                left_on="date_idx",
                right_index=True,
                how="left",
            )
            self.features["vix_z"] = merged["vix_z"].values
            self.features["yield_curve_slope"] = merged["yield_curve_slope"].values
        else:
            self.features["vix_z"] = 0.0
            self.features["yield_curve_slope"] = 0.0

        self.features[["vix_z", "yield_curve_slope"]] = (
            self.features[["vix_z", "yield_curve_slope"]].bfill().ffill().fillna(0.0)
        )

        # Compute 12-month (252-day) momentum and its rolling z-score
        self.features["mom_12m"] = self.features["close"].pct_change(252)
        self.features["mom_12m_Z"] = feature_engineer._rolling_zscore(
            self.features["mom_12m"], window=63
        )
        self.features = self.features.bfill().ffill()

        # Precompute ADV (20-day avg dollar volume) for cost model
        self.features["adv_20"] = (
            self.features["close"] * self.features["volume"]
        ).rolling(20).mean().bfill()

        self.n_steps = len(self.features)

        # Regime detection — run once over the full feature table
        if regime_detector is not None and hasattr(regime_detector, "predict"):
            self.regimes = regime_detector.predict(self.features)
        else:
            self.regimes = np.zeros(self.n_steps, dtype=int)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.OBS_DIM,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_step = 0
        self.current_weight = 0.0
        self.drawdown = 0.0
        self._daily_returns: list[float] = []
        self._weights_log: list[float] = []
        return self._get_obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, action):
        target_weight = float(np.clip(action[0], -1.0, 1.0))
        row = self.features.iloc[self.current_step]
        next_row = self.features.iloc[min(self.current_step + 1, self.n_steps - 1)]

        # Current regime (for adaptive reward)
        regime = int(self.regimes[self.current_step]) if self.current_step < len(self.regimes) else 0

        # Actual market return for this bar
        actual_return = (next_row["close"] - row["close"]) / row["close"]

        # Transaction cost (charge on weight change only)
        weight_change = abs(target_weight - self.current_weight)
        trade_notional = self.balance * weight_change
        trade_qty = trade_notional / row["close"] if row["close"] > 0 else 0.0
        adv = float(row.get("adv_20", self.FALLBACK_ADV))
        transaction_cost, _ = cost_model.compute_cost(
            price=float(row["close"]),
            quantity=trade_qty,
            adv=max(adv, 1.0),
            algo="TWAP",
            ticker=self.ticker,
        )

        self.current_weight = target_weight
        prev_balance = self.balance
        self.balance += self.balance * target_weight * actual_return - transaction_cost
        self.balance = max(self.balance, 1.0)  # floor to avoid log(0)

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        self.drawdown = (self.peak_balance - self.balance) / self.peak_balance

        # Log for metrics
        step_return = (self.balance - prev_balance) / prev_balance
        self._daily_returns.append(step_return)
        self._weights_log.append(target_weight)

        # ── Multi-component reward (regime-adaptive) ──────────────────────
        log_ret = float(np.log(max(self.balance, 1e-8) / prev_balance))

        # Bear Volatile (regime 3) gets more lenient drawdown threshold
        dd_threshold = 0.10 if regime == 3 else 0.05
        dd_penalty = -2.0 * max(0.0, self.drawdown - dd_threshold)

        cost_penalty = -(transaction_cost / prev_balance) if prev_balance > 0 else 0.0
        conc_penalty = -0.1 * target_weight ** 2

        # Short-term volatility smoothing penalty (encourages consistent returns)
        if len(self._daily_returns) >= 5:
            recent_vol = float(np.std(self._daily_returns[-5:]))
            vol_penalty = -0.5 * recent_vol
        else:
            vol_penalty = 0.0

        reward = log_ret + dd_penalty + cost_penalty + conc_penalty + vol_penalty
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
            "transaction_cost": transaction_cost,
            "regime": regime,
            "date": str(row.get("timestamp", "")),
        }
        return self._get_obs(), reward, done, truncated, info

    # ---------------------------------------------------------------- helpers
    def _get_obs(self) -> np.ndarray:
        if self.current_step >= self.n_steps:
            return np.zeros(self.OBS_DIM, dtype=np.float32)

        row = self.features.iloc[self.current_step]

        def safe(key: str, default: float = 0.0) -> float:
            val = row.get(key, default)
            return default if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)

        vol_norm = float(np.clip(safe("Volatility_20_Z"), -2.0, 2.0))
        rsi_z = float(np.clip(safe("RSI_14_Z"), -2.0, 2.0))
        macd_z = float(np.clip(safe("MACD_Z"), -2.0, 2.0))
        bb_z = float(np.clip(safe("BB_Position_Z"), -2.0, 2.0))
        fund_sig = float(np.clip(safe("mom_12m_Z"), -2.0, 2.0))

        # Portfolio state
        current_w = float(np.clip(self.current_weight, -1.0, 1.0))
        drawdown_v = float(np.clip(self.drawdown, 0.0, 2.0))

        # Regime one-hot
        regime_id = int(self.regimes[self.current_step]) if self.current_step < len(self.regimes) else 0
        regime_onehot = np.zeros(4, dtype=np.float32)
        regime_onehot[regime_id % 4] = 1.0

        # 5-day rolling portfolio return
        if len(self._daily_returns) >= 5:
            port_ret_5d = float(np.clip(np.mean(self._daily_returns[-5:]), -2.0, 2.0))
        else:
            port_ret_5d = 0.0

        vix_z = float(np.clip(safe("vix_z"), -2.0, 2.0))
        yc_slope = float(np.clip(safe("yield_curve_slope"), -2.0, 2.0))

        obs = np.array(
            [vol_norm, rsi_z, macd_z, bb_z, fund_sig,
             current_w, drawdown_v,
             regime_onehot[0], regime_onehot[1], regime_onehot[2], regime_onehot[3],
             port_ret_5d, vix_z, yc_slope],
            dtype=np.float32,
        )
        # Final safety: replace any remaining NaN/inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=2.0, neginf=-2.0)
        return obs

    def get_returns(self) -> np.ndarray:
        """Returns the daily return series accumulated so far."""
        return np.array(self._daily_returns, dtype=float)

    def get_weights(self) -> np.ndarray:
        return np.array(self._weights_log, dtype=float)

    @staticmethod
    def _df_to_records(df: pd.DataFrame, ticker: str) -> list:
        """Convert a OHLCV DataFrame to the list-of-dicts format expected by feature_engineer."""
        records = []
        for ts, row in df.iterrows():
            records.append({
                "timestamp": str(ts),
                "ticker": ticker,
                "open": float(row.get("open", row.get("Open", 0))),
                "high": float(row.get("high", row.get("High", 0))),
                "low": float(row.get("low", row.get("Low", 0))),
                "close": float(row.get("close", row.get("Close", 0))),
                "volume": float(row.get("volume", row.get("Volume", 0))),
            })
        return records

    def render(self):
        regime_name = ["Bull Quiet", "Bull Volatile", "Bear Quiet", "Bear Volatile"]
        rid = int(self.regimes[min(self.current_step, self.n_steps - 1)])
        print(
            f"Step {self.current_step}/{self.n_steps} | "
            f"Balance: ${self.balance:,.0f} | DD: {self.drawdown*100:.1f}% | "
            f"Regime: {regime_name[rid % 4]}"
        )
