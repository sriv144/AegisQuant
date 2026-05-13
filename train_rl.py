"""
train_rl.py
===========
Train the PPO portfolio manager.  Default mode uses curriculum training on
real historical data (Nifty50).  Use --synthetic for the old random-noise env.

  Default (curriculum on real data):
      python train_rl.py
      python train_rl.py --ticker ^NSEI --timesteps 80000

  Synthetic env (for debugging only):
      python train_rl.py --synthetic

Curriculum learning stages:
  Stage 1 (25% timesteps): last 2 years  — agent learns basic direction trading
  Stage 2 (35% timesteps): last 5 years  — includes 2022 bear market
  Stage 3 (40% timesteps): full history  — includes 2020 crash, 2018 volatility
"""

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from src.engine.rl_env import HedgeFundEnv

MODEL_PATH = "ppo_portfolio_manager.zip"
CURRICULUM_MODEL_PATH = "ppo_curriculum_{ticker}.zip"

# Global seed — set once here so every run is reproducible.
SEED = 42


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def train_agent(seed: int = SEED) -> None:
    print(f"[train] Seed: {seed}")
    _set_global_seeds(seed)

    print("Initializing HedgeFund RL Environment...")
    env = HedgeFundEnv(seed=seed)

    print("Checking environment compatibility with Stable Baselines3...")
    check_env(env)
    print("Environment OK.")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        seed=seed,
    )

    print("Starting training for 50,000 timesteps...")
    model.learn(total_timesteps=50_000)

    print(f"Training complete. Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)


def evaluate_agent(seed: int = SEED) -> None:
    env = HedgeFundEnv(seed=seed)
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_agent() first.")
        return

    model = PPO.load(MODEL_PATH, env=env)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=5, deterministic=True
    )
    print(f"[eval] Seed={seed} | Mean Reward: {mean_reward:.4f} +/- {std_reward:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum training on real historical data
# ─────────────────────────────────────────────────────────────────────────────

def curriculum_train(
    ticker: str = "SPY",
    total_timesteps: int = 60_000,
    seed: int = SEED,
    history_start: str = "2015-01-01",
) -> None:
    """
    3-stage curriculum training on real historical data.

    Stage 1 (25%): last 2 years  — easy, recent clean market
    Stage 2 (35%): last 5 years  — includes 2022 bear market
    Stage 3 (40%): full history  — includes 2020 crash, 2018 drawdown
    """
    _set_global_seeds(seed)
    import yfinance as yf
    from src.backtest.historical_env import HistoricalHedgeFundEnv
    from src.engine.regime_detector import RegimeDetector

    print(f"\n[Curriculum] Fetching {ticker} data from {history_start}...")
    df = yf.download(ticker, start=history_start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Normalise column names (yfinance MultiIndex or flat)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    today = df.index.max()
    stage_windows = [
        ("Stage 1 (last 2y — easy)",    today - pd.DateOffset(years=2), today),
        ("Stage 2 (last 5y — medium)",  today - pd.DateOffset(years=5), today),
        ("Stage 3 (full history — hard)", df.index.min(), today),
    ]
    timestep_fractions = [0.25, 0.35, 0.40]

    # Fit regime detector on the full dataset first
    print("[Curriculum] Fitting HMM regime detector on full history...")
    from src.data.feature_engineering import feature_engineer
    records = []
    for ts, row in df.iterrows():
        records.append({
            "timestamp": str(ts), "ticker": ticker,
            "open": float(row.get("open", 0)), "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)), "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0)),
        })
    full_features = feature_engineer.compute_technical_indicators(records)
    full_features = full_features.dropna(subset=["Volatility_20"])
    full_features["Daily_Return"] = full_features["close"].pct_change().fillna(0)

    detector = RegimeDetector(n_states=4, random_state=seed)
    detector.fit(full_features)
    if detector._is_fitted:
        regime_preview = detector.predict(full_features)
        detector.print_regime_distribution(regime_preview)
        detector.save()
    else:
        print("[Curriculum] Regime detector not fitted (insufficient data) — using default regimes")

    model = None
    save_path = CURRICULUM_MODEL_PATH.format(ticker=ticker)

    for (label, start_date, end_date), frac in zip(stage_windows, timestep_fractions):
        stage_ts = max(100, int(total_timesteps * frac))
        window_df = df.loc[start_date:end_date]
        if len(window_df) < 100:
            print(f"  Skipping {label} — only {len(window_df)} bars")
            continue

        print(f"\n[Curriculum] {label}")
        print(f"  Data: {window_df.index[0].date()} -> {window_df.index[-1].date()} "
              f"({len(window_df)} bars) | Timesteps: {stage_ts:,}")

        env = HistoricalHedgeFundEnv(
            window_df,
            ticker=ticker,
            regime_detector=detector if detector._is_fitted else None,
        )
        env.reset(seed=seed)

        n_steps = min(2048, max(64, len(env.features) // 2))

        if model is None:
            model = PPO(
                "MlpPolicy", env, verbose=0,
                learning_rate=3e-4, n_steps=n_steps, seed=seed,
            )
        else:
            # Warm-start: keep policy weights, swap environment
            model.set_env(env)

        model.learn(total_timesteps=stage_ts, reset_num_timesteps=(model is None))

        # Quick in-sample evaluation
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)
        print(f"  In-sample mean reward: {mean_r:.4f} +/- {std_r:.4f}")

    if model is not None:
        model.save(save_path)
        print(f"\n[Curriculum] Model saved to {save_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AegisQuant PPO Trainer")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic random env (debugging only — does NOT produce a useful model)"
    )
    parser.add_argument("--ticker", default="^NSEI",
                        help="Asset ticker for curriculum mode (default: Nifty50)")
    parser.add_argument("--timesteps", type=int, default=80_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--history-start", default="2015-01-01",
                        help="Earliest date for curriculum data fetch")
    args = parser.parse_args()

    if args.synthetic:
        print("[WARNING] Training on synthetic random data — model will NOT learn real market patterns")
        train_agent(seed=args.seed)
        evaluate_agent(seed=args.seed)
    else:
        curriculum_train(
            ticker=args.ticker,
            total_timesteps=args.timesteps,
            seed=args.seed,
            history_start=args.history_start,
        )
