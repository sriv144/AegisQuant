"""
Train PPO on Indian market data using HistoricalHedgeFundEnv.
Trains on NIFTY50 constituents, saves to model_registry via ModelRegistry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

from src.backtest.historical_env import HistoricalHedgeFundEnv
from src.data.feature_engineering import feature_engineer
from src.engine.regime_detector import RegimeDetector
from src.models.registry import ModelRegistry


DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "SBIN.NS",
    "LT.NS",
    "HCLTECH.NS",
]


@dataclass
class TrainingResult:
    ticker: str
    model_path: Path
    sharpe: float
    total_return: float
    max_drawdown: float


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    rename = {"adj close": "close"}
    df = df.rename(columns=rename)
    needed = ["open", "high", "low", "close", "volume"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return df[needed].dropna().sort_index()


def _download_history(ticker: str, years: int = 4) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years + 30)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No yfinance data returned for {ticker}")
    return _normalise_ohlcv(df)


def _fit_regime_detector(df: pd.DataFrame, ticker: str) -> RegimeDetector:
    records = HistoricalHedgeFundEnv._df_to_records(df, ticker)
    features = feature_engineer.compute_technical_indicators(records).bfill().ffill()
    detector = RegimeDetector()
    try:
        detector.fit(features)
    except ImportError as exc:
        print(f"[train_india] RegimeDetector unavailable ({exc}); using Bull Quiet fallback.")
    return detector


def _evaluate(model: PPO, df: pd.DataFrame, ticker: str, detector: RegimeDetector) -> dict:
    env = HistoricalHedgeFundEnv(df, initial_balance=1_000_000.0, ticker=ticker, regime_detector=detector)
    obs, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, truncated, _info = env.step(action)

    returns = env.get_returns()
    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0
    equity = np.cumprod(1.0 + returns) if len(returns) else np.array([1.0])
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.max((peak - equity) / np.where(peak == 0, 1.0, peak)))
    return {
        "sharpe": sharpe,
        "total_return": float((env.balance / env.initial_balance) - 1.0),
        "max_drawdown": max_drawdown,
        "final_balance": float(env.balance),
    }


def train_ticker(ticker: str, timesteps: int, output_dir: Path) -> TrainingResult:
    print(f"[train_india] Fetching {ticker}...")
    df = _download_history(ticker)
    if len(df) < 252 * 3:
        raise ValueError(f"{ticker} has only {len(df)} rows; need roughly 3+ years.")

    split_date = df.index.max() - pd.DateOffset(months=6)
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    if len(train_df) < 252 or len(test_df) < 40:
        raise ValueError(f"{ticker} does not have enough train/test rows after split.")

    detector = _fit_regime_detector(train_df, ticker)
    env = HistoricalHedgeFundEnv(train_df, initial_balance=1_000_000.0, ticker=ticker, regime_detector=detector)
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=timesteps)

    metrics = _evaluate(model, test_df, ticker, detector)
    model_path = output_dir / f"india_ppo_{ticker.replace('.', '_')}.zip"
    model.save(str(model_path))
    print(
        f"[train_india] {ticker}: OOS Sharpe={metrics['sharpe']:.2f}, "
        f"return={metrics['total_return']:.2%}, drawdown={metrics['max_drawdown']:.2%}"
    )
    return TrainingResult(
        ticker=ticker,
        model_path=model_path,
        sharpe=metrics["sharpe"],
        total_return=metrics["total_return"],
        max_drawdown=metrics["max_drawdown"],
    )


def train_india(tickers: Iterable[str] = DEFAULT_TICKERS) -> str:
    timesteps = int(os.getenv("AEGIS_TRAIN_TIMESTEPS", "50000"))
    output_dir = Path(os.getenv("AEGIS_TRAIN_OUTPUT_DIR", "models/india_training"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[TrainingResult] = []
    for ticker in tickers:
        try:
            results.append(train_ticker(ticker, timesteps, output_dir))
        except Exception as exc:
            print(f"[train_india] Skipping {ticker}: {exc}")

    if not results:
        raise RuntimeError("No Indian PPO models trained successfully.")

    best = max(results, key=lambda item: (item.sharpe, item.total_return))
    registry = ModelRegistry()
    model_id = registry.register_model(
        model_zip_source=str(best.model_path),
        algorithm="PPO",
        oos_metrics={
            "ticker": best.ticker,
            "sharpe": best.sharpe,
            "total_return": best.total_return,
            "max_drawdown": best.max_drawdown,
            "basket_size": len(results),
        },
        hyperparams={"total_timesteps_per_ticker": timesteps, "policy": "MlpPolicy", "obs_dim": 14},
    )
    registry.promote_model(model_id, "production")
    print(f"[train_india] Promoted {model_id} from {best.ticker} to production.")
    return model_id


def main() -> None:
    tickers_raw = os.getenv("AEGIS_TRAIN_TICKERS", "")
    tickers = [item.strip() for item in tickers_raw.split(",") if item.strip()] or DEFAULT_TICKERS
    train_india(tickers)


if __name__ == "__main__":
    main()
