"""
Live Online Retraining
======================
Monthly warm-start retraining: loads the production PPO model, replays the last
30 days of real market data through HistoricalHedgeFundEnv, and saves a candidate
model for A/B promotion via ModelRegistry.
"""
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from stable_baselines3 import PPO, SAC
    has_sb3 = True
except ImportError:
    has_sb3 = False

logger = logging.getLogger(__name__)

_ALGO_MAP = {"ppo": PPO, "sac": SAC}


class OnlineRetrainer:
    def __init__(
        self,
        model_registry_path: str = "model_registry",
        retrain_timesteps: int = 5_000,
        lookback_days: int = 30,
    ):
        self.model_registry_path = Path(model_registry_path)
        self.retrain_timesteps = retrain_timesteps
        self.lookback_days = lookback_days

    def warm_start_retrain(
        self,
        active_model_path: str,
        ticker: str = "SPY",
        algo: str = "ppo",
    ) -> Optional[str]:
        """
        Loads the active model and continues training on the most recent
        `lookback_days` of real market data via HistoricalHedgeFundEnv.

        Returns the path to the retrained candidate model, or None on failure.
        """
        if not has_sb3:
            logger.error("stable-baselines3 not installed.")
            return None

        # ── 1. Fetch last N days of OHLCV ──────────────────────────────────
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=self.lookback_days + 30)).strftime("%Y-%m-%d")

        try:
            import yfinance as yf
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if df.empty:
                logger.warning("yfinance returned no data for %s. Aborting retrain.", ticker)
                return None
            if isinstance(df.columns, object) and hasattr(df.columns, "get_level_values"):
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
        except Exception as exc:
            logger.error("Data fetch failed: %s", exc)
            return None

        # ── 2. Build environment ────────────────────────────────────────────
        try:
            from src.backtest.historical_env import HistoricalHedgeFundEnv
            from src.engine.regime_detector import RegimeDetector
            from src.data.market_data import market_data

            regime_det = None
            regime_pkl = Path("models/regime_detector.pkl")
            if regime_pkl.exists():
                regime_det = RegimeDetector()
                regime_det.load(str(regime_pkl))

            macro_df = None
            try:
                macro_df = market_data.get_macro_data(start_date=start_date)
            except Exception:
                pass

            env = HistoricalHedgeFundEnv(
                df=df,
                ticker=ticker,
                regime_detector=regime_det,
                macro_df=macro_df,
            )
        except Exception as exc:
            logger.error("Failed to build HistoricalHedgeFundEnv: %s", exc)
            return None

        # ── 3. Warm-start: load model and continue training ─────────────────
        AlgoClass = _ALGO_MAP.get(algo.lower(), PPO)
        try:
            model = AlgoClass.load(active_model_path, env=env)
            print(
                f"\n[Retrainer] Warm-starting {algo.upper()} from {active_model_path} "
                f"on {len(df)} days of {ticker} data "
                f"({self.retrain_timesteps:,} timesteps)."
            )
            model.learn(total_timesteps=self.retrain_timesteps, reset_num_timesteps=False)
        except Exception as exc:
            logger.error("Retraining failed: %s", exc)
            return None

        # ── 4. Save candidate alongside original ───────────────────────────
        new_path = active_model_path.replace(".zip", "_retrained.zip")
        model.save(new_path)
        print(f"[Retrainer] Retrained candidate saved at {new_path}")
        return new_path
