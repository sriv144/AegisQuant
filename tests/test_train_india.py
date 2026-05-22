import numpy as np
import pandas as pd

from src.backtest.historical_env import HistoricalHedgeFundEnv


def test_historical_env_observation_and_step_contract():
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    close = np.linspace(100.0, 120.0, len(dates)) + np.sin(np.arange(len(dates)))
    df = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(dates), 1_000_000),
        },
        index=dates,
    )

    env = HistoricalHedgeFundEnv(df, ticker="TEST.NS")
    assert env.observation_space.shape == (14,)
    assert env.action_space.shape == (1,)

    obs, info = env.reset()
    assert obs.shape == (14,)
    assert isinstance(info, dict)

    next_obs, reward, done, truncated, step_info = env.step(env.action_space.sample())
    assert next_obs.shape == (14,)
    assert isinstance(float(reward), float)
    assert done in {True, False}
    assert truncated in {True, False}
    assert isinstance(step_info, dict)
