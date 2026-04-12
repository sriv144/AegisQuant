import pytest
import numpy as np
import pandas as pd
from src.backtest.multi_asset_env import MultiAssetEnv

@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=100)
    df1 = pd.DataFrame({
        "close": np.linspace(100, 110, 100),
        "Volatility_20_Z": np.zeros(100),
        "RSI_14_Z": np.zeros(100),
        "MACD_Z": np.zeros(100),
        "BB_Position_Z": np.zeros(100),
        "mom_12m_Z": np.zeros(100),
        "vix_z": np.ones(100),
        "yield_curve_slope": np.ones(100),
        "adv_20": np.ones(100) * 1e6
    }, index=dates)
    
    df2 = pd.DataFrame({
        "close": np.linspace(50, 40, 100),
        "Volatility_20_Z": np.zeros(100),
        "RSI_14_Z": np.zeros(100),
        "MACD_Z": np.zeros(100),
        "BB_Position_Z": np.zeros(100),
        "mom_12m_Z": np.zeros(100),
        "vix_z": np.zeros(100),
        "yield_curve_slope": np.zeros(100),
        "adv_20": np.ones(100) * 1e6
    }, index=dates)
    
    return {"A": df1, "B": df2}

def test_multi_asset_env_init(dummy_data):
    env = MultiAssetEnv(dummy_data, tickers=["A", "B"])
    
    # Check space shapes
    # 14 features per asset * 2 assets = 28
    assert env.obs_dim == 28
    assert env.action_space.shape == (2,)
    
def test_multi_asset_env_step(dummy_data):
    env = MultiAssetEnv(dummy_data, tickers=["A", "B"])
    obs, info = env.reset()
    
    assert obs.shape == (28,)
    
    # Try step
    action = np.array([0.5, -0.5]) # Long A, Short B
    obs_next, reward, done, truncated, info = env.step(action)
    
    assert "balance" in info
    assert not done
    assert obs_next.shape == (28,)
    
    # Gross exposure cap test
    env.max_gross_exposure = 1.0
    action_gross = np.array([1.0, 1.0])
    _, _, _, _, _ = env.step(action_gross)
    # Target weights should be 0.5 and 0.5 internally
    np.testing.assert_array_almost_equal(env.current_weights, np.array([0.5, 0.5]))
