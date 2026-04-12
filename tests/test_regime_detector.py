import pytest
import numpy as np
import pandas as pd
from src.engine.regime_detector import RegimeDetector

def test_regime_detector():
    detector = RegimeDetector(n_states=4)
    
    # Synthesize arbitrary df of random walk
    dates = pd.date_range("2024-01-01", periods=200)
    df = pd.DataFrame({
        "close": np.linspace(100, 150, 200) + np.random.normal(0, 1, 200),
        "vix_z": np.random.normal(0, 1, 200)
    }, index=dates)
    
    # Compute dummy returns and volatility randomly
    df["Daily_Return"] = np.random.normal(0, 0.01, 200)
    df["Volatility_20"] = np.random.uniform(0.01, 0.05, 200)
    df["mom_12m_Z"] = np.random.normal(0, 1, 200)
    df["MACD_Z"] = np.random.normal(0, 1, 200)
    df["RSI_14_Z"] = np.random.normal(0, 1, 200)
    
    detector.fit(df)
    
    # Should flag _is_fitted
    assert detector._is_fitted
    
    # Prediction should output (N_Days) array spanning [0, 3]
    preds = detector.predict(df)
    assert len(preds) == len(df)
    assert set(preds).issubset({0, 1, 2, 3})
