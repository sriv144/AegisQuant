import pytest
import numpy as np
from src.backtest.monte_carlo import MonteCarloSimulator

def test_monte_carlo_bootstrap():
    sim = MonteCarloSimulator(n_simulations=1000, horizon=252, ruin_threshold=0.30)
    
    # Mock normal returns, slight positive drift
    np.random.seed(42)
    mock_returns = np.random.normal(0.0005, 0.01, size=500)
    
    result = sim.run(mock_returns)
    
    # Verify presence of core metrics
    assert "sharpe_p50" in result
    assert "probability_of_ruin" in result
    assert "strategy_half_life_days" in result
    
    # Ensure probabilities are bounded [0, 1]
    assert 0.0 <= result["probability_of_ruin"] <= 1.0
    
    # Ensure logical percentiles
    assert result["sharpe_p5"] <= result["sharpe_p50"] <= result["sharpe_p95"]
    assert result["max_dd_p5"] <= result["max_dd_p50"] <= result["max_dd_p95"]

def test_monte_carlo_failure_modes():
    sim = MonteCarloSimulator(n_simulations=10)
    
    # Fails on too few inputs (<10)
    with pytest.raises(ValueError):
        sim.run(np.array([0.01, -0.01]))
