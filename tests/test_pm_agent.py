import numpy as np

from src.agents.portfolio.pm_agent import pm_agent


def test_pm_agent_extracts_14d_historical_env_state():
    state = {
        "technical_indicators": {
            "Volatility_20_Z": 0.4,
            "RSI_14_Z": -0.8,
            "MACD_Z": 0.2,
            "BB_Position_Z": 1.4,
            "mom_12m_Z": 3.5,
        },
        "portfolio_state": {
            "current_drawdown": 0.12,
            "vix_raw": 27.0,
        },
    }

    obs = pm_agent._extract_rl_state(state)

    assert obs.shape == (14,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -2.0)
    assert np.all(obs <= 2.0)
    assert obs[7:11].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert obs[12] == np.float32(0.7)
