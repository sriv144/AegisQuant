from src.agents.research.strategy_runner_agent import strategy_runner_agent


def test_strategy_runner_returns_standard_research_signals():
    state = {
        "current_asset": "RELIANCE.NS",
        "active_strategies": ["momentum", "mean_reversion"],
        "current_strategy": "momentum",
        "technical_indicators": {
            "RSI_14_Z": -1.2,
            "MACD_Z": 0.5,
            "BB_Position_Z": 1.0,
            "Volatility_20_Z": 0.2,
        },
        "portfolio_state": {"current_drawdown": 0.02, "vix_raw": 18.0},
        "alternative_data": {"sentiment": 0.2, "sentiment_score": 0.2, "news_volume": 3},
    }

    result = strategy_runner_agent.invoke(state)

    assert "research_signals" in result
    assert len(result["research_signals"]) == 2
    for signal in result["research_signals"]:
        assert {"agent_name", "action", "confidence", "rationale"} <= set(signal)
        assert signal["action"] in {"PROPOSE_LONG", "PROPOSE_SHORT", "HOLD"}
