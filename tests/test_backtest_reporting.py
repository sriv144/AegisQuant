import json

from src.backtest.reporting import generate_report, summarize_backtest


def _sample_result():
    return {
        "tickers": ["SPY", "QQQ"],
        "aggregate": {
            "annualised_return": -0.2,
            "annualised_volatility": 0.12,
            "sharpe_ratio": -1.4,
            "sortino_ratio": -2.0,
            "max_drawdown": -0.85,
            "calmar_ratio": -0.24,
            "win_rate": 0.4,
            "profit_factor": 0.6,
            "deflated_sharpe_ratio": 0.0,
        },
        "oos_returns_count": 120,
        "feature_importance": {"SPY_rsi": 0.2, "QQQ_macd": 0.1},
        "benchmarks": {
            "buy_hold_spy": {
                "label": "Buy & Hold SPY",
                "annualised_return": 0.1,
                "sharpe_ratio": 0.8,
                "max_drawdown": -0.25,
            },
            "rl_strategy": {
                "label": "AegisQuant RL Strategy",
                "annualised_return": -0.2,
                "sharpe_ratio": -1.4,
                "max_drawdown": -0.85,
            },
        },
        "monte_carlo": {
            "probability_of_ruin": 0.99,
            "sharpe_p5": -2.1,
            "sharpe_p50": -1.4,
            "sharpe_p95": -0.5,
        },
        "windows": [{"window_id": 1}, {"window_id": 2, "error": "failed"}],
    }


def test_summarize_backtest_reports_failed_risk_gate():
    summary = summarize_backtest(_sample_result())

    assert summary["verdict"] == "FAILED_RISK_GATE"
    assert summary["window_count"] == 2
    assert summary["failed_window_count"] == 1
    assert summary["benchmarks"][0]["sharpe_ratio"] == 0.8


def test_generate_report_writes_markdown_and_json(tmp_path):
    input_path = tmp_path / "walk_forward_demo.json"
    input_path.write_text(json.dumps(_sample_result()), encoding="utf-8")

    md_path, json_path = generate_report(input_path, tmp_path)

    assert md_path.exists()
    assert json_path.exists()
    text = md_path.read_text(encoding="utf-8")
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert "AegisQuant Backtest Audit Report" in text
    assert "FAILED_RISK_GATE" in text
    assert summary["verdict"] == "FAILED_RISK_GATE"
