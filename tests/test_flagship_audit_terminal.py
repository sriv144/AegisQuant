import numpy as np

from src.db.models import (
    AgentReasoning,
    DataQualitySnapshot,
    PaperFill,
    PaperOrder,
    RLModelEvaluation,
)
from src.engine.audit import AuditLogger
from src.engine.data_quality import DataQualityMonitor
from src.engine.paper_portfolio import PaperPortfolio
from src.engine.position_manager import PositionManager


def _db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'flagship.db'}"


def test_agent_reasoning_and_rl_evaluation_persist_run_id(tmp_path):
    db_url = _db_url(tmp_path)
    audit = AuditLogger(db_url=db_url)
    state = {
        "market_data": {"ticker": "RELIANCE.NS", "price": 1000},
        "technical_indicators": {"RSI_14": 45, "MACD": 1.2},
        "alternative_data": {"sentiment_score": 0.4, "news_volume": 2},
        "portfolio_state": {"vix_raw": 18.0},
        "current_strategy": "momentum",
        "research_signals": [
            {
                "agent_name": "Quant_Research_Agent",
                "action": "PROPOSE_LONG",
                "confidence": 0.8,
                "rationale": "Momentum and MACD are aligned.",
            }
        ],
        "committee_decision": {
            "agent_name": "Strategy_Committee",
            "action": "PROPOSE",
            "confidence": 0.7,
            "rationale": "Two signals agree.",
            "direction": "LONG",
        },
        "allocation_request": {
            "agent_name": "Portfolio_Manager",
            "action": "REQUEST_ALLOCATION",
            "confidence": 0.7,
            "rationale": "RL sizes conservatively.",
        },
    }

    count = audit.record_agent_reasoning("run-1", "RELIANCE.NS", state)
    assert count == 3

    summary = audit.record_rl_evaluation(
        run_id="run-1",
        model_version="india_ppo_rl_live",
        target_weights=[0.1, 0.0],
        safe_weights=[0.08, 0.0],
        performance_summary={
            "excess_return": 0.01,
            "benchmark_return": 0.002,
            "max_drawdown": 0.01,
            "rolling_sharpe_30": 1.2,
            "readiness_status": "LIVE_READY",
            "benchmark_symbol": "NIFTYBEES.NS",
        },
        turnover_from=[0.0, 0.0],
    )
    assert summary["promotion_eligible"] is True

    session = audit.Session()
    try:
        rows = session.query(AgentReasoning).filter(AgentReasoning.run_id == "run-1").all()
        assert len(rows) == 3
        assert rows[0].ticker == "RELIANCE.NS"
        rl = session.query(RLModelEvaluation).filter(RLModelEvaluation.run_id == "run-1").one()
        assert rl.reward > 0
        assert rl.promotion_eligible == 1
    finally:
        session.close()


def test_paper_order_lifecycle_records_orders_fills_and_positions(tmp_path):
    db_url = _db_url(tmp_path)
    manager = PositionManager(db_url=db_url)
    portfolio = PaperPortfolio(base_capital=100_000.0, db_url=db_url, position_manager=manager)

    fills = portfolio.execute_target_weights(
        run_id="run-2",
        tickers=["AAA.NS", "BAD.NS"],
        weights=np.array([0.10, 0.10]),
        prices={"AAA.NS": 100.0, "BAD.NS": 0.0},
        trade_types={"AAA.NS": "CNC", "BAD.NS": "CNC"},
        equity=100_000.0,
        strategies={"AAA.NS": "momentum"},
    )

    assert "AAA.NS" in fills
    assert "BAD.NS" not in fills

    session = portfolio.Session()
    try:
        orders = session.query(PaperOrder).order_by(PaperOrder.ticker.asc()).all()
        assert len(orders) == 2
        assert {o.status for o in orders} == {"PLACED", "REJECTED"}
        fill = session.query(PaperFill).one()
        assert fill.ticker == "AAA.NS"
        assert fill.fees > 0
    finally:
        session.close()

    closed = portfolio.auto_close_intraday("run-2-eod", {"AAA.NS": 101.0})
    assert closed == 0


def test_zero_universe_data_quality_blocks_cleanly(tmp_path):
    monitor = DataQualityMonitor(db_url=_db_url(tmp_path))
    summary = monitor.record_run(universe=[], prices={}, alt_data={})

    assert summary["status"] == "FAIL"
    assert summary["score"] == 0.0

    session = monitor.Session()
    try:
        row = session.query(DataQualitySnapshot).one()
        assert row.status == "FAIL"
        assert "zero tickers" in row.notes
    finally:
        session.close()


def test_rl_promotion_gate_rejects_negative_excess(tmp_path):
    audit = AuditLogger(db_url=_db_url(tmp_path))
    summary = audit.record_rl_evaluation(
        run_id="run-3",
        model_version="india_ppo_rl_live",
        target_weights=[0.2],
        safe_weights=[0.2],
        performance_summary={
            "excess_return": -0.02,
            "benchmark_return": 0.01,
            "max_drawdown": 0.03,
            "rolling_sharpe_30": 0.5,
            "readiness_status": "LIVE_READY",
        },
    )

    assert summary["promotion_eligible"] is False
    assert summary["reward"] < 0


def test_terminal_api_returns_audit_surfaces(monkeypatch, tmp_path):
    db_url = _db_url(tmp_path)
    monkeypatch.setenv("POSTGRES_URL", db_url)
    audit = AuditLogger(db_url=db_url)
    audit.record_market_observation(
        run_id="run-api",
        vix=19.5,
        universe=["AAA.NS"],
        data_quality={"status": "OK", "notes": "clean"},
        prices={"AAA.NS": 100.0},
        alt_data={"AAA.NS": {"sentiment_score": 0.5, "news_volume": 3}},
    )
    audit.record_agent_reasoning(
        "run-api",
        "AAA.NS",
        {
            "market_data": {"ticker": "AAA.NS", "price": 100.0},
            "technical_indicators": {"RSI_14": 42},
            "alternative_data": {"sentiment_score": 0.5, "news_volume": 3},
            "portfolio_state": {"vix_raw": 19.5},
            "current_strategy": "momentum",
            "research_signals": [{"agent_name": "Quant", "action": "PROPOSE_LONG", "confidence": 0.6, "rationale": "RSI stable."}],
            "committee_decision": {"direction": "LONG", "action": "PROPOSE", "confidence": 0.6, "rationale": "Aligned."},
        },
    )
    audit.record_rl_evaluation(
        "run-api",
        "india_ppo_rl_live",
        [0.1],
        [0.1],
        {"excess_return": 0.0, "readiness_status": "BLOCKED"},
    )

    from src.webapp.server import get_decision_detail, get_rl_lab, get_watchlist

    watchlist = get_watchlist()
    detail = get_decision_detail("run-api")
    rl = get_rl_lab()

    assert watchlist[0]["ticker"] == "AAA.NS"
    assert "strongest agent view" in watchlist[0]["beginner_reason"]
    assert detail["observations"][0]["data_quality_status"] == "OK"
    assert detail["reasoning"]
    assert detail["beginner_explanation"]["headline"]
    assert detail["beginner_explanation"]["ticker_explanations"]
    assert rl["evaluations"][0]["run_id"] == "run-api"
