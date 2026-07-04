from datetime import datetime, timedelta

from src.db.models import BenchmarkDaily, DailyPnL, DataQualitySnapshot, DecisionCycle
from src.engine.audit import AuditLogger
from src.engine.benchmark_tracker import BenchmarkTracker
from src.engine.meta_allocator import MetaAllocator
from src.engine.performance_attribution import PerformanceAttribution


def _db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'spy_alpha.db'}"


def test_spy_benchmark_updates_us_performance_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET", "US")
    monkeypatch.setenv("BENCHMARK_SYMBOL", "SPY")
    db_url = _db_url(tmp_path)
    tracker = BenchmarkTracker(db_url=db_url, symbols=["SPY"])
    perf = PerformanceAttribution(db_url=db_url, base_capital=100_000.0)
    start = datetime(2026, 1, 1)

    session = perf.Session()
    try:
        for i in range(30):
            date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            session.add(DailyPnL(date=date, total_portfolio_value=100_000.0 * (1.003 ** i), total_pnl=0.0))
            session.add(DataQualitySnapshot(date=date, run_timestamp=date, score=1.0, status="OK"))
            session.commit()
            tracker.update_daily([], {"SPY": 100.0 * (1.001 ** i)}, as_of_date=date)
    finally:
        session.close()

    summary = perf.update_daily()

    assert summary["benchmark_symbol"] == "SPY"
    assert summary["verdict"] == "BEATING_SPY"
    assert summary["rolling_excess_5d"] > 0
    assert summary["readiness_status"] == "LIVE_READY"


def test_stale_performance_summary_blocks_today(tmp_path):
    db_url = _db_url(tmp_path)
    perf = PerformanceAttribution(db_url=db_url, benchmark_symbol="SPY")
    session = perf.Session()
    try:
        session.add(DailyPnL(date="2026-01-01", total_portfolio_value=100_000.0))
        session.add(BenchmarkDaily(date="2026-01-01", symbol="SPY", close=100.0))
        session.commit()
    finally:
        session.close()

    perf.update_daily()
    summary = perf.latest_summary(as_of_date="2026-01-02")

    assert summary["is_stale"] is True
    assert summary["readiness_status"] == "BLOCKED"
    assert "stale" in " ".join(summary["reasons"]).lower()


def test_weekly_loss_stop_blocks_promotion(tmp_path, monkeypatch):
    monkeypatch.setenv("WEEKLY_LOSS_STOP", "0.02")
    db_url = _db_url(tmp_path)
    tracker = BenchmarkTracker(db_url=db_url, symbols=["SPY"])
    perf = PerformanceAttribution(db_url=db_url, base_capital=100_000.0, benchmark_symbol="SPY")
    start = datetime(2026, 1, 1)

    session = perf.Session()
    try:
        for i in range(30):
            date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            value = 100_000.0 if i < 25 else 97_000.0
            session.add(DailyPnL(date=date, total_portfolio_value=value, total_pnl=value - 100_000.0))
            session.add(DataQualitySnapshot(date=date, run_timestamp=date, score=1.0, status="OK"))
            session.commit()
            tracker.update_daily([], {"SPY": 100.0}, as_of_date=date)
    finally:
        session.close()

    summary = perf.update_daily()

    assert summary["readiness_status"] == "BLOCKED"
    assert "weekly loss stop" in " ".join(summary["reasons"]).lower()


def test_decision_cycle_records_no_trade(tmp_path):
    audit = AuditLogger(db_url=_db_url(tmp_path))

    row = audit.record_decision_cycle(
        run_id="cycle-1",
        action="NO_TRADE",
        benchmark_symbol="SPY",
        planned_orders=[],
        fills=[],
        rejected_orders=[],
        notes="threshold not met",
    )

    assert row["action"] == "NO_TRADE"
    session = audit.Session()
    try:
        saved = session.query(DecisionCycle).filter(DecisionCycle.run_id == "cycle-1").one()
        assert saved.planned_order_count == 0
        assert saved.notes == "threshold not met"
    finally:
        session.close()


def test_us_v2_loop_uses_thirty_minute_cadence():
    import main_us_v2

    assert main_us_v2._loop_cron_minute() == "5,35"


def test_meta_allocator_caps_sleeves_and_keeps_cash():
    allocator = MetaAllocator(max_total_invested=0.65, max_sleeve_nav=0.325)

    weights = allocator.allocate({"xs_momentum": 0.04, "value_quality_momentum": 0.02})

    assert weights["xs_momentum"] == 0.325
    assert weights["value_quality_momentum"] == 0.325
    assert weights["cash"] == 0.35
