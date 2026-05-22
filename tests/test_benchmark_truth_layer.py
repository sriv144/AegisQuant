from datetime import datetime, timedelta

from src.db.models import BenchmarkDaily, DailyPnL, DataQualitySnapshot
from src.engine.benchmark_tracker import BenchmarkTracker
from src.engine.data_quality import DataQualityMonitor
from src.engine.performance_attribution import PerformanceAttribution


def _db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'truth.db'}"


def test_benchmark_tracker_upserts_returns(tmp_path):
    tracker = BenchmarkTracker(db_url=_db_url(tmp_path), symbols=["NIFTYBEES.NS", "CASH"])

    day1 = tracker.update_daily([], {"NIFTYBEES.NS": 100.0}, as_of_date="2026-01-01")
    day2 = tracker.update_daily([], {"NIFTYBEES.NS": 105.0}, as_of_date="2026-01-02")
    tracker.update_daily([], {"NIFTYBEES.NS": 110.0}, as_of_date="2026-01-02")

    nifty_day1 = next(row for row in day1 if row["symbol"] == "NIFTYBEES.NS")
    nifty_day2 = next(row for row in day2 if row["symbol"] == "NIFTYBEES.NS")
    assert nifty_day1["daily_return"] == 0.0
    assert round(nifty_day2["daily_return"], 4) == 0.05

    session = tracker.Session()
    try:
        assert session.query(BenchmarkDaily).filter(BenchmarkDaily.symbol == "NIFTYBEES.NS").count() == 2
    finally:
        session.close()


def test_performance_attribution_and_readiness_gate(tmp_path):
    db_url = _db_url(tmp_path)
    tracker = BenchmarkTracker(db_url=db_url, symbols=["NIFTYBEES.NS"])
    perf = PerformanceAttribution(db_url=db_url, base_capital=100_000.0)

    session = perf.Session()
    try:
        start = datetime(2026, 1, 1)
        for i in range(30):
            date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            session.add(
                DailyPnL(
                    date=date,
                    total_portfolio_value=100_000.0 * (1.002 ** i),
                    total_pnl=100_000.0 * (1.002 ** i) - 100_000.0,
                    drawdown=0.0,
                )
            )
            session.add(
                DataQualitySnapshot(
                    date=date,
                    run_timestamp=(start + timedelta(days=i)).isoformat(),
                    score=1.0,
                    status="OK",
                )
            )
            session.commit()
            bench_price = 100.0 * (1.001 ** i)
            if i >= 20:
                bench_price *= 0.97
            tracker.update_daily([], {"NIFTYBEES.NS": bench_price}, as_of_date=date)
    finally:
        session.close()

    summary = perf.update_daily()
    assert summary["verdict"] == "BEATING_NIFTY"
    assert summary["readiness_status"] == "LIVE_READY"
    assert summary["readiness_score"] == 100.0
    assert summary["cumulative_aegis_return"] > summary["cumulative_benchmark_return"]


def test_data_quality_scoring_blocks_missing_quotes(tmp_path):
    monitor = DataQualityMonitor(db_url=_db_url(tmp_path))
    summary = monitor.record_run(
        universe=["AAA.NS", "BBB.NS"],
        prices={"AAA.NS": 100.0, "BBB.NS": 0.0},
        alt_data={"AAA.NS": {"news_volume": 1}, "BBB.NS": {"news_volume": 0}},
    )

    assert summary["status"] == "FAIL"
    assert summary["missing_quote_count"] == 1
    assert summary["failed_symbols"] == ["BBB.NS"]


def test_performance_api_returns_latest(monkeypatch, tmp_path):
    db_url = _db_url(tmp_path)
    monkeypatch.setenv("POSTGRES_URL", db_url)
    tracker = BenchmarkTracker(db_url=db_url, symbols=["NIFTYBEES.NS"])
    perf = PerformanceAttribution(db_url=db_url, base_capital=100_000.0)

    session = perf.Session()
    try:
        session.add(DailyPnL(date="2026-01-01", total_portfolio_value=100_000.0, total_pnl=0.0))
        session.add(DailyPnL(date="2026-01-02", total_portfolio_value=101_000.0, total_pnl=1000.0))
        session.add(DataQualitySnapshot(date="2026-01-02", run_timestamp="2026-01-02T10:00:00", score=1.0, status="OK"))
        session.commit()
    finally:
        session.close()

    tracker.update_daily([], {"NIFTYBEES.NS": 100.0}, as_of_date="2026-01-01")
    tracker.update_daily([], {"NIFTYBEES.NS": 100.5}, as_of_date="2026-01-02")
    perf.update_daily()

    from src.webapp.server import get_performance

    body = get_performance()
    assert body["latest"]["date"] == "2026-01-02"
    assert round(body["latest"]["excess_return"], 4) == 0.005


def test_weekly_review_includes_benchmark_verdict():
    from weekly_review import _summarise

    text = _summarise(
        decisions=[],
        pnl=[],
        performance=[
            {
                "date": "2026-01-02",
                "aegis_return": 0.01,
                "benchmark_return": 0.004,
                "excess_return": 0.006,
                "verdict": "BEATING_NIFTY",
                "readiness_status": "BLOCKED",
                "readiness_score": 60,
            }
        ],
    )

    assert "Benchmark truth" in text
    assert "BEATING_NIFTY" in text
    assert "paper edge is improving" in text
