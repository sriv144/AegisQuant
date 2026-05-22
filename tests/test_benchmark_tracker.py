from datetime import datetime, timedelta

from src.db.models import BenchmarkDaily, DailyPnL, PerformanceDaily
from src.engine.benchmark_tracker import BenchmarkTracker


def test_benchmark_tracker_updates_benchmark_and_performance_rows(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'benchmark_tracker.db'}"
    tracker = BenchmarkTracker(db_url=db_url, symbols=["NIFTYBEES.NS"])
    start = datetime(2026, 1, 1)
    nifty_prices = {}

    session = tracker.Session()
    try:
        for i in range(10):
            date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            value = 250_000.0 * (1.003 ** i)
            nifty_prices[date] = 100.0 * (1.001 ** i)
            session.add(
                DailyPnL(
                    date=date,
                    total_portfolio_value=value,
                    total_pnl=value - 250_000.0,
                    drawdown=0.0,
                )
            )
        session.commit()
    finally:
        session.close()

    active_date = {"date": None}

    def fake_close(symbol):
        return nifty_prices[active_date["date"]], "OK"

    monkeypatch.setattr(tracker, "_fetch_latest_close", fake_close)

    for i in range(10):
        active_date["date"] = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        tracker.update_daily(portfolio_value=250_000.0 * (1.003 ** i), date=active_date["date"])

    session = tracker.Session()
    try:
        assert session.query(BenchmarkDaily).count() == 10
        assert session.query(PerformanceDaily).count() == 10

        rows = session.query(PerformanceDaily).order_by(PerformanceDaily.date.asc()).all()
        for row in rows[1:]:
            assert round(row.excess_return, 6) == round(row.aegis_return - row.benchmark_return, 6)
            assert row.verdict == "INSUFFICIENT_DATA"
    finally:
        session.close()
