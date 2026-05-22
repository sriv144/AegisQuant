import numpy as np

from src.db.models import DailyPnL, OpenPosition
from src.engine.paper_portfolio import PaperPortfolio
from src.engine.position_manager import PositionManager


def _portfolio(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'paper.db'}"
    manager = PositionManager(db_url=db_url)
    portfolio = PaperPortfolio(
        base_capital=100_000.0,
        db_url=db_url,
        position_manager=manager,
    )
    return portfolio


def test_simulate_fills_is_idempotent_and_supports_shorts(tmp_path):
    portfolio = _portfolio(tmp_path)
    tickers = ["AAA", "BBB"]
    weights = np.array([0.10, -0.20])
    prices = {"AAA": 100.0, "BBB": 200.0}
    trade_types = {"AAA": "CNC", "BBB": "MIS"}

    opened = portfolio.simulate_fills(tickers, weights, prices, trade_types, equity=100_000.0)
    assert opened == 2

    session = portfolio.Session()
    try:
        rows = {row.ticker: row for row in session.query(OpenPosition).all()}
        assert set(rows) == {"AAA", "BBB"}
        assert rows["AAA"].quantity == 100
        assert rows["BBB"].quantity == -100
        assert rows["AAA"].trade_type == "CNC"
        assert rows["BBB"].trade_type == "MIS"
    finally:
        session.close()

    opened_again = portfolio.simulate_fills(tickers, weights, prices, trade_types, equity=100_000.0)
    assert opened_again == 0

    session = portfolio.Session()
    try:
        assert session.query(OpenPosition).count() == 2
    finally:
        session.close()


def test_mark_to_market_and_daily_pnl_upsert(tmp_path):
    portfolio = _portfolio(tmp_path)
    tickers = ["AAA", "BBB"]
    prices = {"AAA": 100.0, "BBB": 200.0}
    trade_types = {"AAA": "CNC", "BBB": "MIS"}
    weights = np.array([0.10, -0.20])

    portfolio.simulate_fills(tickers, weights, prices, trade_types, equity=100_000.0)

    mtm = portfolio.mark_to_market({"AAA": 110.0, "BBB": 180.0})
    assert mtm == {"AAA": 1000.0, "BBB": 2000.0}

    portfolio.write_daily_pnl(cb_reason="OK", intraday_ratio=0.20)

    session = portfolio.Session()
    try:
        daily_rows = session.query(DailyPnL).all()
        assert len(daily_rows) == 1
        row = daily_rows[0]
        assert row.delivery_pnl == 1000.0
        assert row.intraday_pnl == 2000.0
        assert row.total_pnl == 3000.0
        assert row.total_portfolio_value == 103_000.0

        positions = {pos.ticker: pos for pos in session.query(OpenPosition).all()}
        assert round(positions["AAA"].pnl_pct, 4) == 0.1000
        assert round(positions["BBB"].pnl_pct, 4) == 0.1000
    finally:
        session.close()

    weights_now = portfolio.current_weights(tickers, {"AAA": 110.0, "BBB": 180.0}, 103_000.0)
    assert np.allclose(weights_now, np.array([11_000.0 / 103_000.0, -18_000.0 / 103_000.0]))

    portfolio.mark_to_market({"AAA": 105.0, "BBB": 190.0})
    portfolio.write_daily_pnl(cb_reason="OK", intraday_ratio=0.20)

    session = portfolio.Session()
    try:
        daily_rows = session.query(DailyPnL).all()
        assert len(daily_rows) == 1
        row = daily_rows[0]
        assert row.delivery_pnl == 500.0
        assert row.intraday_pnl == 1000.0
        assert row.total_pnl == 1500.0
        assert row.total_portfolio_value == 101_500.0
    finally:
        session.close()
