from src.db.models import OpenPosition
from src.engine.portfolio_state import PortfolioState


def test_portfolio_state_marks_open_positions_and_drawdown(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'portfolio_state.db'}"
    manager = PortfolioState(db_url=db_url, initial_capital=100_000.0)

    session = manager.Session()
    try:
        for ticker, price, qty in [
            ("AAA.NS", 100.0, 100),
            ("BBB.NS", 200.0, 100),
            ("CCC.NS", 50.0, 200),
        ]:
            session.add(
                OpenPosition(
                    ticker=ticker,
                    entry_price=price,
                    entry_date="2026-01-01",
                    quantity=qty,
                    trade_type="CNC",
                    strategy="test",
                    status="OPEN",
                )
            )
        session.commit()
    finally:
        session.close()

    gain_prices = {"AAA.NS": 105.0, "BBB.NS": 210.0, "CCC.NS": 52.5}
    gain_state = manager.get_portfolio_state(gain_prices)
    assert gain_state["portfolio_value"] > 100_000.0
    assert gain_state["current_drawdown"] == 0.0

    loss_prices = {"AAA.NS": 90.0, "BBB.NS": 180.0, "CCC.NS": 45.0}
    loss_state = manager.get_portfolio_state(loss_prices)
    assert loss_state["current_drawdown"] > 0.0
    assert loss_state["portfolio_value"] < 100_000.0
