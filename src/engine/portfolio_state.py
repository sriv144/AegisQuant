"""
Persistent portfolio state across trading cycles.
Reads/writes to the daily_pnl and open_positions tables.
Computes real portfolio value from: cash + sum(position_value for all open positions).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, DailyPnL, OpenPosition

IST = ZoneInfo("Asia/Kolkata")


class PortfolioState:
    """DB-backed portfolio state and drawdown calculator."""

    def __init__(self, db_url: Optional[str] = None, initial_capital: Optional[float] = None):
        self.db_url = db_url or os.getenv("POSTGRES_URL") or "sqlite:///aegisquant_live.db"
        self.initial_capital = float(
            initial_capital
            if initial_capital is not None
            else os.getenv("AEGIS_INITIAL_CAPITAL", "250000.0")
        )
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._peak_equity = self._load_peak_equity()

    def get_portfolio_state(self, current_prices: Dict[str, float], vix_raw: float = 20.0) -> dict:
        """Return the live portfolio state dict consumed by agents."""
        session = self.Session()
        try:
            positions = session.query(OpenPosition).filter(OpenPosition.status == "OPEN").all()
            latest = session.query(DailyPnL).order_by(DailyPnL.date.desc(), DailyPnL.id.desc()).first()

            entry_value = sum(abs(float(p.quantity or 0)) * float(p.entry_price or 0.0) for p in positions)
            current_values = [
                float(p.quantity or 0) * float(current_prices.get(p.ticker) or p.entry_price or 0.0)
                for p in positions
            ]
            gross_position_value = sum(abs(value) for value in current_values)

            if latest:
                cash_remaining = float(latest.total_portfolio_value or self.initial_capital) - gross_position_value
            else:
                cash_remaining = self.initial_capital - entry_value

            portfolio_value = cash_remaining + sum(current_values)
            self._peak_equity = max(self._peak_equity, portfolio_value)
            drawdown = (
                max(0.0, (self._peak_equity - portfolio_value) / self._peak_equity)
                if self._peak_equity > 0
                else 0.0
            )
            weights = [
                (value / portfolio_value if portfolio_value else 0.0)
                for value in current_values
            ]
            return {
                "portfolio_value": float(portfolio_value),
                "cash_remaining": float(cash_remaining),
                "current_drawdown": float(drawdown),
                "peak_equity": float(self._peak_equity),
                "current_weights": [float(w) for w in weights],
                "vix_raw": float(vix_raw),
            }
        finally:
            session.close()

    def update_after_fills(self, fills: Dict[str, Any], prices: Dict[str, float]) -> dict:
        """
        Persist a fresh DailyPnL row after fills.

        `fills` may be either ticker -> fill_price (current PaperPortfolio shape) or
        ticker -> {quantity, price, side}. Quantity-aware cash updates are applied
        when present; otherwise current open positions drive the MTM calculation.
        """
        state = self.get_portfolio_state(prices)
        cash = float(state["cash_remaining"])

        for ticker, fill in (fills or {}).items():
            if isinstance(fill, dict):
                quantity = abs(int(fill.get("quantity", 0) or 0))
                price = float(fill.get("price") or prices.get(ticker) or 0.0)
                side = str(fill.get("side", "BUY")).upper()
                notional = quantity * price
                cash = cash + notional if side == "SELL" else cash - notional

        portfolio_value = cash + self._open_position_value(prices)
        self._peak_equity = max(self._peak_equity, portfolio_value)
        drawdown = (
            max(0.0, (self._peak_equity - portfolio_value) / self._peak_equity)
            if self._peak_equity > 0
            else 0.0
        )

        today = datetime.now(IST).strftime("%Y-%m-%d")
        session = self.Session()
        try:
            row = session.query(DailyPnL).filter(DailyPnL.date == today).first()
            payload = {
                "total_portfolio_value": float(portfolio_value),
                "total_pnl": float(portfolio_value - self.initial_capital),
                "drawdown": float(drawdown),
            }
            if row:
                for key, value in payload.items():
                    setattr(row, key, value)
            else:
                row = DailyPnL(date=today, **payload)
                session.add(row)
            session.commit()
            return self.get_portfolio_state(prices)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _open_position_value(self, prices: Dict[str, float]) -> float:
        session = self.Session()
        try:
            return float(
                sum(
                    float(p.quantity or 0) * float(prices.get(p.ticker) or p.entry_price or 0.0)
                    for p in session.query(OpenPosition).filter(OpenPosition.status == "OPEN").all()
                )
            )
        finally:
            session.close()

    def _load_peak_equity(self) -> float:
        session = self.Session()
        try:
            peak = session.query(DailyPnL).order_by(DailyPnL.total_portfolio_value.desc()).first()
            return float(peak.total_portfolio_value) if peak else self.initial_capital
        finally:
            session.close()
