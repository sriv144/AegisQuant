"""
Position Manager for NSE Trading
=================================
Tracks open CNC (delivery) positions persistently across daily runs.
Enforces stop-loss, take-profit, and aging exits.
Persists to SQLite OpenPosition table.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, select
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents one open trading position."""
    ticker: str
    entry_price: float
    entry_date: str          # ISO date string
    quantity: int
    trade_type: str          # "MIS" or "CNC"
    strategy: str            # which strategy generated signal
    stop_loss_pct: float     # e.g., 0.08 = -8%
    take_profit_pct: float   # e.g., 0.20 = +20%
    max_hold_days: int       # max days to hold (60-90)
    sector: str = "OTHER"    # stock sector for diversification
    status: str = "OPEN"     # "OPEN" or "CLOSED"
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None  # "SL", "TP", "AGING", "MANUAL"
    pnl_pct: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dict for DB/JSON serialization."""
        return asdict(self)

    def days_held(self, as_of_date: Optional[datetime] = None) -> int:
        """Days held from entry_date to as_of_date (default: today)."""
        as_of = as_of_date or datetime.now()
        entry = datetime.fromisoformat(self.entry_date)
        return (as_of - entry).days

    def check_exit_conditions(self, current_price: float) -> Optional[str]:
        """
        Check if position should be exited.

        Returns:
            Reason string ("SL", "TP", "AGING") if exit triggered, else None
        """
        if self.status != "OPEN":
            return None

        # Check SL
        sl_price = self.entry_price * (1 - self.stop_loss_pct)
        if current_price < sl_price:
            return "SL"

        # Check TP
        tp_price = self.entry_price * (1 + self.take_profit_pct)
        if current_price > tp_price:
            return "TP"

        # Check aging
        if self.days_held() > self.max_hold_days:
            return "AGING"

        return None

    @classmethod
    def default_cnc(cls, ticker: str, entry_price: float, quantity: int, strategy: str, sector: str = "OTHER") -> "Position":
        """Factory: create a default CNC position."""
        return cls(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=datetime.now().isoformat(),
            quantity=quantity,
            trade_type="CNC",
            strategy=strategy,
            stop_loss_pct=0.08,  # -8%
            take_profit_pct=0.20,  # +20%
            max_hold_days=90,
            sector=sector,
        )

    @classmethod
    def default_mis(cls, ticker: str, entry_price: float, quantity: int, strategy: str) -> "Position":
        """Factory: create a default MIS position."""
        return cls(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=datetime.now().isoformat(),
            quantity=quantity,
            trade_type="MIS",
            strategy=strategy,
            stop_loss_pct=0.015,  # -1.5%
            take_profit_pct=0.02,  # +2%
            max_hold_days=1,  # Close same day
            sector="N/A",
        )


class PositionManager:
    """
    Manages open positions: open, close, check exits, persist to DB.
    """

    def __init__(self, db_url: str = "sqlite:///aegisquant_live.db"):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self._positions_cache: Dict[str, Position] = {}
        self._load_from_db()

    def _load_from_db(self):
        """Load all OPEN positions from DB into memory cache."""
        try:
            from src.db.models import OpenPosition as ORMPosition

            session = self.Session()
            rows = session.query(ORMPosition).filter(ORMPosition.status == "OPEN").all()
            for row in rows:
                pos_dict = {
                    "ticker": row.ticker,
                    "entry_price": row.entry_price,
                    "entry_date": row.entry_date,
                    "quantity": row.quantity,
                    "trade_type": row.trade_type,
                    "strategy": row.strategy,
                    "stop_loss_pct": row.stop_loss_pct,
                    "take_profit_pct": row.take_profit_pct,
                    "max_hold_days": row.max_hold_days,
                    "sector": row.sector or "OTHER",
                    "status": row.status,
                    "exit_price": row.exit_price,
                    "exit_date": row.exit_date,
                    "exit_reason": row.exit_reason,
                    "pnl_pct": row.pnl_pct,
                }
                self._positions_cache[row.ticker] = Position(**pos_dict)
                logger.info(f"[PositionManager] Loaded position: {row.ticker} ({row.quantity}x @ {row.entry_price})")
            session.close()
        except Exception as e:
            logger.warning(f"[PositionManager] Failed to load positions from DB: {e}")

    def get_open_positions(self) -> Dict[str, Position]:
        """Return dict of all OPEN positions."""
        return {k: v for k, v in self._positions_cache.items() if v.status == "OPEN"}

    def open_position(self, position: Position) -> None:
        """
        Open a new position.

        Args:
            position: Position object to open
        """
        if position.ticker in self._positions_cache:
            logger.warning(f"[PositionManager] {position.ticker} already open, skipping")
            return

        self._positions_cache[position.ticker] = position
        self._persist_to_db(position)
        logger.info(f"[PositionManager] Opened {position.ticker}: {position.quantity}x @ {position.entry_price} ({position.trade_type})")

    def close_position(self, ticker: str, exit_price: float, reason: str = "MANUAL") -> Optional[Position]:
        """
        Close a position.

        Args:
            ticker: Ticker to close
            exit_price: Exit price
            reason: Reason for exit ("SL", "TP", "AGING", "MANUAL")

        Returns:
            Closed Position object, or None if not found
        """
        if ticker not in self._positions_cache:
            logger.warning(f"[PositionManager] {ticker} not in open positions")
            return None

        position = self._positions_cache[ticker]
        pnl_pct = (exit_price - position.entry_price) / position.entry_price

        # Mark closed
        position.status = "CLOSED"
        position.exit_price = exit_price
        position.exit_date = datetime.now().isoformat()
        position.exit_reason = reason
        position.pnl_pct = pnl_pct

        self._persist_to_db(position)
        logger.info(
            f"[PositionManager] Closed {ticker}: exit @ {exit_price} ({reason}), P&L: {pnl_pct*100:.2f}%"
        )

        return position

    def daily_check(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check all open positions for SL/TP/aging exits.

        Returns:
            List of tickers that should be exited today
        """
        to_exit = []
        for ticker, position in self.get_open_positions().items():
            if ticker not in current_prices:
                continue

            reason = position.check_exit_conditions(current_prices[ticker])
            if reason:
                logger.info(f"[PositionManager] {ticker} triggers exit: {reason}")
                to_exit.append(ticker)

        return to_exit

    def tickers_due_for_re_evaluation(self) -> List[str]:
        """
        Return CNC positions held > 14 days (time to re-screen).

        Returns:
            List of tickers due for re-evaluation
        """
        due = []
        for ticker, position in self.get_open_positions().items():
            if position.trade_type == "CNC" and position.days_held() > 14:
                due.append(ticker)

        return due

    def _persist_to_db(self, position: Position) -> None:
        """Save position to OpenPosition table."""
        try:
            from src.db.models import OpenPosition as ORMPosition

            session = self.Session()
            # Try to find existing row
            existing = session.query(ORMPosition).filter(ORMPosition.ticker == position.ticker).first()

            if existing:
                # Update
                existing.entry_price = position.entry_price
                existing.entry_date = position.entry_date
                existing.quantity = position.quantity
                existing.trade_type = position.trade_type
                existing.strategy = position.strategy
                existing.stop_loss_pct = position.stop_loss_pct
                existing.take_profit_pct = position.take_profit_pct
                existing.max_hold_days = position.max_hold_days
                existing.sector = position.sector
                existing.status = position.status
                existing.exit_price = position.exit_price
                existing.exit_date = position.exit_date
                existing.exit_reason = position.exit_reason
                existing.pnl_pct = position.pnl_pct
            else:
                # Insert
                existing = ORMPosition(
                    ticker=position.ticker,
                    entry_price=position.entry_price,
                    entry_date=position.entry_date,
                    quantity=position.quantity,
                    trade_type=position.trade_type,
                    strategy=position.strategy,
                    stop_loss_pct=position.stop_loss_pct,
                    take_profit_pct=position.take_profit_pct,
                    max_hold_days=position.max_hold_days,
                    sector=position.sector,
                    status=position.status,
                    exit_price=position.exit_price,
                    exit_date=position.exit_date,
                    exit_reason=position.exit_reason,
                    pnl_pct=position.pnl_pct,
                )
                session.add(existing)

            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"[PositionManager] Failed to persist position {position.ticker}: {e}")

    def get_daily_pnl(self, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """
        Compute daily P&L from open + closed positions.

        Returns:
            Dict with "intraday_pnl", "delivery_pnl", "total_pnl"
        """
        try:
            from src.db.models import OpenPosition as ORMPosition

            session = self.Session()
            # Get all closed positions from today
            if as_of_date is None:
                as_of_date = datetime.now().isoformat()

            today = as_of_date.split("T")[0]

            closed_today = session.query(ORMPosition).filter(
                ORMPosition.status == "CLOSED",
                ORMPosition.exit_date.startswith(today),
            ).all()

            intraday_pnl = sum(
                (p.exit_price - p.entry_price) * p.quantity
                for p in closed_today
                if p.trade_type == "MIS" and p.pnl_pct is not None
            )
            delivery_pnl = sum(
                (p.exit_price - p.entry_price) * p.quantity
                for p in closed_today
                if p.trade_type == "CNC" and p.pnl_pct is not None
            )

            session.close()
            return {
                "intraday_pnl": intraday_pnl,
                "delivery_pnl": delivery_pnl,
                "total_pnl": intraday_pnl + delivery_pnl,
            }
        except Exception as e:
            logger.error(f"[PositionManager] Failed to compute daily P&L: {e}")
            return {"intraday_pnl": 0.0, "delivery_pnl": 0.0, "total_pnl": 0.0}


# Module-level singleton
position_manager = PositionManager()
