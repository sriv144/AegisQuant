"""
SQLAlchemy ORM Models
=====================
Defines the `decisions` tracking table allowing seamless transition from SQLite to PostgreSQL.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from sqlalchemy import Column, Integer, String, Float, Text, create_engine, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from src import config  # noqa: F401

logger = logging.getLogger(__name__)

Base = declarative_base()

class DecisionRecord(Base):
    __tablename__ = 'decisions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False, default=lambda: datetime.utcnow().isoformat())
    ticker_universe = Column(Text, default="[]")
    regime_id = Column(Integer, default=0)
    state_vector = Column(Text, default="[]")  # Stored as JSON string
    rl_output = Column(Text, default="[]")
    circuit_breaker_status = Column(String, default="OK")
    final_weights = Column(Text, default="[]")
    transaction_costs = Column(Float, default=0.0)
    model_version = Column(String, default="unknown")


class OpenPosition(Base):
    """Track open CNC (delivery) trading positions."""
    __tablename__ = 'open_positions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, unique=True, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_date = Column(String, nullable=False)  # ISO date
    quantity = Column(Integer, nullable=False)
    trade_type = Column(String, nullable=False)  # "MIS" or "CNC"
    strategy = Column(String, nullable=False)  # which strategy generated this
    stop_loss_pct = Column(Float, default=0.08)
    take_profit_pct = Column(Float, default=0.20)
    max_hold_days = Column(Integer, default=90)
    sector = Column(String, default="OTHER")
    status = Column(String, default="OPEN")  # "OPEN" or "CLOSED"
    exit_price = Column(Float, nullable=True)
    exit_date = Column(String, nullable=True)  # ISO date
    exit_reason = Column(String, nullable=True)  # "SL", "TP", "AGING", "MANUAL"
    pnl_pct = Column(Float, nullable=True)
    created_at = Column(String, nullable=False, default=lambda: datetime.utcnow().isoformat())
    updated_at = Column(String, nullable=False, default=lambda: datetime.utcnow().isoformat())


class DailyPnL(Base):
    """Daily P&L summary across intraday and delivery trading."""
    __tablename__ = 'daily_pnl'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, unique=True, nullable=False)  # ISO date
    total_portfolio_value = Column(Float, nullable=False)
    intraday_pnl = Column(Float, default=0.0)
    delivery_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    sharpe_7d = Column(Float, nullable=True)
    intraday_ratio_used = Column(Float, default=0.20)  # What ratio was used that day
    created_at = Column(String, nullable=False, default=lambda: datetime.utcnow().isoformat())


class UniverseSnapshot(Base):
    """Track weekly universe screens."""
    __tablename__ = 'universe_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(String, unique=True, nullable=False)  # ISO date
    tickers = Column(Text, nullable=False)  # JSON list of selected tickers
    ticker_count = Column(Integer, default=0)
    screen_criteria = Column(Text, nullable=True)  # JSON dict of filter criteria used
    created_at = Column(String, nullable=False, default=lambda: datetime.utcnow().isoformat())

class DatabaseSessionManager:
    """Manages connections to standard SQLite or production Postgres db."""
    def __init__(self):
        # Default to SQLite, override with POSTGRES_URL in .env if in docker
        self.db_url = os.getenv("POSTGRES_URL", "sqlite:///aegisquant_live.db")
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def log_decision_orm(self, **kwargs):
        """Alternative ORM hook replacing raw parameterized queries."""
        try:
            # Convert arrays to JSON strings for compatibility across DB types without JSONB
            for k in ["ticker_universe", "state_vector", "rl_output", "final_weights"]:
                if k in kwargs and not isinstance(kwargs[k], str):
                    if hasattr(kwargs[k], "tolist"):
                        kwargs[k] = json.dumps(kwargs[k].tolist())
                    else:
                        kwargs[k] = json.dumps(kwargs[k])
                        
            record = DecisionRecord(**kwargs)
            with self.SessionLocal() as session:
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"SQLAlchemy Insert Failed: {e}")

    def compute_portfolio_value(self, initial_capital: float, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute real portfolio value from DB-tracked positions.

        Returns dict with: portfolio_value, realized_pnl, unrealized_pnl,
        cash_balance, peak_equity, current_drawdown.
        """
        try:
            with self.SessionLocal() as session:
                closed = session.query(OpenPosition).filter(
                    OpenPosition.status == "CLOSED",
                    OpenPosition.pnl_pct.isnot(None),
                ).all()
                realized_pnl = sum(
                    (p.exit_price - p.entry_price) * p.quantity
                    for p in closed
                    if p.exit_price is not None
                )

                open_pos = session.query(OpenPosition).filter(
                    OpenPosition.status == "OPEN"
                ).all()
                invested = sum(p.entry_price * p.quantity for p in open_pos)
                market_value = sum(
                    current_prices.get(p.ticker, p.entry_price) * p.quantity
                    for p in open_pos
                )
                unrealized_pnl = market_value - invested

                cash_balance = initial_capital + realized_pnl - invested
                portfolio_value = cash_balance + market_value

                rows = session.query(DailyPnL.total_portfolio_value).order_by(
                    DailyPnL.date.desc()
                ).limit(500).all()
                historical_peak = max((r[0] for r in rows), default=initial_capital)
                peak_equity = max(portfolio_value, historical_peak, initial_capital)

                drawdown = max(0.0, (peak_equity - portfolio_value) / peak_equity) if peak_equity > 0 else 0.0

                return {
                    "portfolio_value": round(portfolio_value, 2),
                    "realized_pnl": round(realized_pnl, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "cash_balance": round(cash_balance, 2),
                    "peak_equity": round(peak_equity, 2),
                    "current_drawdown": round(drawdown, 6),
                    "open_position_count": len(open_pos),
                    "closed_trade_count": len(closed),
                }
        except Exception as e:
            logger.error(f"compute_portfolio_value failed: {e}")
            return {
                "portfolio_value": initial_capital,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "cash_balance": initial_capital,
                "peak_equity": initial_capital,
                "current_drawdown": 0.0,
                "open_position_count": 0,
                "closed_trade_count": 0,
            }


db_manager = DatabaseSessionManager()
