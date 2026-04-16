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

db_manager = DatabaseSessionManager()
