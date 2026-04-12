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

from sqlalchemy import Column, Integer, String, Float, Text, create_engine
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
