"""
SQLite Audit Logger
===================
Permanently records all agent state vectors, circuit breaker statuses, and 
execution outputs for retrospective transparency and regulatory audit trails.
"""
import sqlite3
import json
import logging
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class SQLiteAuditLogger:
    def __init__(self, db_path: str = "aegisquant_live.db"):
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        ticker_universe TEXT,
                        regime_id INTEGER,
                        state_vector JSON,
                        rl_output JSON,
                        circuit_breaker_status TEXT,
                        final_weights JSON,
                        transaction_costs REAL,
                        model_version TEXT
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite log: {e}")

    def log_decision(
        self,
        timestamp: str,
        ticker_universe: List[str],
        regime_id: int,
        state_vector: np.ndarray,
        rl_output: np.ndarray,
        circuit_breaker_status: str,
        final_weights: np.ndarray,
        transaction_costs: float,
        model_version: str
    ):
        """Appends a highly structured execution row into the audit table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO decisions (
                        timestamp, ticker_universe, regime_id, state_vector, 
                        rl_output, circuit_breaker_status, final_weights, 
                        transaction_costs, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    json.dumps(ticker_universe),
                    regime_id,
                    json.dumps(state_vector.tolist()),
                    json.dumps(rl_output.tolist()),
                    circuit_breaker_status,
                    json.dumps(final_weights.tolist()),
                    transaction_costs,
                    model_version
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to insert decision row: {e}")
            
    def query_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent N execution lines for the UI."""
        rows = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?', (limit,))
                for r in cursor.fetchall():
                    rows.append(dict(r))
        except sqlite3.Error as e:
            logger.error(f"Failed to query database: {e}")
        return rows
