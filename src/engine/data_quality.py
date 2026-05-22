"""
Run-level data-quality scoring.

This keeps the system honest before it talks about live capital. Missing or
stale inputs should force paper/cash behavior, not confident trading.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, DataQualitySnapshot

IST = ZoneInfo("Asia/Kolkata")


class DataQualityMonitor:
    """Compute and persist a run-level data-quality snapshot."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("POSTGRES_URL") or "sqlite:///aegisquant_live.db"
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def record_run(
        self,
        universe: Iterable[str],
        prices: Dict[str, float],
        alt_data: Optional[Dict[str, dict]] = None,
        broker_prices: Optional[Dict[str, float]] = None,
        stale_quote_count: int = 0,
    ) -> dict:
        universe = list(universe)
        alt_data = alt_data or {}
        broker_prices = broker_prices or {}

        failed_symbols = [
            ticker for ticker in universe
            if not prices.get(ticker) or float(prices.get(ticker) or 0.0) <= 0.0
        ]
        news_failure_count = sum(
            1 for ticker in universe
            if ticker in alt_data and int(alt_data[ticker].get("news_volume", 0) or 0) == 0
        )
        quote_disagreement_count = self._quote_disagreements(prices, broker_prices)

        n = max(1, len(universe))
        score = 1.0
        score -= min(0.70, len(failed_symbols) / n)
        score -= min(0.20, news_failure_count / n * 0.20)
        score -= min(0.20, stale_quote_count / n)
        score -= min(0.20, quote_disagreement_count / n)
        score = max(0.0, min(1.0, score))

        if len(universe) == 0:
            status = "FAIL"
            score = 0.0
        elif len(failed_symbols) > 0 or stale_quote_count > 0 or quote_disagreement_count > 0:
            status = "WARN" if score >= 0.50 else "FAIL"
        else:
            status = "OK" if score >= 0.80 else "WARN"

        notes = []
        if len(universe) == 0:
            notes.append("Universe screener returned zero tickers.")
        if failed_symbols:
            notes.append(f"Missing quotes: {', '.join(failed_symbols[:8])}")
        if news_failure_count:
            notes.append(f"News unavailable for {news_failure_count} symbols.")
        if quote_disagreement_count:
            notes.append(f"Broker/yfinance disagreement count: {quote_disagreement_count}.")

        row = DataQualitySnapshot(
            date=datetime.now(IST).strftime("%Y-%m-%d"),
            run_timestamp=datetime.utcnow().isoformat(),
            missing_quote_count=len(failed_symbols),
            stale_quote_count=int(stale_quote_count),
            news_failure_count=int(news_failure_count),
            quote_disagreement_count=int(quote_disagreement_count),
            failed_symbols=json.dumps(failed_symbols),
            score=float(score),
            status=status,
            notes="; ".join(notes),
        )

        session = self.Session()
        try:
            session.add(row)
            session.commit()
            return self._to_dict(row)
        finally:
            session.close()

    def latest(self) -> dict:
        session = self.Session()
        try:
            row = session.query(DataQualitySnapshot).order_by(DataQualitySnapshot.run_timestamp.desc()).first()
            return self._to_dict(row) if row else {}
        finally:
            session.close()

    @staticmethod
    def _quote_disagreements(prices: Dict[str, float], broker_prices: Dict[str, float]) -> int:
        count = 0
        for ticker, broker_price in broker_prices.items():
            yf_price = float(prices.get(ticker) or 0.0)
            broker_price = float(broker_price or 0.0)
            if yf_price <= 0 or broker_price <= 0:
                continue
            if abs(yf_price - broker_price) / yf_price > 0.01:
                count += 1
        return count

    @staticmethod
    def _to_dict(row: DataQualitySnapshot) -> dict:
        try:
            failed_symbols = json.loads(row.failed_symbols or "[]")
        except Exception:
            failed_symbols = []
        return {
            "date": row.date,
            "run_timestamp": row.run_timestamp,
            "missing_quote_count": int(row.missing_quote_count or 0),
            "stale_quote_count": int(row.stale_quote_count or 0),
            "news_failure_count": int(row.news_failure_count or 0),
            "quote_disagreement_count": int(row.quote_disagreement_count or 0),
            "failed_symbols": failed_symbols,
            "score": float(row.score or 0.0),
            "status": row.status,
            "notes": row.notes or "",
        }
