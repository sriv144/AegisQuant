"""
Benchmark tracker for the India paper-trading truth layer.

This module is intentionally boring: every run records standard benchmark rows
so the rest of the system can answer the only question that matters early on:
are we beating NIFTYBEES after costs and drawdown?
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, BenchmarkDaily, DailyPnL

IST = ZoneInfo("Asia/Kolkata")


class BenchmarkTracker:
    """Fetch and persist daily benchmark close/return rows."""

    DEFAULT_SYMBOLS = ["NIFTYBEES.NS", "^NSEI", "BANKBEES.NS", "CASH", "EQUAL_WEIGHT"]

    def __init__(self, db_url: Optional[str] = None, symbols: Optional[List[str]] = None):
        self.db_url = db_url or os.getenv("POSTGRES_URL") or "sqlite:///aegisquant_live.db"
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def update_daily(
        self,
        universe: Optional[Iterable[str]] = None,
        prices: Optional[Dict[str, float]] = None,
        as_of_date: Optional[str] = None,
        portfolio_value: Optional[float] = None,
        date: Optional[str] = None,
    ) -> List[dict] | dict:
        """
        Upsert benchmark rows for today and return serialisable rows.

        Backwards compatible modes:
          - update_daily(universe, prices): writes configured benchmark symbols.
          - update_daily(portfolio_value=..., date=...): writes NIFTYBEES and
            recomputes performance_daily for the supplied portfolio value.
        """
        prices = prices or {}
        date = date or as_of_date or datetime.now(IST).strftime("%Y-%m-%d")
        universe = list(universe or [])
        rows = []

        for symbol in self.symbols:
            if symbol == "CASH":
                rows.append(self._upsert(date, symbol, 1.0, "cash", "OK", forced_return=0.0))
                continue
            if symbol == "EQUAL_WEIGHT":
                close, daily_return, status = self._equal_weight_proxy(universe)
                rows.append(
                    self._upsert(
                        date,
                        symbol,
                        close,
                        "yfinance_batch",
                        status,
                        forced_return=daily_return,
                    )
                )
                continue

            close = float(prices.get(symbol) or 0.0)
            source = "run_prices" if close > 0 else "yfinance"
            status = "OK"
            if close <= 0:
                close, status = self._fetch_latest_close(symbol)
            rows.append(self._upsert(date, symbol, close, source, status))

        if portfolio_value is not None:
            self._upsert_daily_pnl(date, float(portfolio_value))
            try:
                from src.engine.performance_attribution import PerformanceAttribution

                summary = PerformanceAttribution(
                    db_url=self.db_url,
                    benchmark_symbol="NIFTYBEES.NS",
                ).update_daily(date=date)
            except Exception as exc:
                summary = {"error": str(exc)}
            return {"benchmark": rows, "performance": summary}

        return rows

    def latest(self, symbol: str = "NIFTYBEES.NS") -> Optional[dict]:
        session = self.Session()
        try:
            row = (
                session.query(BenchmarkDaily)
                .filter(BenchmarkDaily.symbol == symbol)
                .order_by(BenchmarkDaily.date.desc(), BenchmarkDaily.id.desc())
                .first()
            )
            return self._to_dict(row) if row else None
        finally:
            session.close()

    def get_latest(self) -> dict:
        """Return latest NIFTYBEES benchmark row plus performance summary."""
        out = {"benchmark": self.latest("NIFTYBEES.NS") or {}}
        try:
            from src.engine.performance_attribution import PerformanceAttribution

            out["performance"] = PerformanceAttribution(db_url=self.db_url).latest_summary()
        except Exception as exc:
            out["performance"] = {"error": str(exc)}
        return out

    def get_history(self, days: int = 90) -> list[dict]:
        """Return recent NIFTYBEES benchmark rows."""
        session = self.Session()
        try:
            rows = (
                session.query(BenchmarkDaily)
                .filter(BenchmarkDaily.symbol == "NIFTYBEES.NS")
                .order_by(BenchmarkDaily.date.desc(), BenchmarkDaily.id.desc())
                .limit(int(days))
                .all()
            )
            return [self._to_dict(row) for row in reversed(rows)]
        finally:
            session.close()

    def _upsert(
        self,
        date: str,
        symbol: str,
        close: float,
        source: str,
        status: str,
        forced_return: Optional[float] = None,
    ) -> dict:
        close = float(close or 0.0)
        session = self.Session()
        try:
            previous = (
                session.query(BenchmarkDaily)
                .filter(BenchmarkDaily.symbol == symbol, BenchmarkDaily.date < date)
                .order_by(BenchmarkDaily.date.desc(), BenchmarkDaily.id.desc())
                .first()
            )
            if forced_return is not None:
                daily_return = float(forced_return)
            elif previous and previous.close:
                daily_return = (close / previous.close) - 1.0 if close > 0 else 0.0
            else:
                daily_return = 0.0

            if previous:
                cumulative_return = (1.0 + float(previous.cumulative_return or 0.0)) * (1.0 + daily_return) - 1.0
            else:
                cumulative_return = 0.0

            row = (
                session.query(BenchmarkDaily)
                .filter(BenchmarkDaily.date == date, BenchmarkDaily.symbol == symbol)
                .first()
            )
            now = datetime.utcnow().isoformat()
            if row:
                row.close = close
                row.daily_return = daily_return
                row.cumulative_return = cumulative_return
                row.source = source
                row.fetch_status = status
                row.updated_at = now
            else:
                row = BenchmarkDaily(
                    date=date,
                    symbol=symbol,
                    close=close,
                    daily_return=daily_return,
                    cumulative_return=cumulative_return,
                    source=source,
                    fetch_status=status,
                    updated_at=now,
                )
                session.add(row)
            session.commit()
            return self._to_dict(row)
        except Exception as exc:
            session.rollback()
            return {
                "date": date,
                "symbol": symbol,
                "close": close,
                "daily_return": 0.0,
                "cumulative_return": 0.0,
                "source": source,
                "fetch_status": f"ERROR: {exc}",
            }
        finally:
            session.close()

    def _fetch_latest_close(self, symbol: str) -> tuple[float, str]:
        try:
            import yfinance as yf

            data = yf.download(symbol, period="5d", auto_adjust=True, progress=False)
            if data is None or data.empty:
                return 0.0, "EMPTY"
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return float(close.dropna().iloc[-1]), "OK"
        except Exception as exc:
            return 0.0, f"ERROR: {exc}"

    def _upsert_daily_pnl(self, date: str, portfolio_value: float) -> None:
        session = self.Session()
        try:
            prior_peak = session.query(DailyPnL).order_by(DailyPnL.total_portfolio_value.desc()).first()
            peak = max(portfolio_value, float(prior_peak.total_portfolio_value) if prior_peak else portfolio_value)
            drawdown = max(0.0, (peak - portfolio_value) / peak) if peak else 0.0
            row = session.query(DailyPnL).filter(DailyPnL.date == date).first()
            payload = {
                "total_portfolio_value": portfolio_value,
                "total_pnl": portfolio_value - 250_000.0,
                "drawdown": drawdown,
            }
            if row:
                for key, value in payload.items():
                    setattr(row, key, value)
            else:
                row = DailyPnL(date=date, **payload)
                session.add(row)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _equal_weight_proxy(self, universe: List[str]) -> tuple[float, float, str]:
        tickers = [t for t in universe if t and t != "CASH"][:50]
        if not tickers:
            return self._synthetic_equal_weight_close(0.0), 0.0, "NO_UNIVERSE"

        try:
            import yfinance as yf

            raw = yf.download(tickers, period="7d", auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                return self._synthetic_equal_weight_close(0.0), 0.0, "EMPTY"

            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            returns = close.ffill().pct_change().dropna(how="all")
            if returns.empty:
                daily_return = 0.0
            else:
                daily_return = float(returns.iloc[-1].dropna().mean())
                if not np.isfinite(daily_return):
                    daily_return = 0.0
            return self._synthetic_equal_weight_close(daily_return), daily_return, "OK"
        except Exception as exc:
            return self._synthetic_equal_weight_close(0.0), 0.0, f"ERROR: {exc}"

    def _synthetic_equal_weight_close(self, daily_return: float) -> float:
        prior = self.latest("EQUAL_WEIGHT")
        base = float(prior["close"]) if prior else 100.0
        return base * (1.0 + float(daily_return or 0.0))

    @staticmethod
    def _to_dict(row: BenchmarkDaily) -> dict:
        return {
            "date": row.date,
            "symbol": row.symbol,
            "close": float(row.close or 0.0),
            "daily_return": float(row.daily_return or 0.0),
            "cumulative_return": float(row.cumulative_return or 0.0),
            "source": row.source,
            "fetch_status": row.fetch_status,
        }


benchmark_tracker = BenchmarkTracker()
