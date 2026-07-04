"""
Performance attribution and readiness gates.

This is the blunt scoreboard: AegisQuant must beat NIFTYBEES with acceptable
drawdown before live capital should even be considered.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    Base,
    BenchmarkDaily,
    DailyPnL,
    DataQualitySnapshot,
    PerformanceDaily,
)


class PerformanceAttribution:
    """Compute and persist daily AegisQuant-vs-benchmark metrics."""

    def __init__(
        self,
        db_url: Optional[str] = None,
        base_capital: float = 250_000.0,
        benchmark_symbol: Optional[str] = None,
        max_drawdown_limit: float = 0.15,
    ):
        self.db_url = db_url or os.getenv("POSTGRES_URL") or "sqlite:///aegisquant_live.db"
        self.base_capital = float(base_capital)
        self.benchmark_symbol = benchmark_symbol or os.getenv("BENCHMARK_SYMBOL", "NIFTYBEES.NS")
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self._run_migrations()
        self.Session = sessionmaker(bind=self.engine)

    def _run_migrations(self) -> None:
        from sqlalchemy import text as sql_text

        for stmt in [
            "ALTER TABLE performance_daily ADD COLUMN rolling_excess_5d FLOAT DEFAULT 0.0",
            "ALTER TABLE performance_daily ADD COLUMN hit_rate_5d FLOAT DEFAULT 0.0",
        ]:
            try:
                with self.engine.connect() as conn:
                    conn.execute(sql_text(stmt))
                    conn.commit()
            except Exception:
                pass

    def update_daily(self, date: Optional[str] = None) -> dict:
        """Recompute the full scoreboard and upsert today's performance row."""
        session = self.Session()
        try:
            pnl_rows = session.query(DailyPnL).order_by(DailyPnL.date.asc()).all()
            bench_rows = (
                session.query(BenchmarkDaily)
                .filter(BenchmarkDaily.symbol == self.benchmark_symbol)
                .order_by(BenchmarkDaily.date.asc())
                .all()
            )
            if not pnl_rows:
                return self._empty_summary()

            pnl_df = pd.DataFrame(
                [
                    {
                        "date": r.date,
                        "portfolio_value": float(r.total_portfolio_value or 0.0),
                        "drawdown": float(r.drawdown or 0.0),
                    }
                    for r in pnl_rows
                ]
            ).drop_duplicates("date", keep="last")

            bench_df = pd.DataFrame(
                [
                    {
                        "date": r.date,
                        "benchmark_return": float(r.daily_return or 0.0),
                        "benchmark_cumulative": float(r.cumulative_return or 0.0),
                        "benchmark_close": float(r.close or 0.0),
                    }
                    for r in bench_rows
                ]
            ).drop_duplicates("date", keep="last")

            df = pnl_df.merge(bench_df, on="date", how="left").fillna(0.0)
            df["aegis_return"] = df["portfolio_value"].pct_change().fillna(0.0)
            first_value = float(df["portfolio_value"].iloc[0] or self.base_capital)
            df["aegis_cumulative"] = (df["portfolio_value"] / first_value) - 1.0
            df["excess_return"] = df["aegis_return"] - df["benchmark_return"]
            df["hit"] = df["excess_return"] > 0
            df["rolling_aegis_5d"] = df["aegis_return"].rolling(5, min_periods=1).sum()
            df["rolling_excess_5d"] = df["excess_return"].rolling(5, min_periods=1).sum()
            df["hit_rate_5d"] = df["hit"].rolling(5, min_periods=1).mean()

            df["rolling_sharpe_7"] = self._rolling_sharpe(df["aegis_return"], 7)
            df["rolling_sharpe_30"] = self._rolling_sharpe(df["aegis_return"], 30)
            df["benchmark_sharpe_7"] = self._rolling_sharpe(df["benchmark_return"], 7)
            df["benchmark_sharpe_30"] = self._rolling_sharpe(df["benchmark_return"], 30)
            df["hit_rate_30"] = df["hit"].rolling(30, min_periods=1).mean()
            df["benchmark_drawdown"] = self._benchmark_drawdown(df["benchmark_close"])

            target_date = date or str(df["date"].iloc[-1])
            latest = df[df["date"] == target_date].tail(1)
            if latest.empty:
                latest = df.tail(1)
            row_data = latest.iloc[0].to_dict()

            verdict = self._verdict(row_data, len(df))
            readiness_score, readiness_status, reasons = self._readiness(df, row_data, session)
            perf = self._upsert(session, row_data, len(df), verdict, readiness_score, readiness_status, reasons)
            session.commit()
            return perf
        except Exception as exc:
            session.rollback()
            return {**self._empty_summary(), "error": str(exc)}
        finally:
            session.close()

    def latest_summary(self, as_of_date: Optional[str] = None) -> dict:
        session = self.Session()
        try:
            row = session.query(PerformanceDaily).order_by(PerformanceDaily.date.desc()).first()
            summary = self._to_dict(row) if row else self._empty_summary()
            if as_of_date and summary.get("date") and summary["date"] != as_of_date:
                summary["is_stale"] = True
                summary["readiness_status"] = "BLOCKED"
                summary.setdefault("reasons", []).append(
                    f"Latest performance row is stale: {summary['date']} < {as_of_date}."
                )
            else:
                summary["is_stale"] = False
            return summary
        finally:
            session.close()

    def should_block_live_execution(self) -> tuple[bool, str]:
        """Return whether broker execution must be blocked by readiness gates."""
        execution_enabled = os.getenv("ENABLE_BROKER_EXECUTION", "False").lower() == "true"
        if not execution_enabled:
            return False, "Broker execution disabled."
        summary = self.latest_summary()
        if summary.get("readiness_status") == "LIVE_READY":
            return False, "Readiness gate passed."
        return True, "Live execution blocked: " + "; ".join(summary.get("reasons", []))

    def _upsert(self, session, row_data, days_observed, verdict, readiness_score, readiness_status, reasons) -> dict:
        date = str(row_data["date"])
        row = session.query(PerformanceDaily).filter(PerformanceDaily.date == date).first()
        payload = {
            "portfolio_value": float(row_data["portfolio_value"]),
            "benchmark_symbol": self.benchmark_symbol,
            "aegis_return": float(row_data["aegis_return"]),
            "benchmark_return": float(row_data["benchmark_return"]),
            "excess_return": float(row_data["excess_return"]),
            "cumulative_aegis_return": float(row_data["aegis_cumulative"]),
            "cumulative_benchmark_return": float(row_data["benchmark_cumulative"]),
            "rolling_sharpe_7": float(row_data["rolling_sharpe_7"]),
            "rolling_sharpe_30": float(row_data["rolling_sharpe_30"]),
            "benchmark_sharpe_7": float(row_data["benchmark_sharpe_7"]),
            "benchmark_sharpe_30": float(row_data["benchmark_sharpe_30"]),
            "max_drawdown": float(row_data["drawdown"]),
            "benchmark_drawdown": float(row_data["benchmark_drawdown"]),
            "hit_rate_30": float(row_data["hit_rate_30"]),
            "rolling_excess_5d": float(row_data["rolling_excess_5d"]),
            "hit_rate_5d": float(row_data["hit_rate_5d"]),
            "days_observed": int(days_observed),
            "verdict": verdict,
            "readiness_score": float(readiness_score),
            "readiness_status": readiness_status,
            "reasons": json.dumps(reasons),
            "updated_at": datetime.utcnow().isoformat(),
        }
        if row:
            for key, value in payload.items():
                setattr(row, key, value)
        else:
            row = PerformanceDaily(date=date, **payload)
            session.add(row)
        session.flush()
        return self._to_dict(row)

    def _readiness(self, df: pd.DataFrame, latest: dict, session) -> tuple[float, str, list[str]]:
        reasons = []
        score = 0.0
        days = len(df)
        cumulative_excess = float(latest["aegis_cumulative"] - latest["benchmark_cumulative"])

        if days >= 30:
            score += 20
        else:
            reasons.append(f"Need 30 paper days; have {days}.")

        if cumulative_excess > 0:
            score += 20
        else:
            reasons.append(f"Cumulative excess return vs {self.benchmark_symbol} is not positive.")

        sharpe_ok = float(latest["rolling_sharpe_30"]) > float(latest["benchmark_sharpe_30"])
        drawdown_ok = float(latest["drawdown"]) < float(latest["benchmark_drawdown"])
        weekly_excess_ok = float(latest.get("rolling_excess_5d", 0.0) or 0.0) > 0.0
        if sharpe_ok or drawdown_ok or weekly_excess_ok:
            score += 20
        else:
            reasons.append("Risk-adjusted return has not beaten benchmark yet.")

        if float(latest["drawdown"]) <= self.max_drawdown_limit:
            score += 20
        else:
            reasons.append(f"Drawdown exceeds {self.max_drawdown_limit:.0%} limit.")

        weekly_loss_stop = float(os.getenv("WEEKLY_LOSS_STOP", "0.02") or 0.02)
        if float(latest.get("rolling_aegis_5d", 0.0) or 0.0) <= -weekly_loss_stop:
            reasons.append(f"Weekly loss stop breached at {weekly_loss_stop:.0%}.")

        recent_dq = (
            session.query(DataQualitySnapshot)
            .order_by(DataQualitySnapshot.run_timestamp.desc())
            .limit(5)
            .all()
        )
        bad_dq = [row for row in recent_dq if row.status != "OK"]
        if not bad_dq:
            score += 20
        else:
            reasons.append("Recent data-quality snapshot is not OK.")

        status = "LIVE_READY" if score >= 100 and not reasons else "BLOCKED"
        return score, status, reasons

    def _verdict(self, latest: dict, days: int) -> str:
        if days < 30:
            return "INSUFFICIENT_DATA"
        excess = float(latest["aegis_cumulative"] - latest["benchmark_cumulative"])
        suffix = self._verdict_suffix()
        if excess > 0:
            return f"BEATING_{suffix}"
        if excess < -0.01:
            return f"LAGGING_{suffix}"
        return "ROUGHLY_IN_LINE"

    def _verdict_suffix(self) -> str:
        if self.benchmark_symbol == "NIFTYBEES.NS":
            return "NIFTY"
        return self.benchmark_symbol.replace(".", "_").replace("^", "").upper()

    @staticmethod
    def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
        def calc(values):
            arr = np.asarray(values, dtype=float)
            if len(arr) < 2 or np.std(arr, ddof=1) == 0:
                return 0.0
            return float((np.mean(arr) / np.std(arr, ddof=1)) * np.sqrt(252))

        return returns.rolling(window, min_periods=2).apply(calc, raw=True).fillna(0.0)

    @staticmethod
    def _benchmark_drawdown(close: pd.Series) -> pd.Series:
        close = close.replace(0, np.nan).ffill().fillna(1.0)
        peak = close.cummax()
        return ((peak - close) / peak).fillna(0.0)

    @staticmethod
    def _to_dict(row: PerformanceDaily) -> dict:
        if row is None:
            return PerformanceAttribution._empty_summary()
        try:
            reasons = json.loads(row.reasons or "[]")
        except Exception:
            reasons = []
        return {
            "date": row.date,
            "portfolio_value": float(row.portfolio_value or 0.0),
            "benchmark_symbol": row.benchmark_symbol,
            "aegis_return": float(row.aegis_return or 0.0),
            "benchmark_return": float(row.benchmark_return or 0.0),
            "excess_return": float(row.excess_return or 0.0),
            "cumulative_aegis_return": float(row.cumulative_aegis_return or 0.0),
            "cumulative_benchmark_return": float(row.cumulative_benchmark_return or 0.0),
            "rolling_sharpe_7": float(row.rolling_sharpe_7 or 0.0),
            "rolling_sharpe_30": float(row.rolling_sharpe_30 or 0.0),
            "benchmark_sharpe_7": float(row.benchmark_sharpe_7 or 0.0),
            "benchmark_sharpe_30": float(row.benchmark_sharpe_30 or 0.0),
            "max_drawdown": float(row.max_drawdown or 0.0),
            "benchmark_drawdown": float(row.benchmark_drawdown or 0.0),
            "hit_rate_30": float(row.hit_rate_30 or 0.0),
            "rolling_excess_5d": float(row.rolling_excess_5d or 0.0),
            "hit_rate_5d": float(row.hit_rate_5d or 0.0),
            "days_observed": int(row.days_observed or 0),
            "verdict": row.verdict,
            "readiness_score": float(row.readiness_score or 0.0),
            "readiness_status": row.readiness_status,
            "reasons": reasons,
        }

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "readiness_score": 0.0,
            "readiness_status": "BLOCKED",
            "reasons": ["No performance rows yet."],
            "is_stale": False,
        }
