"""
Weekly self-review - fires from GHA every Friday 15:30 IST.

Reads the last 5 trading days from `decisions` + `daily_pnl`, computes a small
set of signals (win-rate, avg drawdown, CB triggers, most-traded strategies),
appends a dated block to `memory/learnings.md`, and posts the same summary
to Slack. The result is read by every subsequent cron fire via
`journal.read_learnings()` seeded into `AgentState.context_memory`.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text

from src import config  # noqa: F401 - loads .env
from src.db.models import Base
from src.engine import journal
from src.engine.alerting import AlertEngine

IST = ZoneInfo("Asia/Kolkata")


def _engine():
    engine = create_engine(os.getenv("POSTGRES_URL") or "sqlite:///aegisquant_live.db")
    Base.metadata.create_all(engine)
    return engine


def _fetch_week(days: int = 7):
    engine = _engine()
    cutoff = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
    with engine.connect() as conn:
        decisions = conn.execute(
            text("SELECT timestamp, circuit_breaker_status, model_version, final_weights "
                 "FROM decisions WHERE substr(timestamp,1,10) >= :c ORDER BY id ASC"),
            {"c": cutoff},
        ).fetchall()
        pnl = conn.execute(
            text("SELECT date, total_portfolio_value, total_pnl, drawdown "
                 "FROM daily_pnl WHERE date >= :c ORDER BY date ASC"),
            {"c": cutoff},
        ).fetchall()
        performance = conn.execute(
            text("SELECT date, aegis_return, benchmark_return, excess_return, "
                 "cumulative_aegis_return, cumulative_benchmark_return, verdict, "
                 "readiness_score, readiness_status "
                 "FROM performance_daily WHERE date >= :c ORDER BY date ASC"),
            {"c": cutoff},
        ).fetchall()
    return (
        [dict(r._mapping) for r in decisions],
        [dict(r._mapping) for r in pnl],
        [dict(r._mapping) for r in performance],
    )


def _summarise(decisions, pnl, performance=None) -> str:
    performance = performance or []
    if not decisions and not pnl and not performance:
        return "_No activity in the last 7 days._"

    lines: list[str] = []

    # P&L trajectory
    if pnl:
        start = pnl[0]["total_portfolio_value"]
        end = pnl[-1]["total_portfolio_value"]
        change = end - start
        pct = (change / start * 100) if start else 0
        max_dd = max((r["drawdown"] or 0) for r in pnl)
        winning_days = sum(1 for r in pnl if (r["total_pnl"] or 0) > 0)
        lines.append(
            f"- Portfolio: Rs {start:,.0f} -> Rs {end:,.0f} "
            f"({'+' if change>=0 else ''}Rs {change:,.0f}, {pct:+.2f}%)"
        )
        lines.append(f"- Max drawdown: {max_dd*100:.2f}%")
        lines.append(f"- Winning days: {winning_days}/{len(pnl)}")

    # Decision cadence + circuit breakers
    if decisions:
        cb_hits = Counter(d["circuit_breaker_status"] for d in decisions)
        cb_blocked = sum(v for k, v in cb_hits.items() if k and k != "OK")
        active = 0
        for d in decisions:
            try:
                fw = json.loads(d["final_weights"] or "[]")
                if any(abs(float(w)) >= 1e-3 for w in fw):
                    active += 1
            except Exception:
                pass
        lines.append(f"- Runs: {len(decisions)} - with active weights: {active}")
        lines.append(
            f"- Circuit breaker: {cb_blocked} blocked / "
            f"{cb_hits.get('OK', 0)} clean"
        )
        models = Counter(d["model_version"] for d in decisions)
        top = ", ".join(f"{m} ({c})" for m, c in models.most_common(3))
        lines.append(f"- Models seen: {top}")

    if performance:
        latest = performance[-1]
        week_excess = sum(float(r.get("excess_return") or 0.0) for r in performance)
        hit_rate = sum(1 for r in performance if float(r.get("excess_return") or 0.0) > 0) / len(performance)
        lines.append("")
        lines.append("**Benchmark truth:**")
        lines.append(
            f"- Latest: Aegis {float(latest.get('aegis_return') or 0.0)*100:+.2f}% vs "
            f"NIFTY {float(latest.get('benchmark_return') or 0.0)*100:+.2f}% "
            f"(excess {float(latest.get('excess_return') or 0.0)*100:+.2f}%)"
        )
        lines.append(f"- Weekly excess sum: {week_excess*100:+.2f}%")
        lines.append(f"- Beat-NIFTY hit rate: {hit_rate*100:.0f}%")
        lines.append(
            f"- Verdict: {latest.get('verdict')} - readiness "
            f"{latest.get('readiness_status')} ({float(latest.get('readiness_score') or 0.0):.0f}/100)"
        )

        if latest.get("verdict") == "LAGGING_NIFTY":
            lines.append("- Blunt read: system lagged benchmark; do not add capital or more complexity.")
        elif latest.get("verdict") == "BEATING_NIFTY":
            lines.append("- Blunt read: paper edge is improving, but live capital still needs readiness gates.")
        else:
            lines.append("- Blunt read: edge is unproven; keep paper mode and collect more days.")

    lines.append("")
    lines.append("**Action items for next week:**")
    lines.append("- Review any persistent CB blocks - tune VIX / drawdown thresholds if miscalibrated.")
    lines.append("- If winning-days ratio < 0.4, revisit committee confidence floor.")
    lines.append("- Confirm universe screener is returning fresh tickers, not cached seed.")

    return "\n".join(lines)


def run_weekly_review() -> None:
    decisions, pnl, performance = _fetch_week(days=7)
    summary = _summarise(decisions, pnl, performance)
    journal.write_weekly_review(summary)
    stamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    header = f"*AegisQuant Weekly Review - {stamp}*\n"
    AlertEngine().notify(header + summary, level="INFO")
    print(
        f"[WeeklyReview] wrote learnings.md - decisions={len(decisions)} "
        f"pnl_rows={len(pnl)} performance_rows={len(performance)}"
    )


if __name__ == "__main__":
    run_weekly_review()
