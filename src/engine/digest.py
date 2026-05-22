"""
End-of-Day Slack digest.

Fired only on the last cron of the IST trading day (15:05 IST run) from
`main_india.py`. Reuses the existing AlertEngine so webhook/email wiring is
already configured via SLACK_WEBHOOK_URL.
"""

from __future__ import annotations

from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

from src.engine.alerting import AlertEngine
from src.engine.paper_portfolio import PortfolioSnapshot


IST = ZoneInfo("Asia/Kolkata")


def build_eod_digest(
    snapshot: PortfolioSnapshot,
    decisions_today: List[dict],
    cb_reason: str,
    performance: dict | None = None,
) -> str:
    """Format a multi-line Slack message."""
    stamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    pnl_sign = "+" if snapshot.today_pnl >= 0 else ""
    pct = (snapshot.today_pnl / snapshot.base_capital * 100) if snapshot.base_capital else 0

    lines = [
        f"*AegisQuant EOD - {stamp}*",
        f"Portfolio: Rs {snapshot.portfolio_value:,.0f} "
        f"({pnl_sign}Rs {snapshot.today_pnl:,.0f}, {pnl_sign}{pct:.2f}%)",
        f"Open positions: {snapshot.open_count} - Drawdown: {snapshot.drawdown*100:.2f}%",
        f"Runs today: {len(decisions_today)} - Last CB: {cb_reason}",
    ]

    if performance:
        aegis = float(performance.get("aegis_return", 0.0) or 0.0) * 100
        bench = float(performance.get("benchmark_return", 0.0) or 0.0) * 100
        excess = float(performance.get("excess_return", 0.0) or 0.0) * 100
        verdict = performance.get("verdict", "INSUFFICIENT_DATA")
        readiness = performance.get("readiness_status", "BLOCKED")
        score = float(performance.get("readiness_score", 0.0) or 0.0)
        lines.append(
            f"Benchmark: AegisQuant {aegis:+.2f}% vs NIFTY {bench:+.2f}% "
            f"(excess {excess:+.2f}%)"
        )
        lines.append(f"Truth verdict: {verdict} - readiness {readiness} ({score:.0f}/100)")

    if snapshot.winners:
        wins = " - ".join(
            f"{w['ticker']} +Rs {w['pnl_inr']:,.0f}" for w in snapshot.winners
        )
        lines.append(f"Winners: {wins}")
    if snapshot.losers:
        loss = " - ".join(
            f"{l['ticker']} Rs {l['pnl_inr']:,.0f}" for l in snapshot.losers
        )
        lines.append(f"Losers:  {loss}")
    if not snapshot.winners and not snapshot.losers and snapshot.open_count == 0:
        lines.append("_No open positions yet - circuit breaker or low confidence._")

    return "\n".join(lines)


def send_eod_digest(
    snapshot: PortfolioSnapshot,
    decisions_today: List[dict],
    cb_reason: str,
    performance: dict | None = None,
    alert_engine: AlertEngine | None = None,
) -> None:
    ae = alert_engine or AlertEngine()
    msg = build_eod_digest(snapshot, decisions_today, cb_reason, performance)
    ae.notify(msg, level="INFO")
