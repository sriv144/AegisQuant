from __future__ import annotations

import importlib
from datetime import UTC, date, datetime
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.v3_models import (
    BenchmarkMark,
    ExecutionRun,
    OrderIntentRecord,
    PortfolioSnapshotRecord,
    V3Base,
)


def test_v3_dashboard_reads_durable_status_and_performance(monkeypatch, tmp_path) -> None:
    database = tmp_path / "dashboard.db"
    database_url = f"sqlite:///{database.as_posix()}"
    engine = create_engine(database_url)
    V3Base.metadata.create_all(engine)
    now = datetime(2026, 7, 1, 20, 45, tzinfo=UTC)
    with Session(engine) as session:
        session.add(
            ExecutionRun(
                run_id="run-dashboard",
                strategy_id="spy_xsmom_core_satellite",
                strategy_version="3.0.0",
                account_key="hashed-account",
                mode="shadow",
                purpose="eod",
                decision_key="ops|eod|2026-07-01",
                trigger="test",
                commit_sha="c" * 40,
                target_hash="d" * 64,
                status="blocked",
                failure_reason="benchmark mark missing",
                metadata_json={},
                started_at=now,
                completed_at=now,
            )
        )
        session.add(
            OrderIntentRecord(
                client_order_id="aq3-s-202607-abcdefghijklmnopqrst",
                run_id="run-dashboard",
                decision_key="ops|eod|2026-07-01",
                sleeve="core",
                symbol="SPY",
                side="buy",
                requested_quantity=None,
                requested_notional=Decimal("1000"),
                frozen_order_amount="1000",
                target_weight=Decimal("0.69"),
                arrival_bid=Decimal("600"),
                arrival_ask=Decimal("600.10"),
                arrival_quote_at=now,
                created_at=now,
            )
        )
        session.add(
            PortfolioSnapshotRecord(
                snapshot_id="snapshot-dashboard",
                run_id="run-dashboard",
                account_key="hashed-account",
                mode="shadow",
                observed_at=now,
                session_date=date(2026, 7, 1),
                nav=Decimal("101000"),
                cash=Decimal("1000"),
                invested_weight=Decimal("0.9900"),
                peak_nav=Decimal("102000"),
                drawdown=Decimal("0.0098039216"),
                beta=Decimal("1.0"),
                tracking_error=Decimal("0.04"),
                cumulative_return=Decimal("0.01"),
                cumulative_benchmark_return=Decimal("0.008"),
                cumulative_excess_return=Decimal("0.002"),
            )
        )
        session.add(
            BenchmarkMark(
                account_key="hashed-account",
                mode="shadow",
                session_date=date(2026, 7, 1),
                symbol="SPY",
                total_return_level=Decimal("700"),
                daily_total_return=Decimal("0.008"),
                source="fixture",
                source_sha256="e" * 64,
                observed_at=now,
            )
        )
        session.commit()

    monkeypatch.setenv("POSTGRES_URL", database_url)
    monkeypatch.setenv("AEGIS_API_KEY", "dashboard-test-key")
    monkeypatch.setenv("AEGISQUANT_ACCOUNT_KEY", "hashed-account")
    monkeypatch.delenv("AEGIS_PASSWORD", raising=False)
    import src.webapp.server as server

    server = importlib.reload(server)
    client = TestClient(server.app)

    headers = {"Authorization": "Bearer dashboard-test-key"}
    status_response = client.get("/api/v3/execution/status", headers=headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["available"] is True
    assert status_payload["latest_run"]["run_id"] == "run-dashboard"
    assert status_payload["gate_failure"] == "benchmark mark missing"
    assert status_payload["nonterminal_order_count"] == 1

    run_response = client.get("/api/v3/execution/runs/run-dashboard", headers=headers)
    assert run_response.status_code == 200
    assert run_response.json()["order_intents"][0]["symbol"] == "SPY"

    performance = client.get("/api/v3/performance", headers=headers).json()
    assert performance["available"] is True
    assert performance["missing_benchmark_rows"] == 0
    assert performance["series"][0]["cumulative_excess_return"] == 0.002


def test_v3_dashboard_fails_closed_without_auth(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("POSTGRES_URL", f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    monkeypatch.setenv("AEGISQUANT_ACCOUNT_KEY", "hashed-account")
    monkeypatch.delenv("AEGIS_API_KEY", raising=False)
    monkeypatch.delenv("AEGIS_PASSWORD", raising=False)
    import src.webapp.server as server

    server = importlib.reload(server)
    response = TestClient(server.app).get("/api/v3/execution/status")
    assert response.status_code == 503
