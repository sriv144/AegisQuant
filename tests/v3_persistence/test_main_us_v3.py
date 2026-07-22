from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, inspect

import main_us_v3
from src.execution.v3 import (
    InMemoryLedger,
    AccountSnapshot,
    OpenOrderSnapshot,
    OrderSide,
    OrderState,
    PositionSnapshot,
    QuoteSnapshot,
    RunPurpose,
    RunRecord,
    RunResult,
    RunStatus,
    RuntimeSettings,
    TradingMode,
)
from src.v3.artifacts import REQUIRED_ARTIFACTS


def _clean_runtime_environment(monkeypatch, run_id: str) -> None:
    for name in (
        "DATABASE_URL",
        "POSTGRES_URL",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "AEGISQUANT_ACCOUNT_KEY",
        "PAPER_EXECUTION_ENABLED",
        "EXECUTION_ENABLED",
        "KILL_SWITCH",
        "RL_ENABLED",
        "TRADING_MODE",
        "RUN_PURPOSE",
        "GITHUB_EVENT_NAME",
        "STRATEGY_ID",
        "STRATEGY_VERSION",
        "BENCHMARK_SYMBOL",
        "STRATEGY_CONFIG_PATH",
        "V3_RUNTIME_INPUT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AEGISQUANT_SKIP_DOTENV", "true")
    monkeypatch.setenv("AEGISQUANT_RUN_ID", run_id)


def _artifact_payload(tmp_path: Path, run_id: str, filename: str) -> dict:
    return json.loads((tmp_path / "artifacts" / run_id / filename).read_text(encoding="utf-8"))


def test_default_health_is_shadow_and_emits_complete_bundle(monkeypatch, tmp_path) -> None:
    _clean_runtime_environment(monkeypatch, "health-default")
    monkeypatch.chdir(tmp_path)

    exit_code = main_us_v3.main([])

    assert exit_code == 0
    run_dir = tmp_path / "artifacts" / "health-default"
    assert {path.name for path in run_dir.iterdir()} == set(REQUIRED_ARTIFACTS)
    manifest = _artifact_payload(tmp_path, "health-default", "manifest.json")
    assert manifest["mode"] == "shadow"
    assert manifest["purpose"] == "health"
    assert manifest["exit_code"] == 0


def test_live_mode_is_blocked_and_still_audited(monkeypatch, tmp_path) -> None:
    _clean_runtime_environment(monkeypatch, "reject-live")
    monkeypatch.chdir(tmp_path)

    exit_code = main_us_v3.main(["--mode", "live", "--purpose", "rebalance"])

    assert exit_code == 2
    manifest = _artifact_payload(tmp_path, "reject-live", "manifest.json")
    assert manifest["status"] == "blocked"
    preflight = _artifact_payload(tmp_path, "reject-live", "preflight.json")
    assert "live" in preflight["message"]


def test_shadow_rebalance_without_frozen_input_is_blocked(monkeypatch, tmp_path) -> None:
    _clean_runtime_environment(monkeypatch, "bootstrap-before-missing-input")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'v3.db').as_posix()}")
    monkeypatch.chdir(tmp_path)
    assert main_us_v3.main(["--mode", "shadow", "--purpose", "bootstrap"]) == 0
    monkeypatch.setenv("AEGISQUANT_RUN_ID", "missing-input")

    exit_code = main_us_v3.main(["--mode", "shadow", "--purpose", "rebalance"])

    assert exit_code == 2
    assert "input" in _artifact_payload(tmp_path, "missing-input", "preflight.json")["message"]


def test_bootstrap_runs_alembic_and_never_constructs_gateway(monkeypatch, tmp_path) -> None:
    _clean_runtime_environment(monkeypatch, "bootstrap")
    database = tmp_path / "v3.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.as_posix()}")
    monkeypatch.chdir(tmp_path)

    class ForbiddenGateway:
        def __init__(self, *args, **kwargs):
            raise AssertionError("bootstrap must not construct an Alpaca gateway")

    monkeypatch.setattr(main_us_v3, "AlpacaPyGateway", ForbiddenGateway)
    exit_code = main_us_v3.main(["--mode", "shadow", "--purpose", "bootstrap"])

    assert exit_code == 0
    assert "execution_runs" in inspect(create_engine(f"sqlite:///{database.as_posix()}" )).get_table_names()
    assert _artifact_payload(tmp_path, "bootstrap", "manifest.json")["status"] == "completed"


class _DurableTestLedger(InMemoryLedger):
    paper_durable_truth = True


def _paper_settings(purpose: RunPurpose) -> RuntimeSettings:
    return RuntimeSettings(
        mode=TradingMode.PAPER,
        purpose=purpose,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        strategy_config_sha256="a" * 64,
        commit_sha="b" * 40,
        database_url="postgresql://durable.example/aegis",
        execution_enabled=False,
        kill_switch=True,
        alpaca_api_key="key",
        alpaca_secret_key="secret",
        account_key="paper-account",
    )


def test_precoordinator_paper_failure_is_recorded_durably() -> None:
    ledger = _DurableTestLedger()
    preflight = {}
    now = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)

    result = main_us_v3._record_runtime_failure(
        ledger=ledger,
        settings=_paper_settings(RunPurpose.REBALANCE),
        previous=main_us_v3._synthetic_result(RunStatus.FAILED, 1, "not started"),
        status=RunStatus.BLOCKED,
        exit_code=2,
        message="frozen input failed validation",
        now=now,
        invocation_id="paper-input-failure",
        preflight=preflight,
    )

    assert result.run_id
    assert result.status is RunStatus.BLOCKED
    assert preflight["durable_failure_recorded"] is True
    persisted = ledger.get_run_by_decision_key(
        result.decision_key + "|runtime-attempt|paper-input-failure"
    )
    assert persisted is not None
    assert persisted.status is RunStatus.BLOCKED
    assert persisted.failure_reason == "frozen input failed validation"


def test_postcoordinator_failure_updates_existing_run_status() -> None:
    ledger = _DurableTestLedger()
    now = datetime(2026, 7, 1, 21, 0, tzinfo=UTC)
    record = RunRecord(
        run_id="eod-run",
        decision_key="strategy|3.0.0|paper-account|paper|eod|2026-07-01",
        strategy_id="strategy",
        strategy_version="3.0.0",
        account_key="paper-account",
        mode=TradingMode.PAPER,
        purpose=RunPurpose.EOD,
        target_hash="c" * 64,
        created_at=now,
    )
    ledger.create_run(record)
    previous = RunResult(
        run_id=record.run_id,
        status=RunStatus.SKIPPED_NOT_DUE,
        exit_code=0,
        message="coordinator handoff",
        decision_key=record.decision_key,
        target_hash=record.target_hash,
    )

    result = main_us_v3._record_runtime_failure(
        ledger=ledger,
        settings=_paper_settings(RunPurpose.EOD),
        previous=previous,
        status=RunStatus.FAILED,
        exit_code=1,
        message="EOD persistence failed",
        now=now,
        invocation_id="eod-failure",
        preflight={},
    )

    assert result.run_id == record.run_id
    persisted = ledger.get_run_by_decision_key(record.decision_key)
    assert persisted is not None
    assert persisted.status is RunStatus.FAILED


def test_bootstrap_migration_preview_includes_open_orders_and_never_submits() -> None:
    now = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)
    settings = _paper_settings(RunPurpose.BOOTSTRAP)
    bundle = SimpleNamespace(
        quotes={
            "SPY": QuoteSnapshot(
                symbol="SPY",
                bid_price=Decimal("99"),
                ask_price=Decimal("101"),
                observed_at=now,
                adv_dollars_30d=Decimal("1000000"),
            )
        },
        research_data_sha256="d" * 64,
    )
    research_plan = SimpleNamespace(
        target_weights=(("SPY", 0.69),),
        weight_sha256="e" * 64,
        promotable=True,
    )
    preview = main_us_v3._migration_delta_preview(
        account=AccountSnapshot(
            account_key="paper-account",
            equity=Decimal("1000"),
            cash=Decimal("0"),
            buying_power=Decimal("0"),
            status="active",
            observed_at=now,
        ),
        positions=(PositionSnapshot("SPY", Decimal("10"), Decimal("100")),),
        open_orders=(
            OpenOrderSnapshot(
                broker_order_id="legacy-order",
                client_order_id="manual-order",
                symbol="SPY",
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                filled_quantity=Decimal("0"),
                state=OrderState.ACCEPTED,
                submitted_at=now,
            ),
        ),
        research_plan=research_plan,
        bundle=bundle,
        settings=settings,
    )

    assert preview["broker_post_count"] == 0
    assert preview["implementation_ready"] is False
    assert preview["unattributed_open_order_symbols"] == ["SPY"]
    assert preview["deltas"][0]["side"] == "sell"
    assert preview["deltas"][0]["current_quantity_including_open_orders"] == "11"
