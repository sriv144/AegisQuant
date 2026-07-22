from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime
from decimal import Decimal

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from src.db.v3_models import V3Base, V3_TABLE_NAMES
from src.db.v3_ledger import SQLAlchemyLedger
from src.execution.v3 import (
    OrderEvent,
    OrderIntent,
    OrderSide,
    OrderState,
    RunPurpose,
    RunRecord,
    RunStatus,
    TradingMode,
)


EXPECTED_TABLES = {
    "strategy_epochs",
    "execution_runs",
    "order_intents",
    "order_events",
    "execution_leases",
    "portfolio_snapshots",
    "position_snapshots",
    "shadow_accounts",
    "shadow_positions",
    "benchmark_marks",
    "data_manifests",
    "experiment_runs",
}


def test_v3_metadata_is_isolated_from_legacy_tables() -> None:
    assert set(V3_TABLE_NAMES) == EXPECTED_TABLES
    assert "decisions" not in V3Base.metadata.tables
    assert "open_positions" not in V3Base.metadata.tables


def test_alembic_upgrade_creates_v3_schema_without_legacy_tables(tmp_path: Path) -> None:
    database = tmp_path / "v3-ledger.db"
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")

    command.upgrade(config, "head")

    tables = set(inspect(create_engine(f"sqlite:///{database.as_posix()}")).get_table_names())
    assert EXPECTED_TABLES <= tables
    assert "alembic_version" in tables
    assert "decisions" not in tables


def test_sqlalchemy_ledger_persists_idempotent_run_intent_and_events() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    ledger = SQLAlchemyLedger(engine=engine)
    now = datetime(2026, 7, 1, 14, 20, tzinfo=UTC)
    record = RunRecord(
        run_id="run-1",
        decision_key="strategy|3.0.0|acct|shadow|2026-07",
        strategy_id="strategy",
        strategy_version="3.0.0",
        account_key="acct",
        mode=TradingMode.SHADOW,
        purpose=RunPurpose.REBALANCE,
        target_hash="a" * 64,
        created_at=now,
        metadata={"trigger": "test", "commit_sha": "b" * 40},
    )

    assert ledger.acquire_lease("acct", TradingMode.SHADOW, "owner")
    assert ledger.create_run(record).run_id == "run-1"
    intent = OrderIntent(
        client_order_id="aq3-s-202607-abcdefghijklmnopqrst",
        run_id="run-1",
        decision_key=record.decision_key,
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.69"),
        arrival_price=Decimal("600"),
        created_at=now,
        notional=Decimal("69000"),
    )
    ledger.add_intents((intent,))
    ledger.add_intents((intent,))
    ledger.append_order_event(
        OrderEvent(
            event_id="event-1",
            client_order_id=intent.client_order_id,
            state=OrderState.ACCEPTED,
            observed_at=now,
            broker_order_id="broker-1",
        )
    )
    ledger.append_order_event(
        OrderEvent(
            event_id="event-2",
            client_order_id=intent.client_order_id,
            state=OrderState.FILLED,
            observed_at=now,
            broker_order_id="broker-1",
            filled_quantity=Decimal("115"),
            filled_average_price=Decimal("600"),
            slippage_bps=Decimal("1.25"),
        )
    )
    ledger.update_run("run-1", RunStatus.COMPLETED, now, metadata={"ending_nav": "100000"})
    ledger.release_lease("acct", TradingMode.SHADOW, "owner")

    restored = ledger.get_run_by_decision_key(record.decision_key)
    assert restored is not None
    assert restored.status is RunStatus.COMPLETED
    assert restored.metadata["ending_nav"] == "100000"
    assert ledger.current_order_state(intent.client_order_id) is OrderState.FILLED
    events = ledger.events_for(intent.client_order_id)
    assert len(events) == 2
    assert events[-1].slippage_bps == Decimal("1.25")
    assert ledger.intents_for_run("run-1") == (intent,)


def test_sql_ledger_reconciles_all_filled_run_that_crashed_before_completion() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    ledger = SQLAlchemyLedger(engine=engine)
    now = datetime(2026, 7, 1, 14, 20, tzinfo=UTC)
    run = RunRecord(
        run_id="crashed-run",
        decision_key="strategy|3.0.0|acct|paper|2026-07",
        strategy_id="strategy",
        strategy_version="3.0.0",
        account_key="acct",
        mode=TradingMode.PAPER,
        purpose=RunPurpose.REBALANCE,
        target_hash="c" * 64,
        created_at=now,
    )
    ledger.create_run(run)
    intent = OrderIntent(
        client_order_id="aq3-p-202607-abcdefghijklmnopqrst",
        run_id=run.run_id,
        decision_key=run.decision_key,
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.69"),
        arrival_price=Decimal("600"),
        created_at=now,
        notional=Decimal("100"),
    )
    ledger.add_intents((intent,))
    ledger.append_order_event(
        OrderEvent(
            event_id="accepted",
            client_order_id=intent.client_order_id,
            state=OrderState.ACCEPTED,
            observed_at=now,
            broker_order_id="broker",
        )
    )
    ledger.append_order_event(
        OrderEvent(
            event_id="filled",
            client_order_id=intent.client_order_id,
            state=OrderState.FILLED,
            observed_at=now,
            broker_order_id="broker",
            filled_quantity=Decimal("0.16666667"),
            filled_average_price=Decimal("600"),
        )
    )

    unresolved = ledger.oldest_run_requiring_reconciliation("acct", TradingMode.PAPER)
    assert unresolved is not None
    assert unresolved.run_id == run.run_id

    ledger.update_run(run.run_id, RunStatus.COMPLETED, now)
    rejected_run = RunRecord(
        run_id="rejected-run",
        decision_key="strategy|3.0.0|acct|paper|2026-08",
        strategy_id="strategy",
        strategy_version="3.0.0",
        account_key="acct",
        mode=TradingMode.PAPER,
        purpose=RunPurpose.REBALANCE,
        target_hash="d" * 64,
        created_at=now,
    )
    ledger.create_run(rejected_run)
    rejected = OrderIntent(
        client_order_id="aq3-p-202608-abcdefghijklmnopqrst",
        run_id=rejected_run.run_id,
        decision_key=rejected_run.decision_key,
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.69"),
        arrival_price=Decimal("600"),
        created_at=now,
        notional=Decimal("100"),
    )
    ledger.add_intents((rejected,))
    ledger.append_order_event(
        OrderEvent(
            event_id="rejected",
            client_order_id=rejected.client_order_id,
            state=OrderState.REJECTED,
            observed_at=now,
            broker_order_id="broker-rejected",
        )
    )
    ledger.update_run(rejected_run.run_id, RunStatus.BLOCKED, now)

    assert ledger.oldest_run_requiring_reconciliation(
        "acct", TradingMode.PAPER
    ) is None
