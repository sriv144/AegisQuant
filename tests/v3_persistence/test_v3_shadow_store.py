from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.db.v3_ledger import SQLAlchemyLedger
from src.db.v3_models import PortfolioSnapshotRecord, V3Base
from src.db.v3_shadow import DurableShadowExecutor, ShadowAccountStore
from src.execution.v3 import (
    ExecutionCoordinator,
    PortfolioPlan,
    QuoteSnapshot,
    RunPurpose,
    RunStatus,
    RuntimeSettings,
    TradingMode,
)


def test_shadow_checkpoint_persists_account_fills_events_and_snapshot_atomically() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    ledger = SQLAlchemyLedger(engine=engine)
    store = ShadowAccountStore(
        engine,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        config_sha256="a" * 64,
    )
    account = store.load_or_create("shadow-account", Decimal("100000"))
    now = datetime(2026, 7, 1, 14, 20, tzinfo=UTC)
    quotes = {
        "SPY": QuoteSnapshot(
            symbol="SPY",
            bid_price=Decimal("99.99"),
            ask_price=Decimal("100.01"),
            observed_at=now,
            adv_dollars_30d=Decimal("1000000000"),
        )
    }
    plan = PortfolioPlan(
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        as_of=now,
        target_weights={"SPY": Decimal("0.99")},
        metadata={"portfolio_beta": "0.99", "tracking_error": "0.01"},
    )
    settings = RuntimeSettings(
        mode=TradingMode.SHADOW,
        purpose=RunPurpose.REBALANCE,
        strategy_config_sha256="a" * 64,
        account_key="shadow-account",
    )
    coordinator = ExecutionCoordinator(
        settings,
        ledger,
        shadow_executor=DurableShadowExecutor(store),
    )

    result = coordinator.run(
        plan=plan,
        now=now,
        shadow_account=account,
        shadow_quotes=quotes,
    )

    assert result.status is RunStatus.COMPLETED
    intents = ledger.intents_for_run(result.run_id)
    assert len(intents) == 1
    assert [event.state.value for event in ledger.events_for(intents[0].client_order_id)] == [
        "accepted",
        "filled",
    ]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(PortfolioSnapshotRecord)) == 1


def test_shadow_config_change_creates_a_new_epoch_and_account() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    first = ShadowAccountStore(
        engine,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        config_sha256="a" * 64,
    )
    changed = ShadowAccountStore(
        engine,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        config_sha256="b" * 64,
    )

    first.load_or_create("shadow-account", Decimal("100000"))
    changed.load_or_create("shadow-account", Decimal("100000"))

    assert first.shadow_account_id("shadow-account") != changed.shadow_account_id(
        "shadow-account"
    )
    assert first.epoch_id("shadow-account") != changed.epoch_id("shadow-account")


def test_shadow_existing_epoch_rejects_changed_starting_nav() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    store = ShadowAccountStore(
        engine,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        config_sha256="a" * 64,
    )
    store.load_or_create("shadow-account", Decimal("100000"))

    import pytest

    with pytest.raises(RuntimeError, match="starting NAV"):
        store.load_or_create("shadow-account", Decimal("100001"))
