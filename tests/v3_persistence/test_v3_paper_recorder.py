from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.db.v3_ledger import SQLAlchemyLedger
from src.db.v3_models import (
    ExecutionRun,
    PortfolioSnapshotRecord,
    PositionSnapshotRecord,
    StrategyEpoch,
    V3Base,
)
from src.db.v3_paper import SQLPaperCompletionRecorder
from src.execution.v3 import (
    AccountSnapshot,
    OrderEvent,
    OrderIntent,
    OrderSide,
    OrderState,
    PaperCompletionSnapshot,
    PositionSnapshot,
    RunPurpose,
    RunRecord,
    RunStatus,
    TradingMode,
)


def test_paper_completion_starts_epoch_and_persists_final_truth() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    ledger = SQLAlchemyLedger(engine=engine)
    now = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)
    ledger.create_run(
        RunRecord(
            run_id="paper-run",
            decision_key="strategy|3.0.0|account|paper|2026-07",
            strategy_id="strategy",
            strategy_version="3.0.0",
            account_key="account",
            mode=TradingMode.PAPER,
            purpose=RunPurpose.REBALANCE,
            target_hash="a" * 64,
            created_at=now,
        )
    )
    intent = OrderIntent(
        client_order_id="aq3-p-202607-abcdefghijklmnopqrst",
        run_id="paper-run",
        decision_key="strategy|3.0.0|account|paper|2026-07",
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.99"),
        arrival_price=Decimal("100"),
        created_at=now,
        notional=Decimal("99000"),
    )
    event = OrderEvent(
        event_id="event-filled",
        client_order_id=intent.client_order_id,
        state=OrderState.FILLED,
        observed_at=now,
        broker_order_id="broker-order",
        filled_quantity=Decimal("989"),
        filled_average_price=Decimal("100.10"),
    )
    snapshot = PaperCompletionSnapshot(
        run_id="paper-run",
        decision_key=intent.decision_key,
        target_hash="a" * 64,
        account=AccountSnapshot(
            account_key="account",
            equity=Decimal("99901.10"),
            cash=Decimal("901.10"),
            buying_power=Decimal("901.10"),
            status="active",
            observed_at=now,
        ),
        positions=(
            PositionSnapshot(
                symbol="SPY", quantity=Decimal("989"), market_price=Decimal("100.10")
            ),
        ),
        intents=(intent,),
        events=(event,),
        observed_at=now,
        target_weights={"SPY": Decimal("0.99")},
    )
    recorder = SQLPaperCompletionRecorder(
        engine,
        strategy_id="strategy",
        strategy_version="3.0.0",
        config_sha256="b" * 64,
    )

    recorder.record_paper_completion(snapshot)

    with Session(engine) as session:
        epoch = session.scalar(select(StrategyEpoch))
        assert epoch is not None
        assert Decimal(epoch.migration_cost) == Decimal("98.9000000000")
        assert session.scalar(select(func.count()).select_from(PortfolioSnapshotRecord)) == 1
        position = session.scalar(select(PositionSnapshotRecord))
        assert position is not None
        assert position.attribution == "V3_ATTRIBUTED"
        run = session.get(ExecutionRun, "paper-run")
        assert run.metadata_json["post_fill_drift_within_50bps"] is True


def test_paper_completion_preserves_off_target_legacy_attribution() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    ledger = SQLAlchemyLedger(engine=engine)
    now = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)
    run = RunRecord(
        run_id="legacy-residual-run",
        decision_key="strategy|3.0.0|account|paper|2026-07",
        strategy_id="strategy",
        strategy_version="3.0.0",
        account_key="account",
        mode=TradingMode.PAPER,
        purpose=RunPurpose.REBALANCE,
        target_hash="c" * 64,
        created_at=now,
    )
    ledger.create_run(run)
    snapshot = PaperCompletionSnapshot(
        run_id=run.run_id,
        decision_key=run.decision_key,
        target_hash=run.target_hash,
        account=AccountSnapshot(
            account_key="account",
            equity=Decimal("100000"),
            cash=Decimal("99950"),
            buying_power=Decimal("99950"),
            status="active",
            observed_at=now,
        ),
        positions=(
            PositionSnapshot(
                symbol="OLD", quantity=Decimal("0.5"), market_price=Decimal("100")
            ),
        ),
        intents=(),
        events=(),
        observed_at=now,
        target_weights={"SPY": Decimal("0.69")},
    )
    SQLPaperCompletionRecorder(
        engine,
        strategy_id="strategy",
        strategy_version="3.0.0",
        config_sha256="d" * 64,
    ).record_paper_completion(snapshot)

    with Session(engine) as session:
        position = session.scalar(select(PositionSnapshotRecord))
        assert position is not None
        assert position.attribution == "LEGACY_UNATTRIBUTED"
        assert position.sleeve == "legacy_unattributed"


def test_paper_completion_splits_legacy_and_v3_lots_without_relabeling() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    ledger = SQLAlchemyLedger(engine=engine)
    now = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)
    with Session(engine) as session:
        session.add(
            PortfolioSnapshotRecord(
                snapshot_id="legacy-bootstrap",
                run_id=None,
                epoch_id=None,
                account_key="account",
                mode="paper",
                observed_at=now - timedelta(minutes=1),
                session_date=now.date(),
                nav=Decimal("1000"),
                cash=Decimal("0"),
                invested_weight=Decimal("1"),
                peak_nav=Decimal("1000"),
                drawdown=Decimal("0"),
                beta=None,
                tracking_error=None,
                cumulative_return=None,
                cumulative_benchmark_return=None,
                cumulative_excess_return=None,
            )
        )
        session.flush()
        session.add(
            PositionSnapshotRecord(
                snapshot_id="legacy-bootstrap",
                symbol="SPY",
                sleeve="legacy_unattributed",
                attribution="LEGACY_UNATTRIBUTED",
                quantity=Decimal("10"),
                market_price=Decimal("100"),
                market_value=Decimal("1000"),
                weight=Decimal("1"),
            )
        )
        session.commit()

    run = RunRecord(
        run_id="mixed-lot-run",
        decision_key="strategy|3.0.0|account|paper|2026-07",
        strategy_id="strategy",
        strategy_version="3.0.0",
        account_key="account",
        mode=TradingMode.PAPER,
        purpose=RunPurpose.REBALANCE,
        target_hash="e" * 64,
        created_at=now,
    )
    ledger.create_run(run)
    intent = OrderIntent(
        client_order_id="aq3-p-202607-lotssplitabcdefghijk",
        run_id=run.run_id,
        decision_key=run.decision_key,
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("1"),
        arrival_price=Decimal("100"),
        created_at=now,
        notional=Decimal("500"),
    )
    event = OrderEvent(
        event_id="mixed-fill",
        client_order_id=intent.client_order_id,
        state=OrderState.FILLED,
        observed_at=now,
        filled_quantity=Decimal("5"),
        filled_average_price=Decimal("100"),
    )
    snapshot = PaperCompletionSnapshot(
        run_id=run.run_id,
        decision_key=run.decision_key,
        target_hash=run.target_hash,
        account=AccountSnapshot(
            account_key="account",
            equity=Decimal("1500"),
            cash=Decimal("0"),
            buying_power=Decimal("0"),
            status="active",
            observed_at=now,
        ),
        positions=(PositionSnapshot("SPY", Decimal("15"), Decimal("100")),),
        intents=(intent,),
        events=(event,),
        observed_at=now,
        target_weights={"SPY": Decimal("1")},
    )
    recorder = SQLPaperCompletionRecorder(
        engine,
        strategy_id="strategy",
        strategy_version="3.0.0",
        config_sha256="f" * 64,
    )
    recorder.record_paper_completion(snapshot)

    with Session(engine) as session:
        final_id = session.scalar(
            select(PortfolioSnapshotRecord.snapshot_id).where(
                PortfolioSnapshotRecord.run_id == run.run_id
            )
        )
        rows = session.scalars(
            select(PositionSnapshotRecord).where(
                PositionSnapshotRecord.snapshot_id == final_id
            )
        ).all()
        quantities = {row.attribution: Decimal(row.quantity) for row in rows}
        assert quantities == {
            "LEGACY_UNATTRIBUTED": Decimal("10"),
            "V3_ATTRIBUTED": Decimal("5"),
        }

    # A crash-retry can observe moved marks but the already committed final
    # snapshot remains the authoritative completion for this run.
    recorder.record_paper_completion(
        replace(
            snapshot,
            account=replace(snapshot.account, equity=Decimal("1515")),
            positions=(PositionSnapshot("SPY", Decimal("15"), Decimal("101")),),
            observed_at=now + timedelta(minutes=1),
        )
    )


def test_durable_de_risk_activation_timestamp_tracks_current_episode() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    ledger = SQLAlchemyLedger(engine=engine)
    config_sha = "9" * 64
    recorder = SQLPaperCompletionRecorder(
        engine,
        strategy_id="strategy",
        strategy_version="3.0.0",
        config_sha256=config_sha,
    )
    base = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)
    for index, de_risked in enumerate((False, True, True)):
        run_id = f"risk-run-{index}"
        at = base + timedelta(days=index * 31)
        ledger.create_run(
            RunRecord(
                run_id=run_id,
                decision_key=f"strategy|3.0.0|account|paper|2026-{6 + index:02d}",
                strategy_id="strategy",
                strategy_version="3.0.0",
                account_key="account",
                mode=TradingMode.PAPER,
                purpose=RunPurpose.REBALANCE,
                target_hash=str(index) * 64,
                created_at=at,
            )
        )
        ledger.update_run(
            run_id,
            RunStatus.COMPLETED,
            at,
            metadata={
                "plan_metadata": {
                    "config_sha256": config_sha,
                    "de_risk_active": de_risked,
                }
            },
        )

    assert recorder.is_de_risked("account") is True
    assert recorder.de_risked_since("account") == base + timedelta(days=31)
