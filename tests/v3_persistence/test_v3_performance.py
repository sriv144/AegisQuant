from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.db.v3_models import (
    BenchmarkMark,
    PortfolioSnapshotRecord,
    PositionSnapshotRecord,
    StrategyEpoch,
    V3Base,
)
from src.db.v3_performance import EODPerformanceStore
from src.db.v3_shadow import ShadowAccountStore
from src.execution.v3 import PositionSnapshot
from src.v3.runtime_input import BenchmarkRuntimeMark


def _mark(session_date: date, level: str, observed_at: datetime) -> BenchmarkRuntimeMark:
    return BenchmarkRuntimeMark(
        session_date=session_date,
        symbol="SPY",
        total_return_level=Decimal(level),
        daily_total_return=None,
        source="fixture",
        source_sha256=(session_date.isoformat().replace("-", "") * 8)[:64],
        observed_at=observed_at,
    )


def test_eod_store_persists_same_epoch_spy_relative_performance() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    config_sha = "a" * 64
    account = ShadowAccountStore(
        engine,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        config_sha256=config_sha,
    ).load_or_create("shadow", Decimal("100000"))
    store = EODPerformanceStore(
        engine,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        config_sha256=config_sha,
    )
    first_at = datetime(2026, 7, 1, 21, 0, tzinfo=UTC)
    second_at = datetime(2026, 7, 2, 21, 0, tzinfo=UTC)
    with Session(engine) as session:
        epoch = session.scalar(select(StrategyEpoch))
        assert epoch is not None
        epoch.activated_at = datetime(2026, 7, 1, 13, 0, tzinfo=UTC)
        session.commit()

    first = store.record_shadow(
        run_id="eod-1",
        account=account,
        quotes={},
        benchmark=_mark(date(2026, 7, 1), "100", first_at),
        observed_at=first_at,
    )
    second = store.record_shadow(
        run_id="eod-2",
        account=account,
        quotes={},
        benchmark=_mark(date(2026, 7, 2), "101", second_at),
        observed_at=second_at,
    )

    assert first.cumulative_benchmark_return == Decimal("0")
    assert second.cumulative_benchmark_return == Decimal("0.01")
    assert second.cumulative_excess_return == Decimal("-0.01")
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(BenchmarkMark)) == 2
        assert (
            session.scalar(select(func.count()).select_from(PortfolioSnapshotRecord))
            == 2
        )


def test_paper_eod_preserves_split_legacy_and_v3_lots() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    config_sha = "b" * 64
    first_at = datetime(2026, 7, 1, 20, 0, tzinfo=UTC)
    eod_at = datetime(2026, 7, 2, 21, 0, tzinfo=UTC)
    with Session(engine) as session:
        session.add(
            StrategyEpoch(
                epoch_id="paper-epoch",
                account_key="paper-account",
                account_fingerprint="fingerprint",
                mode="paper",
                starting_nav=Decimal("1500"),
                strategy_id="spy_xsmom_core_satellite",
                strategy_version="3.0.0",
                config_sha256=config_sha,
                activated_at=first_at,
                migration_cost=Decimal("0"),
            )
        )
        session.add(
            PortfolioSnapshotRecord(
                snapshot_id="paper-completion",
                run_id=None,
                epoch_id="paper-epoch",
                account_key="paper-account",
                mode="paper",
                observed_at=first_at,
                session_date=first_at.date(),
                nav=Decimal("1500"),
                cash=Decimal("0"),
                invested_weight=Decimal("1"),
                peak_nav=Decimal("1500"),
                drawdown=Decimal("0"),
                beta=None,
                tracking_error=None,
                cumulative_return=Decimal("0"),
                cumulative_benchmark_return=None,
                cumulative_excess_return=None,
            )
        )
        session.flush()
        session.add_all(
            [
                PositionSnapshotRecord(
                    snapshot_id="paper-completion",
                    symbol="SPY",
                    sleeve="legacy_unattributed",
                    attribution="LEGACY_UNATTRIBUTED",
                    quantity=Decimal("10"),
                    market_price=Decimal("100"),
                    market_value=Decimal("1000"),
                    weight=Decimal("0.666666666667"),
                ),
                PositionSnapshotRecord(
                    snapshot_id="paper-completion",
                    symbol="SPY",
                    sleeve="core",
                    attribution="V3_ATTRIBUTED",
                    quantity=Decimal("5"),
                    market_price=Decimal("100"),
                    market_value=Decimal("500"),
                    weight=Decimal("0.333333333333"),
                ),
            ]
        )
        session.commit()

    store = EODPerformanceStore(
        engine,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        config_sha256=config_sha,
    )
    result = store.record_paper(
        run_id="",
        account_key="paper-account",
        nav=Decimal("1515"),
        cash=Decimal("0"),
        positions=(PositionSnapshot("SPY", Decimal("15"), Decimal("101")),),
        benchmark=_mark(date(2026, 7, 2), "101", eod_at),
        observed_at=eod_at,
    )

    with Session(engine) as session:
        rows = session.scalars(
            select(PositionSnapshotRecord).where(
                PositionSnapshotRecord.snapshot_id == result.snapshot_id
            )
        ).all()
        assert {row.attribution: Decimal(row.quantity) for row in rows} == {
            "LEGACY_UNATTRIBUTED": Decimal("10"),
            "V3_ATTRIBUTED": Decimal("5"),
        }
