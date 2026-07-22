"""Isolated SQLAlchemy schema for the AegisQuant v3 execution truth.

The legacy ORM in :mod:`src.db.models` remains untouched.  These tables use
string identifiers and portable SQLAlchemy types so research and shadow tests
can run on SQLite, while paper execution is separately gated to PostgreSQL.
Order intents, order events, snapshots, benchmark marks, data manifests and
experiment runs are append-only by application contract.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


MONEY = Numeric(30, 10)
WEIGHT = Numeric(20, 12)


class V3Base(DeclarativeBase):
    """Metadata root deliberately separate from the legacy schema."""


class StrategyEpoch(V3Base):
    __tablename__ = "strategy_epochs"

    epoch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    account_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    starting_nav: Mapped[object] = mapped_column(MONEY, nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(96), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    activated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    migration_cost: Mapped[object] = mapped_column(MONEY, nullable=False, default=0)

    __table_args__ = (
        CheckConstraint("mode IN ('shadow', 'paper')", name="ck_strategy_epochs_mode"),
        CheckConstraint("starting_nav >= 0", name="ck_strategy_epochs_starting_nav"),
    )


class ExecutionRun(V3Base):
    __tablename__ = "execution_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    epoch_id: Mapped[str | None] = mapped_column(
        ForeignKey("strategy_epochs.epoch_id"), nullable=True, index=True
    )
    strategy_id: Mapped[str] = mapped_column(String(96), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    account_key: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    purpose: Mapped[str] = mapped_column(String(16), nullable=False)
    decision_key: Mapped[str] = mapped_column(String(255), nullable=False)
    trigger: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    target_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[object] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "strategy_id",
            "strategy_version",
            "account_key",
            "mode",
            "decision_key",
            name="uq_execution_runs_decision",
        ),
        CheckConstraint("mode IN ('shadow', 'paper')", name="ck_execution_runs_mode"),
    )


class OrderIntentRecord(V3Base):
    __tablename__ = "order_intents"

    client_order_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("execution_runs.run_id"), nullable=False, index=True
    )
    decision_key: Mapped[str] = mapped_column(String(255), nullable=False)
    sleeve: Mapped[str] = mapped_column(String(96), nullable=False)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    requested_quantity: Mapped[object | None] = mapped_column(MONEY, nullable=True)
    requested_notional: Mapped[object | None] = mapped_column(MONEY, nullable=True)
    frozen_order_amount: Mapped[str] = mapped_column(String(64), nullable=False)
    target_weight: Mapped[object] = mapped_column(WEIGHT, nullable=False)
    arrival_bid: Mapped[object] = mapped_column(MONEY, nullable=False)
    arrival_ask: Mapped[object] = mapped_column(MONEY, nullable=False)
    arrival_quote_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("run_id", "symbol", "side", name="uq_order_intents_run_symbol_side"),
        CheckConstraint("side IN ('buy', 'sell')", name="ck_order_intents_side"),
        CheckConstraint(
            "(requested_quantity IS NULL) <> (requested_notional IS NULL)",
            name="ck_order_intents_exactly_one_amount",
        ),
    )


class OrderEventRecord(V3Base):
    __tablename__ = "order_events"

    record_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    client_order_id: Mapped[str] = mapped_column(
        ForeignKey("order_intents.client_order_id"), nullable=False, index=True
    )
    broker_order_id: Mapped[str | None] = mapped_column(String(96), nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    filled_quantity: Mapped[object] = mapped_column(MONEY, nullable=False, default=0)
    filled_average_price: Mapped[object | None] = mapped_column(MONEY, nullable=True)
    slippage_bps: Mapped[object | None] = mapped_column(WEIGHT, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ExecutionLease(V3Base):
    __tablename__ = "execution_leases"

    lease_key: Mapped[str] = mapped_column(String(160), primary_key=True)
    account_key: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    holder_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    acquired_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    released_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_rebalance: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("account_key", "mode", name="uq_execution_leases_account_mode"),
        CheckConstraint("mode IN ('shadow', 'paper')", name="ck_execution_leases_mode"),
    )


class PortfolioSnapshotRecord(V3Base):
    __tablename__ = "portfolio_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("execution_runs.run_id"), nullable=True, index=True
    )
    epoch_id: Mapped[str | None] = mapped_column(
        ForeignKey("strategy_epochs.epoch_id"), nullable=True, index=True
    )
    account_key: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    observed_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    session_date: Mapped[object] = mapped_column(Date, nullable=False, index=True)
    nav: Mapped[object] = mapped_column(MONEY, nullable=False)
    cash: Mapped[object] = mapped_column(MONEY, nullable=False)
    invested_weight: Mapped[object] = mapped_column(WEIGHT, nullable=False)
    peak_nav: Mapped[object] = mapped_column(MONEY, nullable=False)
    drawdown: Mapped[object] = mapped_column(WEIGHT, nullable=False)
    beta: Mapped[object | None] = mapped_column(WEIGHT, nullable=True)
    tracking_error: Mapped[object | None] = mapped_column(WEIGHT, nullable=True)
    cumulative_return: Mapped[object | None] = mapped_column(WEIGHT, nullable=True)
    cumulative_benchmark_return: Mapped[object | None] = mapped_column(WEIGHT, nullable=True)
    cumulative_excess_return: Mapped[object | None] = mapped_column(WEIGHT, nullable=True)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("mode IN ('shadow', 'paper')", name="ck_portfolio_snapshots_mode"),
        CheckConstraint("drawdown >= 0", name="ck_portfolio_snapshots_drawdown"),
    )


class PositionSnapshotRecord(V3Base):
    __tablename__ = "position_snapshots"

    position_snapshot_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("portfolio_snapshots.snapshot_id"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(24), nullable=False)
    sleeve: Mapped[str] = mapped_column(String(96), nullable=False)
    attribution: Mapped[str] = mapped_column(String(64), nullable=False)
    quantity: Mapped[object] = mapped_column(MONEY, nullable=False)
    market_price: Mapped[object] = mapped_column(MONEY, nullable=False)
    market_value: Mapped[object] = mapped_column(MONEY, nullable=False)
    weight: Mapped[object] = mapped_column(WEIGHT, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "symbol",
            "attribution",
            name="uq_position_snapshots_symbol_attribution",
        ),
    )


class ShadowAccountRecord(V3Base):
    __tablename__ = "shadow_accounts"

    shadow_account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    epoch_id: Mapped[str] = mapped_column(
        ForeignKey("strategy_epochs.epoch_id"), nullable=False, unique=True
    )
    account_key: Mapped[str] = mapped_column(String(64), nullable=False)
    cash: Mapped[object] = mapped_column(MONEY, nullable=False)
    peak_nav: Mapped[object] = mapped_column(MONEY, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ShadowPositionRecord(V3Base):
    __tablename__ = "shadow_positions"

    shadow_account_id: Mapped[str] = mapped_column(
        ForeignKey("shadow_accounts.shadow_account_id"), primary_key=True
    )
    symbol: Mapped[str] = mapped_column(String(24), primary_key=True)
    sleeve: Mapped[str] = mapped_column(String(96), nullable=False)
    quantity: Mapped[object] = mapped_column(MONEY, nullable=False)
    cost_basis: Mapped[object] = mapped_column(MONEY, nullable=False)
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class BenchmarkMark(V3Base):
    __tablename__ = "benchmark_marks"

    benchmark_mark_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_key: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    session_date: Mapped[object] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False, default="SPY")
    total_return_level: Mapped[object] = mapped_column(MONEY, nullable=False)
    daily_total_return: Mapped[object | None] = mapped_column(WEIGHT, nullable=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "account_key", "mode", "session_date", "symbol", name="uq_benchmark_marks_session"
        ),
        CheckConstraint("mode IN ('shadow', 'paper')", name="ck_benchmark_marks_mode"),
    )


class DataManifestRecord(V3Base):
    __tablename__ = "data_manifests"

    manifest_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    data_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    availability_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    freeze_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    coverage: Mapped[object] = mapped_column(WEIGHT, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    warnings_json: Mapped[object] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("dataset_name", "sha256", name="uq_data_manifests_dataset_sha"),
    )


class ExperimentRun(V3Base):
    __tablename__ = "experiment_runs"

    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(96), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    data_manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    trial_family: Mapped[str] = mapped_column(String(96), nullable=False)
    split_name: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters_json: Mapped[object] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[object] = mapped_column(JSON, nullable=False)
    warnings_json: Mapped[object] = mapped_column(JSON, nullable=False, default=list)
    promotion_status: Mapped[str] = mapped_column(String(32), nullable=False)
    gate_failures_json: Mapped[object] = mapped_column(JSON, nullable=False, default=list)
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    attempted_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


V3_TABLE_NAMES = frozenset(V3Base.metadata.tables)


__all__ = [
    "BenchmarkMark",
    "DataManifestRecord",
    "ExecutionLease",
    "ExecutionRun",
    "ExperimentRun",
    "OrderEventRecord",
    "OrderIntentRecord",
    "PortfolioSnapshotRecord",
    "PositionSnapshotRecord",
    "ShadowAccountRecord",
    "ShadowPositionRecord",
    "StrategyEpoch",
    "V3Base",
    "V3_TABLE_NAMES",
]
