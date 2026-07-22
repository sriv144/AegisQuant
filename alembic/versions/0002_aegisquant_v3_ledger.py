"""Create the isolated AegisQuant v3 execution and research ledger.

Revision ID: 0002_v3_ledger
Revises: 0001_legacy_baseline
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_v3_ledger"
down_revision: str | None = "0001_legacy_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MONEY = sa.Numeric(30, 10)
WEIGHT = sa.Numeric(20, 12)


def upgrade() -> None:
    op.create_table(
        "strategy_epochs",
        sa.Column("epoch_id", sa.String(64), primary_key=True),
        sa.Column("account_key", sa.String(64), nullable=False),
        sa.Column("account_fingerprint", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("starting_nav", MONEY, nullable=False),
        sa.Column("strategy_id", sa.String(96), nullable=False),
        sa.Column("strategy_version", sa.String(32), nullable=False),
        sa.Column("config_sha256", sa.String(64), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("migration_cost", MONEY, nullable=False, server_default="0"),
        sa.CheckConstraint("mode IN ('shadow', 'paper')", name="ck_strategy_epochs_mode"),
        sa.CheckConstraint("starting_nav >= 0", name="ck_strategy_epochs_starting_nav"),
    )
    op.create_index("ix_strategy_epochs_account_key", "strategy_epochs", ["account_key"])

    op.create_table(
        "execution_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("epoch_id", sa.String(64), sa.ForeignKey("strategy_epochs.epoch_id")),
        sa.Column("strategy_id", sa.String(96), nullable=False),
        sa.Column("strategy_version", sa.String(32), nullable=False),
        sa.Column("account_key", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("purpose", sa.String(16), nullable=False),
        sa.Column("decision_key", sa.String(255), nullable=False),
        sa.Column("trigger", sa.String(64), nullable=False, server_default="manual"),
        sa.Column("commit_sha", sa.String(64), nullable=False, server_default="unknown"),
        sa.Column("target_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("failure_reason", sa.Text()),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "strategy_id", "strategy_version", "account_key", "mode", "decision_key",
            name="uq_execution_runs_decision",
        ),
        sa.CheckConstraint("mode IN ('shadow', 'paper')", name="ck_execution_runs_mode"),
    )
    op.create_index("ix_execution_runs_epoch_id", "execution_runs", ["epoch_id"])

    op.create_table(
        "execution_leases",
        sa.Column("lease_key", sa.String(160), primary_key=True),
        sa.Column("account_key", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("holder_run_id", sa.String(64), nullable=False),
        sa.Column("decision_key", sa.String(255), nullable=False, unique=True),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True)),
        sa.Column("completed_rebalance", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("account_key", "mode", name="uq_execution_leases_account_mode"),
        sa.CheckConstraint("mode IN ('shadow', 'paper')", name="ck_execution_leases_mode"),
    )

    op.create_table(
        "order_intents",
        sa.Column("client_order_id", sa.String(48), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("execution_runs.run_id"), nullable=False),
        sa.Column("decision_key", sa.String(255), nullable=False),
        sa.Column("sleeve", sa.String(96), nullable=False),
        sa.Column("symbol", sa.String(24), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("requested_quantity", MONEY),
        sa.Column("requested_notional", MONEY),
        sa.Column("frozen_order_amount", sa.String(64), nullable=False),
        sa.Column("target_weight", WEIGHT, nullable=False),
        sa.Column("arrival_bid", MONEY, nullable=False),
        sa.Column("arrival_ask", MONEY, nullable=False),
        sa.Column("arrival_quote_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("run_id", "symbol", "side", name="uq_order_intents_run_symbol_side"),
        sa.CheckConstraint("side IN ('buy', 'sell')", name="ck_order_intents_side"),
        sa.CheckConstraint(
            "(requested_quantity IS NULL) <> (requested_notional IS NULL)",
            name="ck_order_intents_exactly_one_amount",
        ),
    )
    op.create_index("ix_order_intents_run_id", "order_intents", ["run_id"])

    op.create_table(
        "order_events",
        sa.Column("record_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(64), nullable=False, unique=True),
        sa.Column(
            "client_order_id", sa.String(48), sa.ForeignKey("order_intents.client_order_id"), nullable=False
        ),
        sa.Column("broker_order_id", sa.String(96)),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("filled_quantity", MONEY, nullable=False, server_default="0"),
        sa.Column("filled_average_price", MONEY),
        sa.Column("slippage_bps", WEIGHT),
        sa.Column("reason", sa.Text()),
        sa.Column("raw_status", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_order_events_client_order_id", "order_events", ["client_order_id"])
    op.create_index("ix_order_events_broker_order_id", "order_events", ["broker_order_id"])

    op.create_table(
        "portfolio_snapshots",
        sa.Column("snapshot_id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("execution_runs.run_id")),
        sa.Column("epoch_id", sa.String(64), sa.ForeignKey("strategy_epochs.epoch_id")),
        sa.Column("account_key", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("nav", MONEY, nullable=False),
        sa.Column("cash", MONEY, nullable=False),
        sa.Column("invested_weight", WEIGHT, nullable=False),
        sa.Column("peak_nav", MONEY, nullable=False),
        sa.Column("drawdown", WEIGHT, nullable=False),
        sa.Column("beta", WEIGHT),
        sa.Column("tracking_error", WEIGHT),
        sa.Column("cumulative_return", WEIGHT),
        sa.Column("cumulative_benchmark_return", WEIGHT),
        sa.Column("cumulative_excess_return", WEIGHT),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("mode IN ('shadow', 'paper')", name="ck_portfolio_snapshots_mode"),
        sa.CheckConstraint("drawdown >= 0", name="ck_portfolio_snapshots_drawdown"),
    )
    op.create_index("ix_portfolio_snapshots_run_id", "portfolio_snapshots", ["run_id"])
    op.create_index("ix_portfolio_snapshots_epoch_id", "portfolio_snapshots", ["epoch_id"])
    op.create_index("ix_portfolio_snapshots_session_date", "portfolio_snapshots", ["session_date"])

    op.create_table(
        "position_snapshots",
        sa.Column("position_snapshot_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "snapshot_id", sa.String(64), sa.ForeignKey("portfolio_snapshots.snapshot_id"), nullable=False
        ),
        sa.Column("symbol", sa.String(24), nullable=False),
        sa.Column("sleeve", sa.String(96), nullable=False),
        sa.Column("attribution", sa.String(64), nullable=False),
        sa.Column("quantity", MONEY, nullable=False),
        sa.Column("market_price", MONEY, nullable=False),
        sa.Column("market_value", MONEY, nullable=False),
        sa.Column("weight", WEIGHT, nullable=False),
        sa.UniqueConstraint(
            "snapshot_id",
            "symbol",
            "attribution",
            name="uq_position_snapshots_symbol_attribution",
        ),
    )
    op.create_index("ix_position_snapshots_snapshot_id", "position_snapshots", ["snapshot_id"])

    op.create_table(
        "shadow_accounts",
        sa.Column("shadow_account_id", sa.String(64), primary_key=True),
        sa.Column("epoch_id", sa.String(64), sa.ForeignKey("strategy_epochs.epoch_id"), nullable=False, unique=True),
        sa.Column("account_key", sa.String(64), nullable=False),
        sa.Column("cash", MONEY, nullable=False),
        sa.Column("peak_nav", MONEY, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "shadow_positions",
        sa.Column(
            "shadow_account_id", sa.String(64), sa.ForeignKey("shadow_accounts.shadow_account_id"),
            primary_key=True,
        ),
        sa.Column("symbol", sa.String(24), primary_key=True),
        sa.Column("sleeve", sa.String(96), nullable=False),
        sa.Column("quantity", MONEY, nullable=False),
        sa.Column("cost_basis", MONEY, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "benchmark_marks",
        sa.Column("benchmark_mark_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("account_key", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(24), nullable=False, server_default="SPY"),
        sa.Column("total_return_level", MONEY, nullable=False),
        sa.Column("daily_total_return", WEIGHT),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "account_key", "mode", "session_date", "symbol", name="uq_benchmark_marks_session"
        ),
        sa.CheckConstraint("mode IN ('shadow', 'paper')", name="ck_benchmark_marks_mode"),
    )

    op.create_table(
        "data_manifests",
        sa.Column("manifest_id", sa.String(64), primary_key=True),
        sa.Column("dataset_name", sa.String(128), nullable=False),
        sa.Column("data_tier", sa.String(32), nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("availability_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("freeze_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("coverage", WEIGHT, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("validation_status", sa.String(32), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("dataset_name", "sha256", name="uq_data_manifests_dataset_sha"),
    )

    op.create_table(
        "experiment_runs",
        sa.Column("experiment_id", sa.String(64), primary_key=True),
        sa.Column("strategy_id", sa.String(96), nullable=False),
        sa.Column("strategy_version", sa.String(32), nullable=False),
        sa.Column("config_sha256", sa.String(64), nullable=False),
        sa.Column("data_manifest_sha256", sa.String(64), nullable=False),
        sa.Column("trial_family", sa.String(96), nullable=False),
        sa.Column("split_name", sa.String(64), nullable=False),
        sa.Column("parameters_json", sa.JSON(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("promotion_status", sa.String(32), nullable=False),
        sa.Column("gate_failures_json", sa.JSON(), nullable=False),
        sa.Column("commit_sha", sa.String(64), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "experiment_runs",
        "data_manifests",
        "benchmark_marks",
        "shadow_positions",
        "shadow_accounts",
        "position_snapshots",
        "portfolio_snapshots",
        "order_events",
        "order_intents",
        "execution_leases",
        "execution_runs",
        "strategy_epochs",
    ):
        op.drop_table(table)
