"""SQLAlchemy implementation of the v3 execution ledger contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Mapping

from sqlalchemy import Engine, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.db.v3_models import (
    ExecutionLease,
    ExecutionRun,
    OrderEventRecord,
    OrderIntentRecord,
    V3_TABLE_NAMES,
)
from src.execution.v3.contracts import (
    OrderIntent,
    OrderSide,
    OrderState,
    RunPurpose,
    RunStatus,
    TradingMode,
)
from src.execution.v3.ledger import ConflictingIntent, OrderEvent, RunRecord
from src.execution.v3.lifecycle import validate_order_transition


class V3SchemaMissing(RuntimeError):
    """Raised when Alembic has not created every required v3 table."""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _mode_text(mode: TradingMode | str) -> str:
    return mode.value if isinstance(mode, TradingMode) else TradingMode(mode).value


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


class SQLAlchemyLedger:
    """Durable ledger with transactional leases and idempotent frozen intents.

    Migrations are never run implicitly.  Callers must use Alembic (or create
    the isolated metadata in a test fixture) before constructing the adapter.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
        lease_ttl: timedelta = timedelta(minutes=20),
        verify_schema: bool = True,
    ) -> None:
        if engine is None:
            if not database_url:
                raise ValueError("database_url or engine is required")
            engine = create_engine(database_url, pool_pre_ping=True)
        self.engine = engine
        self.lease_ttl = lease_ttl
        self.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        self._schema_verified = False
        if verify_schema:
            existing = set(inspect(engine).get_table_names())
            missing = set(V3_TABLE_NAMES) - existing
            if missing:
                raise V3SchemaMissing(
                    "v3 database schema is incomplete; run 'alembic upgrade head': "
                    + ", ".join(sorted(missing))
                )
            self._schema_verified = True

    @property
    def paper_durable_truth(self) -> bool:
        """Capability checked by the coordinator before any paper broker read."""

        return self._schema_verified and self.engine.dialect.name == "postgresql"

    def acquire_lease(self, account_key: str, mode: TradingMode, owner: str) -> bool:
        now = datetime.now(UTC)
        mode_text = _mode_text(mode)
        lease_key = f"{account_key}|{mode_text}"
        try:
            with self.SessionLocal.begin() as session:
                row = session.get(ExecutionLease, lease_key, with_for_update=True)
                if row is not None and row.released_at is None and _utc(row.expires_at) > now:
                    return row.holder_run_id == owner
                if row is None:
                    session.add(
                        ExecutionLease(
                            lease_key=lease_key,
                            account_key=account_key,
                            mode=mode_text,
                            holder_run_id=owner,
                            decision_key=f"lease|{lease_key}|{owner}",
                            acquired_at=now,
                            expires_at=now + self.lease_ttl,
                            released_at=None,
                            completed_rebalance=False,
                        )
                    )
                else:
                    row.holder_run_id = owner
                    row.decision_key = f"lease|{lease_key}|{owner}"
                    row.acquired_at = now
                    row.expires_at = now + self.lease_ttl
                    row.released_at = None
                    row.completed_rebalance = False
            return True
        except IntegrityError:
            return False

    def release_lease(self, account_key: str, mode: TradingMode, owner: str) -> None:
        lease_key = f"{account_key}|{_mode_text(mode)}"
        with self.SessionLocal.begin() as session:
            row = session.get(ExecutionLease, lease_key, with_for_update=True)
            if row is not None and row.holder_run_id == owner:
                row.released_at = datetime.now(UTC)

    def renew_lease(self, account_key: str, mode: TradingMode, owner: str) -> bool:
        lease_key = f"{account_key}|{_mode_text(mode)}"
        now = datetime.now(UTC)
        with self.SessionLocal.begin() as session:
            row = session.get(ExecutionLease, lease_key, with_for_update=True)
            if (
                row is None
                or row.holder_run_id != owner
                or row.released_at is not None
            ):
                return False
            row.expires_at = now + self.lease_ttl
            return True

    def get_run_by_decision_key(self, decision_key: str) -> RunRecord | None:
        with self.SessionLocal() as session:
            row = session.scalar(
                select(ExecutionRun).where(ExecutionRun.decision_key == decision_key)
            )
            return None if row is None else self._to_run_record(row)

    def create_run(self, record: RunRecord) -> RunRecord:
        payload = ExecutionRun(
            run_id=record.run_id,
            epoch_id=None,
            strategy_id=record.strategy_id,
            strategy_version=record.strategy_version,
            account_key=record.account_key,
            mode=record.mode.value,
            purpose=record.purpose.value,
            decision_key=record.decision_key,
            trigger=str(record.metadata.get("trigger", "manual")),
            commit_sha=str(record.metadata.get("commit_sha", "unknown")),
            target_hash=record.target_hash,
            status=record.status.value if record.status is not None else "running",
            failure_reason=record.failure_reason or None,
            metadata_json=dict(record.metadata),
            started_at=record.created_at,
            completed_at=record.completed_at,
        )
        try:
            with self.SessionLocal.begin() as session:
                session.add(payload)
            return record
        except IntegrityError:
            existing = self.get_run_by_decision_key(record.decision_key)
            if existing is None:
                raise
            return existing

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        at: datetime,
        *,
        failure_reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        with self.SessionLocal.begin() as session:
            row = session.get(ExecutionRun, run_id, with_for_update=True)
            if row is None:
                raise KeyError(f"unknown execution run {run_id}")
            row.status = status.value
            row.completed_at = at
            row.failure_reason = failure_reason or None
            merged = dict(row.metadata_json or {})
            if metadata:
                merged.update(dict(metadata))
            row.metadata_json = merged
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        with self.SessionLocal() as session:
            row = session.get(ExecutionRun, run_id)
            if row is None:
                raise KeyError(f"unknown execution run {run_id}")
            return self._to_run_record(row)

    def add_intents(self, intents: tuple[OrderIntent, ...]) -> None:
        with self.SessionLocal.begin() as session:
            for intent in intents:
                existing = session.get(OrderIntentRecord, intent.client_order_id)
                if existing is not None:
                    if not self._intent_matches(existing, intent):
                        raise ConflictingIntent(
                            f"client order id {intent.client_order_id} maps to conflicting intents"
                        )
                    continue
                amount = intent.quantity if intent.quantity is not None else intent.notional
                assert amount is not None
                session.add(
                    OrderIntentRecord(
                        client_order_id=intent.client_order_id,
                        run_id=intent.run_id,
                        decision_key=intent.decision_key,
                        sleeve=intent.sleeve,
                        symbol=intent.symbol,
                        side=intent.side.value,
                        requested_quantity=intent.quantity,
                        requested_notional=intent.notional,
                        frozen_order_amount=_decimal_text(amount),
                        target_weight=intent.target_weight,
                        arrival_bid=intent.arrival_price,
                        arrival_ask=intent.arrival_price,
                        arrival_quote_at=intent.created_at,
                        created_at=intent.created_at,
                    )
                )

    def append_order_event(self, event: OrderEvent) -> None:
        with self.SessionLocal.begin() as session:
            intent = session.get(
                OrderIntentRecord, event.client_order_id, with_for_update=True
            )
            if intent is None:
                raise KeyError(f"unknown order intent {event.client_order_id}")
            duplicate = session.scalar(
                select(OrderEventRecord).where(OrderEventRecord.event_id == event.event_id)
            )
            if duplicate is not None:
                return
            prior = session.scalar(
                select(OrderEventRecord)
                .where(OrderEventRecord.client_order_id == event.client_order_id)
                .order_by(OrderEventRecord.observed_at.desc(), OrderEventRecord.record_id.desc())
                .limit(1)
            )
            previous_state = OrderState.INTENT if prior is None else OrderState(prior.state)
            validate_order_transition(previous_state, event.state)
            session.add(
                OrderEventRecord(
                    event_id=event.event_id,
                    client_order_id=event.client_order_id,
                    broker_order_id=event.broker_order_id or None,
                    state=event.state.value,
                    observed_at=event.observed_at,
                    filled_quantity=event.filled_quantity,
                    filled_average_price=event.filled_average_price,
                    slippage_bps=event.slippage_bps,
                    reason=str(event.details.get("reason", "")) or None,
                    raw_status=str(event.details.get("raw_status", "")) or None,
                )
            )

    def current_order_state(self, client_order_id: str) -> OrderState:
        with self.SessionLocal() as session:
            if session.get(OrderIntentRecord, client_order_id) is None:
                raise KeyError(f"unknown order intent {client_order_id}")
            row = session.scalar(
                select(OrderEventRecord)
                .where(OrderEventRecord.client_order_id == client_order_id)
                .order_by(OrderEventRecord.observed_at.desc(), OrderEventRecord.record_id.desc())
                .limit(1)
            )
            return OrderState.INTENT if row is None else OrderState(row.state)

    def events_for(self, client_order_id: str) -> tuple[OrderEvent, ...]:
        with self.SessionLocal() as session:
            rows = session.scalars(
                select(OrderEventRecord)
                .where(OrderEventRecord.client_order_id == client_order_id)
                .order_by(OrderEventRecord.observed_at, OrderEventRecord.record_id)
            ).all()
            return tuple(
                OrderEvent(
                    event_id=row.event_id,
                    client_order_id=row.client_order_id,
                    state=OrderState(row.state),
                    observed_at=_utc(row.observed_at),
                    broker_order_id=row.broker_order_id or "",
                    filled_quantity=Decimal(row.filled_quantity),
                    filled_average_price=(
                        None
                        if row.filled_average_price is None
                        else Decimal(row.filled_average_price)
                    ),
                    slippage_bps=(
                        None if row.slippage_bps is None else Decimal(row.slippage_bps)
                    ),
                    details={
                        key: value
                        for key, value in {
                            "reason": row.reason,
                            "raw_status": row.raw_status,
                        }.items()
                        if value
                    },
                )
                for row in rows
            )

    def get_intent(self, client_order_id: str) -> OrderIntent | None:
        with self.SessionLocal() as session:
            row = session.get(OrderIntentRecord, client_order_id)
            return None if row is None else self._to_order_intent(row)

    def intents_for_run(self, run_id: str) -> tuple[OrderIntent, ...]:
        with self.SessionLocal() as session:
            rows = session.scalars(
                select(OrderIntentRecord)
                .where(OrderIntentRecord.run_id == run_id)
                .order_by(OrderIntentRecord.symbol, OrderIntentRecord.side)
            ).all()
            return tuple(self._to_order_intent(row) for row in rows)

    def oldest_run_requiring_reconciliation(
        self, account_key: str, mode: TradingMode
    ) -> RunRecord | None:
        with self.SessionLocal() as session:
            rows = session.scalars(
                select(ExecutionRun)
                .where(
                    ExecutionRun.account_key == account_key,
                    ExecutionRun.mode == _mode_text(mode),
                    ExecutionRun.purpose == RunPurpose.REBALANCE.value,
                )
                .order_by(ExecutionRun.started_at.asc())
            ).all()
            for row in rows:
                intents = session.scalars(
                    select(OrderIntentRecord).where(OrderIntentRecord.run_id == row.run_id)
                ).all()
                states: list[OrderState] = []
                for intent in intents:
                    latest = session.scalar(
                        select(OrderEventRecord)
                        .where(
                            OrderEventRecord.client_order_id == intent.client_order_id
                        )
                        .order_by(
                            OrderEventRecord.observed_at.desc(),
                            OrderEventRecord.record_id.desc(),
                        )
                        .limit(1)
                    )
                    states.append(
                        OrderState.INTENT if latest is None else OrderState(latest.state)
                    )
                unresolved_states = {
                    OrderState.INTENT,
                    OrderState.ACCEPTED,
                    OrderState.PARTIALLY_FILLED,
                    OrderState.UNKNOWN,
                }
                if any(state in unresolved_states for state in states) or (
                    row.status != RunStatus.COMPLETED.value
                    and bool(states)
                    and all(state is OrderState.FILLED for state in states)
                ):
                    return self._to_run_record(row)
            return None

    # One-release compatibility alias for callers compiled against the first
    # v3 ledger draft.  New coordination always drains the oldest unresolved run.
    def latest_run_requiring_reconciliation(
        self, account_key: str, mode: TradingMode
    ) -> RunRecord | None:
        return self.oldest_run_requiring_reconciliation(account_key, mode)

    @staticmethod
    def _intent_matches(row: OrderIntentRecord, intent: OrderIntent) -> bool:
        return (
            row.run_id == intent.run_id
            and row.decision_key == intent.decision_key
            and row.sleeve == intent.sleeve
            and row.symbol == intent.symbol
            and row.side == intent.side.value
            and (
                None if row.requested_quantity is None else Decimal(row.requested_quantity)
            )
            == intent.quantity
            and (
                None if row.requested_notional is None else Decimal(row.requested_notional)
            )
            == intent.notional
            and Decimal(row.target_weight) == intent.target_weight
            and Decimal(row.arrival_bid) == intent.arrival_price
        )

    @staticmethod
    def _to_run_record(row: ExecutionRun) -> RunRecord:
        known_status = {status.value: status for status in RunStatus}
        return RunRecord(
            run_id=row.run_id,
            decision_key=row.decision_key,
            strategy_id=row.strategy_id,
            strategy_version=row.strategy_version,
            account_key=row.account_key,
            mode=TradingMode(row.mode),
            purpose=RunPurpose(row.purpose),
            target_hash=row.target_hash,
            created_at=_utc(row.started_at),
            status=known_status.get(row.status),
            completed_at=None if row.completed_at is None else _utc(row.completed_at),
            failure_reason=row.failure_reason or "",
            metadata=dict(row.metadata_json or {}),
        )

    @staticmethod
    def _to_order_intent(row: OrderIntentRecord) -> OrderIntent:
        return OrderIntent(
            client_order_id=row.client_order_id,
            run_id=row.run_id,
            decision_key=row.decision_key,
            sleeve=row.sleeve,
            symbol=row.symbol,
            side=OrderSide(row.side),
            target_weight=Decimal(row.target_weight),
            arrival_price=Decimal(row.arrival_bid),
            created_at=_utc(row.created_at),
            quantity=None if row.requested_quantity is None else Decimal(row.requested_quantity),
            notional=None if row.requested_notional is None else Decimal(row.requested_notional),
        )


__all__ = ["SQLAlchemyLedger", "V3SchemaMissing"]
