"""Persistence boundary and a deterministic in-memory v3 implementation.

``Ledger`` is intentionally storage-agnostic.  A SQLAlchemy/PostgreSQL adapter
can implement the same atomic methods without leaking ORM objects into the
coordinator.  The in-memory implementation is used by unit tests and shadow
research only; paper settings independently require a PostgreSQL URL.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import (
    OrderIntent,
    OrderState,
    RunPurpose,
    RunStatus,
    TradingMode,
)
from .lifecycle import validate_order_transition


class LeaseUnavailable(RuntimeError):
    pass


class ConflictingIntent(RuntimeError):
    pass


@dataclass(slots=True)
class RunRecord:
    run_id: str
    decision_key: str
    strategy_id: str
    strategy_version: str
    account_key: str
    mode: TradingMode
    purpose: RunPurpose
    target_hash: str
    created_at: datetime
    status: RunStatus | None = None
    completed_at: datetime | None = None
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    event_id: str
    client_order_id: str
    state: OrderState
    observed_at: datetime
    broker_order_id: str = ""
    filled_quantity: Decimal = Decimal("0")
    filled_average_price: Decimal | None = None
    slippage_bps: Decimal | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Ledger(Protocol):
    """Atomic operations required from an in-memory or SQLAlchemy ledger."""

    @property
    def paper_durable_truth(self) -> bool: ...

    def acquire_lease(self, account_key: str, mode: TradingMode, owner: str) -> bool: ...

    def release_lease(self, account_key: str, mode: TradingMode, owner: str) -> None: ...

    def renew_lease(self, account_key: str, mode: TradingMode, owner: str) -> bool: ...

    def get_run_by_decision_key(self, decision_key: str) -> RunRecord | None: ...

    def create_run(self, record: RunRecord) -> RunRecord: ...

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        at: datetime,
        *,
        failure_reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord: ...

    def add_intents(self, intents: tuple[OrderIntent, ...]) -> None: ...

    def append_order_event(self, event: OrderEvent) -> None: ...

    def current_order_state(self, client_order_id: str) -> OrderState: ...

    def events_for(self, client_order_id: str) -> tuple[OrderEvent, ...]: ...

    def get_intent(self, client_order_id: str) -> OrderIntent | None: ...

    def intents_for_run(self, run_id: str) -> tuple[OrderIntent, ...]: ...

    def latest_run_requiring_reconciliation(
        self, account_key: str, mode: TradingMode
    ) -> RunRecord | None: ...

    def oldest_run_requiring_reconciliation(
        self, account_key: str, mode: TradingMode
    ) -> RunRecord | None: ...


class InMemoryLedger:
    """Thread-safe reference semantics for the durable v3 ledger contract."""

    paper_durable_truth = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._leases: dict[tuple[str, TradingMode], str] = {}
        self._runs: dict[str, RunRecord] = {}
        self._run_by_decision: dict[str, str] = {}
        self._intents: dict[str, OrderIntent] = {}
        self._events: dict[str, list[OrderEvent]] = {}

    def acquire_lease(self, account_key: str, mode: TradingMode, owner: str) -> bool:
        key = (account_key, mode)
        with self._lock:
            current = self._leases.get(key)
            if current is None or current == owner:
                self._leases[key] = owner
                return True
            return False

    def release_lease(self, account_key: str, mode: TradingMode, owner: str) -> None:
        key = (account_key, mode)
        with self._lock:
            if self._leases.get(key) == owner:
                del self._leases[key]

    def renew_lease(self, account_key: str, mode: TradingMode, owner: str) -> bool:
        with self._lock:
            return self._leases.get((account_key, mode)) == owner

    def get_run_by_decision_key(self, decision_key: str) -> RunRecord | None:
        with self._lock:
            run_id = self._run_by_decision.get(decision_key)
            return None if run_id is None else self._runs[run_id]

    def create_run(self, record: RunRecord) -> RunRecord:
        with self._lock:
            existing_id = self._run_by_decision.get(record.decision_key)
            if existing_id is not None:
                return self._runs[existing_id]
            if record.run_id in self._runs:
                raise ValueError(f"duplicate run id {record.run_id}")
            self._runs[record.run_id] = record
            self._run_by_decision[record.decision_key] = record.run_id
            return record

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        at: datetime,
        *,
        failure_reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        with self._lock:
            record = self._runs[run_id]
            record.status = status
            record.completed_at = at
            record.failure_reason = failure_reason
            if metadata:
                record.metadata.update(metadata)
            return record

    def add_intents(self, intents: tuple[OrderIntent, ...]) -> None:
        with self._lock:
            for intent in intents:
                existing = self._intents.get(intent.client_order_id)
                if existing is not None and existing != intent:
                    raise ConflictingIntent(
                        f"client order id {intent.client_order_id} maps to conflicting intents"
                    )
            for intent in intents:
                self._intents.setdefault(intent.client_order_id, intent)
                self._events.setdefault(intent.client_order_id, [])

    def append_order_event(self, event: OrderEvent) -> None:
        with self._lock:
            if event.client_order_id not in self._intents:
                raise KeyError(f"unknown order intent {event.client_order_id}")
            previous = self.current_order_state(event.client_order_id)
            validate_order_transition(previous, event.state)
            self._events[event.client_order_id].append(event)

    def current_order_state(self, client_order_id: str) -> OrderState:
        with self._lock:
            if client_order_id not in self._intents:
                raise KeyError(f"unknown order intent {client_order_id}")
            events = self._events[client_order_id]
            return events[-1].state if events else OrderState.INTENT

    def events_for(self, client_order_id: str) -> tuple[OrderEvent, ...]:
        with self._lock:
            return tuple(self._events.get(client_order_id, ()))

    def get_intent(self, client_order_id: str) -> OrderIntent | None:
        with self._lock:
            return self._intents.get(client_order_id)

    def intents_for_run(self, run_id: str) -> tuple[OrderIntent, ...]:
        with self._lock:
            return tuple(i for i in self._intents.values() if i.run_id == run_id)

    def latest_run_requiring_reconciliation(
        self, account_key: str, mode: TradingMode
    ) -> RunRecord | None:
        with self._lock:
            candidates = [
                run
                for run in self._runs.values()
                if run.account_key == account_key
                and run.mode is mode
                and run.purpose is RunPurpose.REBALANCE
                and self._requires_reconciliation(run)
            ]
            return max(candidates, key=lambda run: run.created_at, default=None)

    def oldest_run_requiring_reconciliation(
        self, account_key: str, mode: TradingMode
    ) -> RunRecord | None:
        with self._lock:
            candidates = [
                run
                for run in self._runs.values()
                if run.account_key == account_key
                and run.mode is mode
                and run.purpose is RunPurpose.REBALANCE
                and self._requires_reconciliation(run)
            ]
            return min(candidates, key=lambda run: run.created_at, default=None)

    def _requires_reconciliation(self, run: RunRecord) -> bool:
        intents = self.intents_for_run(run.run_id)
        if not intents:
            return False
        states = tuple(
            self.current_order_state(intent.client_order_id) for intent in intents
        )
        if any(
            state
            in {
                OrderState.INTENT,
                OrderState.ACCEPTED,
                OrderState.PARTIALLY_FILLED,
                OrderState.UNKNOWN,
            }
            for state in states
        ):
            return True
        # A process can fail after every fill is durable but before final NAV,
        # positions and performance truth commit.  Such a run still needs the
        # completion phase, but a terminal reject/cancel/expiry does not.
        return run.status is not RunStatus.COMPLETED and all(
            state is OrderState.FILLED for state in states
        )

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())
