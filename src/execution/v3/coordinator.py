"""Fail-closed execution orchestration for shadow and Alpaca paper modes."""

from __future__ import annotations

import time as time_module
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

from .contracts import (
    AccountSnapshot,
    AssetSnapshot,
    BrokerOrderSnapshot,
    CalendarSession,
    OpenOrderSnapshot,
    OrderIntent,
    OrderSide,
    OrderState,
    PortfolioPlan,
    PositionSnapshot,
    QuoteSnapshot,
    RunPurpose,
    RunResult,
    RunStatus,
    RuntimeSettings,
    TradingMode,
)
from .gateway import (
    AlpacaGateway,
    BrokerReadError,
    BrokerSubmissionError,
    BrokerUncertainOutcome,
)
from .ids import (
    build_client_order_id,
    build_decision_key,
    build_operational_key,
    build_risk_decision_key,
    build_target_hash,
)
from .ledger import InMemoryLedger, Ledger, OrderEvent, RunRecord
from .shadow import ShadowAccount, ShadowExecutor


NEW_YORK = ZoneInfo("America/New_York")
EXECUTION_WINDOW_START = time(10, 5)
EXECUTION_WINDOW_END = time(11, 30)


class SafetyBlock(RuntimeError):
    pass


class ReconciliationRequired(RuntimeError):
    pass


class _LeaseLostBeforeSubmission(SafetyBlock):
    """The lease check failed before the named broker POST was attempted."""

    def __init__(self, client_order_id: str) -> None:
        self.client_order_id = client_order_id
        super().__init__(
            "execution lease was lost before broker submission; "
            f"{client_order_id} is eligible for confirmed-absence recovery"
        )


@dataclass(frozen=True, slots=True)
class PaperPreflight:
    account: AccountSnapshot
    positions: tuple[PositionSnapshot, ...]
    open_orders: tuple[OpenOrderSnapshot, ...]
    sessions: tuple[CalendarSession, ...]
    assets: Mapping[str, AssetSnapshot]
    quotes: Mapping[str, QuoteSnapshot]
    drawdown: Decimal


@dataclass(frozen=True, slots=True)
class PaperCompletionSnapshot:
    run_id: str
    decision_key: str
    target_hash: str
    account: AccountSnapshot
    positions: tuple[PositionSnapshot, ...]
    intents: tuple[OrderIntent, ...]
    events: tuple[OrderEvent, ...]
    observed_at: datetime
    target_weights: Mapping[str, Decimal] = field(default_factory=dict)
    plan_metadata: Mapping[str, object] = field(default_factory=dict)
    drawdown: Decimal = Decimal("0")
    max_position_drift: Decimal = Decimal("0")
    invested_weight_drift: Decimal = Decimal("0")
    exposure_breach: bool = False


class PaperCompletionRecorder(Protocol):
    def record_paper_completion(self, snapshot: PaperCompletionSnapshot) -> None: ...


class PaperRiskStateProvider(Protocol):
    """Durable peak-NAV source used to recompute paper drawdown from fresh equity."""

    def current_peak(self, account_key: str) -> Decimal | None: ...

    def is_de_risked(self, account_key: str) -> bool: ...

    def de_risked_since(self, account_key: str) -> datetime | None: ...


@dataclass(frozen=True, slots=True)
class PaperPlanningContext:
    account: AccountSnapshot
    positions: tuple[PositionSnapshot, ...]
    open_orders: tuple[OpenOrderSnapshot, ...]
    fresh_drawdown: Decimal
    prior_de_risked: bool
    de_risked_since: datetime | None
    now: datetime

    @property
    def current_holdings(self) -> frozenset[str]:
        return frozenset(
            position.symbol for position in self.positions if position.quantity > 0
        )


class PaperPlanFactory(Protocol):
    """Construct a deterministic target from fresh, lease-protected broker truth."""

    def construct(self, context: PaperPlanningContext) -> PortfolioPlan: ...


class ExecutionCoordinator:
    """One-shot coordinator shared by scheduled, dispatch, and manual callers.

    ``gateway`` may be omitted for shadow mode.  No shadow branch reads or
    submits through it, which makes the zero-broker-call invariant structural.
    """

    def __init__(
        self,
        settings: RuntimeSettings,
        ledger: Ledger,
        *,
        gateway: AlpacaGateway | None = None,
        shadow_executor: ShadowExecutor | None = None,
        paper_completion_recorder: PaperCompletionRecorder | None = None,
        paper_risk_state_provider: PaperRiskStateProvider | None = None,
        poll_interval_seconds: float = 5.0,
        poll_wait: Callable[[float], bool] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("order poll interval must be positive")
        self.settings = settings
        self.ledger = ledger
        self.gateway = gateway
        self.paper_completion_recorder = paper_completion_recorder
        self.paper_risk_state_provider = paper_risk_state_provider
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.poll_wait = poll_wait or self._default_poll_wait
        self.monotonic_clock = monotonic_clock or time_module.monotonic
        self.shadow_executor = shadow_executor or ShadowExecutor(
            min_trade_notional=settings.min_trade_notional,
            min_drift_fraction=settings.min_drift_fraction,
        )

    @staticmethod
    def _default_poll_wait(seconds: float) -> bool:
        time_module.sleep(seconds)
        return True

    def run(
        self,
        *,
        plan: PortfolioPlan | None = None,
        paper_plan_factory: PaperPlanFactory | None = None,
        now: datetime,
        trigger: str = "manual",
        shadow_account: ShadowAccount | None = None,
        shadow_quotes: Mapping[str, QuoteSnapshot] | None = None,
    ) -> RunResult:
        if plan is not None and paper_plan_factory is not None:
            raise ValueError("provide either plan or paper_plan_factory, not both")
        if (
            self.settings.mode is TradingMode.PAPER
            and self.settings.purpose is RunPurpose.REBALANCE
            and paper_plan_factory is None
        ):
            if now.tzinfo is None:
                raise ValueError("coordinator timestamp must be timezone-aware")
            local_now = now.astimezone(NEW_YORK)
            decision_key = build_decision_key(
                self.settings.strategy_id,
                self.settings.strategy_version,
                self.settings.account_key,
                self.settings.mode,
                local_now,
            )
            return self._record_paper_planning_failure(
                decision_key=decision_key,
                now=now.astimezone(UTC),
                trigger=trigger,
                message=(
                    "paper rebalance requires PaperPlanFactory so targets are "
                    "constructed from fresh lease-protected broker truth"
                ),
            )
        if paper_plan_factory is not None:
            return self._run_with_paper_plan_factory(
                paper_plan_factory, now=now, trigger=trigger
            )
        return self._run_with_plan(
            plan=plan,
            now=now,
            trigger=trigger,
            shadow_account=shadow_account,
            shadow_quotes=shadow_quotes,
        )

    def _run_with_plan(
        self,
        *,
        plan: PortfolioPlan | None,
        now: datetime,
        trigger: str,
        shadow_account: ShadowAccount | None,
        shadow_quotes: Mapping[str, QuoteSnapshot] | None,
        _lease_owner: str | None = None,
        _prior_de_risked: bool = False,
        _planning_context: PaperPlanningContext | None = None,
    ) -> RunResult:
        """Plan and execute one attributable v3 decision.

        Paper failures are converted to explicit statuses and exit codes.  The
        method never changes ``mode`` as a fallback behavior.
        """

        if now.tzinfo is None:
            raise ValueError("coordinator timestamp must be timezone-aware")
        now = now.astimezone(UTC)
        if self.settings.purpose is RunPurpose.RECONCILE:
            return self.reconcile(now=now, trigger=trigger)
        if self.settings.purpose is RunPurpose.REBALANCE and plan is None:
            raise ValueError("rebalance purpose requires a portfolio plan")
        if plan is not None:
            self._validate_plan_identity(plan)
        decision_time = now.astimezone(NEW_YORK)
        if self.settings.purpose is RunPurpose.REBALANCE:
            assert plan is not None
            if self._is_drawdown_kill_plan(plan):
                decision_key = build_risk_decision_key(
                    self.settings.strategy_id,
                    self.settings.strategy_version,
                    self.settings.account_key,
                    self.settings.mode,
                    decision_time,
                )
            else:
                decision_key = build_decision_key(
                    self.settings.strategy_id,
                    self.settings.strategy_version,
                    self.settings.account_key,
                    self.settings.mode,
                    decision_time,
                )
            target_hash = build_target_hash(plan.target_weights)
        else:
            decision_key = build_operational_key(
                self.settings.strategy_id,
                self.settings.strategy_version,
                self.settings.account_key,
                self.settings.mode,
                self.settings.purpose,
                decision_time,
            )
            # Operational probes cannot freeze or conflict with strategy targets.
            target_hash = build_target_hash({})

        existing = self.ledger.get_run_by_decision_key(decision_key)
        recoverable_existing_intents: tuple[OrderIntent, ...] = ()
        if existing is not None:
            if existing.target_hash != target_hash:
                return RunResult(
                    run_id=existing.run_id,
                    status=RunStatus.BLOCKED,
                    exit_code=2,
                    message="a different target is already frozen for this monthly decision",
                    decision_key=decision_key,
                    target_hash=target_hash,
                )
            existing_intents = self.ledger.intents_for_run(existing.run_id)
            states = {
                intent.client_order_id: self.ledger.current_order_state(
                    intent.client_order_id
                )
                for intent in existing_intents
            }
            all_filled = bool(existing_intents) and all(
                state is OrderState.FILLED for state in states.values()
            )
            if existing.status is RunStatus.COMPLETED and (
                not existing_intents or all_filled
            ):
                return RunResult(
                    run_id=existing.run_id,
                    status=RunStatus.SKIPPED_NOT_DUE,
                    exit_code=0,
                    message="monthly decision already completed; no duplicate execution",
                    decision_key=decision_key,
                    target_hash=target_hash,
                    order_client_ids=tuple(
                        intent.client_order_id for intent in existing_intents
                    ),
                )
            safe_retry_ids = self._safe_retry_ids(existing)
            recoverable_existing_intents = tuple(
                intent
                for intent in existing_intents
                if states[intent.client_order_id] is OrderState.INTENT
                and intent.client_order_id in safe_retry_ids
            )
            has_live_or_ambiguous = any(
                state
                in {
                    OrderState.ACCEPTED,
                    OrderState.PARTIALLY_FILLED,
                    OrderState.UNKNOWN,
                }
                or (
                    state is OrderState.INTENT
                    and client_order_id not in safe_retry_ids
                )
                for client_order_id, state in states.items()
            )
            has_terminal_failure = any(
                state
                in {
                    OrderState.REJECTED,
                    OrderState.CANCELED,
                    OrderState.EXPIRED,
                }
                for state in states.values()
            )
            if existing_intents and has_terminal_failure:
                return RunResult(
                    run_id=existing.run_id,
                    status=RunStatus.BLOCKED,
                    exit_code=2,
                    message=(
                        "the frozen monthly batch ended with a terminal order failure; "
                        "automatic resubmission is prohibited"
                    ),
                    decision_key=decision_key,
                    target_hash=target_hash,
                    order_client_ids=tuple(states),
                )
            if existing_intents and (has_live_or_ambiguous or all_filled):
                self.ledger.update_run(
                    existing.run_id,
                    RunStatus.RECONCILIATION_REQUIRED,
                    now,
                    metadata={"recovery_reason": "persisted intents require reconciliation"},
                )
                return RunResult(
                    run_id=existing.run_id,
                    status=RunStatus.RECONCILIATION_REQUIRED,
                    exit_code=3,
                    message="the prior attempt must be reconciled before another execution",
                    decision_key=decision_key,
                    target_hash=target_hash,
                )

        owner = _lease_owner or str(uuid.uuid4())
        lease_ready = (
            self.ledger.renew_lease(
                self.settings.account_key, self.settings.mode, owner
            )
            if _lease_owner is not None
            else self.ledger.acquire_lease(
                self.settings.account_key, self.settings.mode, owner
            )
        )
        if not lease_ready:
            return RunResult(
                run_id=existing.run_id if existing else "",
                status=RunStatus.BLOCKED,
                exit_code=2,
                message="another execution owns the account lease",
                decision_key=decision_key,
                target_hash=target_hash,
            )

        if (
            self.settings.mode is TradingMode.PAPER
            and self.settings.purpose is RunPurpose.REBALANCE
        ):
            unresolved = self.ledger.oldest_run_requiring_reconciliation(
                self.settings.account_key, self.settings.mode
            )
            if unresolved is not None and (
                existing is None or unresolved.run_id != existing.run_id
            ):
                self.ledger.release_lease(
                    self.settings.account_key, self.settings.mode, owner
                )
                return RunResult(
                    run_id=unresolved.run_id,
                    status=RunStatus.RECONCILIATION_REQUIRED,
                    exit_code=3,
                    message="an older account order must reconcile before a new monthly decision",
                    decision_key=unresolved.decision_key,
                    target_hash=unresolved.target_hash,
                )

        run_id = existing.run_id if existing else self._new_run_id()
        record = self.ledger.create_run(
            RunRecord(
                run_id=run_id,
                decision_key=decision_key,
                strategy_id=self.settings.strategy_id,
                strategy_version=self.settings.strategy_version,
                account_key=self.settings.account_key,
                mode=self.settings.mode,
                purpose=self.settings.purpose,
                target_hash=target_hash,
                created_at=now,
                metadata=self._run_metadata(trigger, plan),
            )
        )
        run_id = record.run_id
        try:
            if self.settings.purpose is not RunPurpose.REBALANCE:
                return self._finish(
                    run_id,
                    decision_key,
                    target_hash,
                    RunStatus.SKIPPED_NOT_DUE,
                    0,
                    f"{self.settings.purpose.value} does not submit rebalance orders",
                    now,
                )
            assert plan is not None
            self._enforce_drawdown_plan(plan)
            if self.settings.mode is TradingMode.SHADOW:
                if shadow_account is None or shadow_quotes is None:
                    raise SafetyBlock("shadow execution requires an isolated account and quotes")
                if shadow_account.account_key != self.settings.account_key:
                    raise SafetyBlock("shadow account key does not match runtime settings")
                result = self.shadow_executor.execute(
                    account=shadow_account,
                    plan=plan,
                    quotes=shadow_quotes,
                    decision_key=decision_key,
                    now=now,
                )
                return self._finish(
                    run_id,
                    decision_key,
                    target_hash,
                    RunStatus.COMPLETED,
                    0,
                    "shadow rebalance completed",
                    now,
                    order_client_ids=tuple(fill.client_order_id for fill in result.fills),
                    metadata={
                        "ending_nav": str(result.ending_nav),
                        "ending_cash": str(result.ending_cash),
                        "fill_count": len(result.fills),
                    },
                )

            gate_errors = self.settings.paper_gate_errors()
            if gate_errors:
                raise SafetyBlock("; ".join(gate_errors))
            if getattr(self.ledger, "paper_durable_truth", False) is not True:
                raise SafetyBlock(
                    "paper execution requires a verified durable PostgreSQL ledger"
                )
            self._validate_paper_plan(plan)
            if self.gateway is None:
                raise SafetyBlock("paper gateway is not configured")
            preflight = self._paper_preflight(
                plan,
                now,
                owner,
                prior_de_risked=_prior_de_risked,
                planning_context=_planning_context,
            )
            planned_intents = build_order_intents(
                run_id=run_id,
                decision_key=decision_key,
                plan=plan,
                account=preflight.account,
                positions=preflight.positions,
                open_orders=preflight.open_orders,
                assets=preflight.assets,
                quotes=preflight.quotes,
                settings=self.settings,
                now=now,
            )
            if recoverable_existing_intents:
                self._validate_recoverable_intents(
                    recoverable_existing_intents,
                    planned_intents,
                    now,
                )
                intents = tuple(
                    sorted(
                        self.ledger.intents_for_run(run_id),
                        key=lambda item: (item.side is OrderSide.BUY, item.symbol),
                    )
                )
            else:
                intents = planned_intents
            if self._is_drawdown_kill_plan(plan) and any(
                intent.side is OrderSide.BUY for intent in intents
            ):
                raise SafetyBlock(
                    "drawdown containment generated an exposure-increasing order"
                )
            # This append is the final operation before the first possible POST.
            self.ledger.add_intents(intents)
            self._submit_paper_intents(
                intents,
                preflight.account,
                now,
                owner,
                safe_retry_client_ids={
                    intent.client_order_id for intent in recoverable_existing_intents
                },
            )
            completion = self._capture_paper_completion(
                run_id,
                decision_key,
                target_hash,
                intents,
                now,
                owner,
                target_weights=plan.target_weights,
                plan_metadata=plan.metadata,
                drawdown=preflight.drawdown,
            )
            if completion.exposure_breach:
                raise ReconciliationRequired(
                    "post-fill exposure drift exceeds 50 basis points; "
                    "final broker truth was persisted for critical review"
                )
            return self._finish(
                run_id,
                decision_key,
                target_hash,
                RunStatus.COMPLETED,
                0,
                "paper rebalance completed and reconciled",
                now,
                order_client_ids=tuple(intent.client_order_id for intent in intents),
                metadata=self._completion_metadata(completion),
            )
        except ReconciliationRequired as exc:
            return self._finish(
                run_id,
                decision_key,
                target_hash,
                RunStatus.RECONCILIATION_REQUIRED,
                3,
                str(exc),
                now,
            )
        except _MarketClosed as exc:
            return self._finish(
                run_id,
                decision_key,
                target_hash,
                RunStatus.SKIPPED_MARKET_CLOSED,
                0,
                str(exc),
                now,
            )
        except (
            SafetyBlock,
            BrokerReadError,
            BrokerSubmissionError,
            BrokerUncertainOutcome,
            ValueError,
        ) as exc:
            if self._run_has_live_or_unknown_orders(run_id):
                return self._finish(
                    run_id,
                    decision_key,
                    target_hash,
                    RunStatus.RECONCILIATION_REQUIRED,
                    3,
                    f"{exc}; persisted broker order requires reconciliation",
                    now,
                )
            return self._finish(
                run_id,
                decision_key,
                target_hash,
                RunStatus.BLOCKED,
                2,
                str(exc),
                now,
            )
        except Exception as exc:  # pragma: no cover - last-resort status persistence
            if self._run_has_live_or_unknown_orders(run_id):
                return self._finish(
                    run_id,
                    decision_key,
                    target_hash,
                    RunStatus.RECONCILIATION_REQUIRED,
                    3,
                    "unexpected failure after broker acceptance; reconciliation required",
                    now,
                )
            return self._finish(
                run_id,
                decision_key,
                target_hash,
                RunStatus.FAILED,
                1,
                f"unexpected execution failure: {type(exc).__name__}",
                now,
            )
        finally:
            self.ledger.release_lease(
                self.settings.account_key, self.settings.mode, owner
            )

    def _run_with_paper_plan_factory(
        self,
        factory: PaperPlanFactory,
        *,
        now: datetime,
        trigger: str,
    ) -> RunResult:
        """Construct paper targets from broker truth while holding the account lease."""

        if now.tzinfo is None:
            raise ValueError("coordinator timestamp must be timezone-aware")
        now = now.astimezone(UTC)
        if (
            self.settings.mode is not TradingMode.PAPER
            or self.settings.purpose is not RunPurpose.REBALANCE
        ):
            raise ValueError(
                "paper_plan_factory is only valid for a paper rebalance"
            )
        fallback_key = build_decision_key(
            self.settings.strategy_id,
            self.settings.strategy_version,
            self.settings.account_key,
            self.settings.mode,
            now.astimezone(NEW_YORK),
        )
        gate_errors = self.settings.paper_gate_errors()
        if gate_errors:
            return self._record_paper_planning_failure(
                decision_key=fallback_key,
                now=now,
                trigger=trigger,
                message="; ".join(gate_errors),
            )
        if getattr(self.ledger, "paper_durable_truth", False) is not True:
            return self._record_paper_planning_failure(
                decision_key=fallback_key,
                now=now,
                trigger=trigger,
                message="paper execution requires a verified durable PostgreSQL ledger",
            )
        if self.gateway is None:
            return self._record_paper_planning_failure(
                decision_key=fallback_key,
                now=now,
                trigger=trigger,
                message="paper gateway is not configured",
            )
        if self.paper_risk_state_provider is None:
            return self._record_paper_planning_failure(
                decision_key=fallback_key,
                now=now,
                trigger=trigger,
                message="lease-safe paper planning requires durable risk state",
            )

        owner = str(uuid.uuid4())
        if not self.ledger.acquire_lease(
            self.settings.account_key, self.settings.mode, owner
        ):
            return self._record_paper_planning_failure(
                decision_key=fallback_key,
                now=now,
                trigger=trigger,
                message="another execution owns the account lease",
            )
        decision_key = fallback_key
        try:
            unresolved = self.ledger.oldest_run_requiring_reconciliation(
                self.settings.account_key, self.settings.mode
            )
            current_risk_key = build_risk_decision_key(
                self.settings.strategy_id,
                self.settings.strategy_version,
                self.settings.account_key,
                self.settings.mode,
                now.astimezone(NEW_YORK),
            )
            if unresolved is not None and not (
                unresolved.decision_key in {fallback_key, current_risk_key}
                and self._run_has_only_confirmed_retry_intents(unresolved)
            ):
                return self._reconciliation_block_for_run(unresolved, now)

            self._require_active_lease(owner)
            account = self.gateway.get_account()
            self._validate_account_snapshot(account, now)
            self._require_active_lease(owner)
            positions = self.gateway.get_positions()
            self._validate_long_positions(positions)
            fresh_drawdown = self._account_drawdown(
                account, fallback=Decimal("0")
            )
            try:
                raw_de_risked = self.paper_risk_state_provider.is_de_risked(
                    account.account_key
                )
            except Exception as exc:
                raise SafetyBlock("durable de-risk state read failed") from exc
            if not isinstance(raw_de_risked, bool):
                raise SafetyBlock("durable de-risk state must be a boolean")
            prior_de_risked = raw_de_risked
            de_risked_since: datetime | None = None
            if prior_de_risked:
                try:
                    de_risked_since = self.paper_risk_state_provider.de_risked_since(
                        account.account_key
                    )
                except Exception as exc:
                    raise SafetyBlock("durable de-risk activation read failed") from exc
                if de_risked_since is None or de_risked_since.tzinfo is None:
                    raise SafetyBlock("durable de-risk activation timestamp is unavailable")
            containment_active = (
                prior_de_risked or fresh_drawdown >= Decimal("0.15")
            )
            if containment_active:
                decision_key = build_risk_decision_key(
                    self.settings.strategy_id,
                    self.settings.strategy_version,
                    self.settings.account_key,
                    self.settings.mode,
                    now.astimezone(NEW_YORK),
                )
            if unresolved is not None and not (
                unresolved.decision_key == decision_key
                and self._run_has_only_confirmed_retry_intents(unresolved)
            ):
                return self._reconciliation_block_for_run(unresolved, now)
            self._require_active_lease(owner)
            open_orders = self.gateway.get_open_orders()
            self._check_open_orders(
                open_orders, containment_active, now, owner
            )
            context = PaperPlanningContext(
                account=account,
                positions=positions,
                open_orders=open_orders,
                fresh_drawdown=fresh_drawdown,
                prior_de_risked=prior_de_risked,
                de_risked_since=de_risked_since,
                now=now,
            )
            plan = factory.construct(context)
            self._validate_plan_identity(plan)
            if abs(plan.drawdown - fresh_drawdown) > Decimal("0.000001"):
                raise SafetyBlock(
                    "paper plan drawdown does not match fresh lease-protected equity"
                )
            de_risk_plan = self._is_drawdown_kill_plan(plan)
            if fresh_drawdown >= Decimal("0.15") and not de_risk_plan:
                raise SafetyBlock("fresh drawdown requires a de-risk plan")
            if prior_de_risked and not de_risk_plan:
                if (
                    fresh_drawdown >= Decimal("0.10")
                    or plan.metadata.get("satellite_reentry_approved") is not True
                ):
                    raise SafetyBlock(
                        "satellite re-entry requires drawdown below 10% and manual approval"
                    )
                assert de_risked_since is not None
                activation_month = de_risked_since.astimezone(NEW_YORK).strftime(
                    "%Y-%m"
                )
                current_month = now.astimezone(NEW_YORK).strftime("%Y-%m")
                if activation_month >= current_month:
                    raise SafetyBlock(
                        "satellite re-entry is not eligible until the next monthly rebalance"
                    )
            return self._run_with_plan(
                plan=plan,
                now=now,
                trigger=trigger,
                shadow_account=None,
                shadow_quotes=None,
                _lease_owner=owner,
                _prior_de_risked=prior_de_risked,
                _planning_context=context,
            )
        except ReconciliationRequired as exc:
            return self._record_paper_planning_failure(
                decision_key=decision_key,
                now=now,
                trigger=trigger,
                message=str(exc),
                status=RunStatus.RECONCILIATION_REQUIRED,
                exit_code=3,
            )
        except (SafetyBlock, BrokerReadError, ValueError) as exc:
            return self._record_paper_planning_failure(
                decision_key=decision_key,
                now=now,
                trigger=trigger,
                message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - last-resort planning guard
            return self._record_paper_planning_failure(
                decision_key=decision_key,
                now=now,
                trigger=trigger,
                message=f"unexpected lease-safe planning failure: {type(exc).__name__}",
                status=RunStatus.FAILED,
                exit_code=1,
            )
        finally:
            self.ledger.release_lease(
                self.settings.account_key, self.settings.mode, owner
            )

    def reconcile(self, *, now: datetime, trigger: str = "manual") -> RunResult:
        """Reconcile the oldest unresolved monthly paper run by client ID.

        The reconciliation operation has its own daily key.  It updates the
        referenced monthly rebalance only after every persisted intent reaches
        a known terminal state, and never retries an order POST blindly.
        """

        if now.tzinfo is None:
            raise ValueError("reconciliation timestamp must be timezone-aware")
        now = now.astimezone(UTC)
        local_now = now.astimezone(NEW_YORK)
        operation_key = build_operational_key(
            self.settings.strategy_id,
            self.settings.strategy_version,
            self.settings.account_key,
            self.settings.mode,
            RunPurpose.RECONCILE,
            local_now,
        )
        empty_hash = build_target_hash({})
        if self.settings.mode is not TradingMode.PAPER:
            return RunResult(
                run_id="",
                status=RunStatus.SKIPPED_NOT_DUE,
                exit_code=0,
                message="shadow mode has no broker orders to reconcile",
                decision_key=operation_key,
                target_hash=empty_hash,
            )

        target_run = self.ledger.oldest_run_requiring_reconciliation(
            self.settings.account_key, self.settings.mode
        )
        if target_run is None:
            existing_operation = self.ledger.get_run_by_decision_key(operation_key)
            if existing_operation is not None:
                return RunResult(
                    run_id=existing_operation.run_id,
                    status=RunStatus.SKIPPED_NOT_DUE,
                    exit_code=0,
                    message="no paper rebalance requires reconciliation",
                    decision_key=operation_key,
                    target_hash=empty_hash,
                )
            return RunResult(
                run_id="",
                status=RunStatus.SKIPPED_NOT_DUE,
                exit_code=0,
                message="no paper rebalance requires reconciliation",
                decision_key=operation_key,
                target_hash=empty_hash,
            )

        operation_key = f"{operation_key}|target|{target_run.run_id}"
        existing_operation = self.ledger.get_run_by_decision_key(operation_key)

        owner = str(uuid.uuid4())
        if not self.ledger.acquire_lease(
            self.settings.account_key, self.settings.mode, owner
        ):
            return RunResult(
                run_id=existing_operation.run_id if existing_operation else "",
                status=RunStatus.BLOCKED,
                exit_code=2,
                message="another execution owns the account lease",
                decision_key=operation_key,
                target_hash=target_run.target_hash,
            )
        operation_id = (
            existing_operation.run_id if existing_operation else self._new_run_id()
        )
        operation = self.ledger.create_run(
            RunRecord(
                run_id=operation_id,
                decision_key=operation_key,
                strategy_id=self.settings.strategy_id,
                strategy_version=self.settings.strategy_version,
                account_key=self.settings.account_key,
                mode=self.settings.mode,
                purpose=RunPurpose.RECONCILE,
                target_hash=target_run.target_hash,
                created_at=now,
                metadata={
                    "trigger": trigger,
                    "target_run_id": target_run.run_id,
                    "commit_sha": self.settings.commit_sha,
                },
            )
        )
        operation_id = operation.run_id
        try:
            gate_errors = self.settings.paper_reconciliation_gate_errors()
            if gate_errors:
                raise SafetyBlock("; ".join(gate_errors))
            if getattr(self.ledger, "paper_durable_truth", False) is not True:
                raise SafetyBlock(
                    "paper reconciliation requires a verified durable PostgreSQL ledger"
                )
            if self.gateway is None:
                raise SafetyBlock("paper gateway is not configured")
            self._require_active_lease(owner)
            reconciliation_account = self.gateway.get_account()
            self._validate_account_snapshot(reconciliation_account, now)
            reconciliation_drawdown = self._account_drawdown(
                reconciliation_account,
                fallback=self._run_drawdown(target_run),
            )
            reconciliation_containment = (
                reconciliation_drawdown >= Decimal("0.15")
                or self._run_is_de_risked(target_run)
            )
            if self.paper_risk_state_provider is not None:
                try:
                    raw_de_risked = self.paper_risk_state_provider.is_de_risked(
                        reconciliation_account.account_key
                    )
                except Exception as exc:
                    raise SafetyBlock("durable de-risk state read failed") from exc
                if not isinstance(raw_de_risked, bool):
                    raise SafetyBlock("durable de-risk state must be a boolean")
                reconciliation_containment = (
                    reconciliation_containment or raw_de_risked
                )
            intents = tuple(
                sorted(
                    self.ledger.intents_for_run(target_run.run_id),
                    key=lambda item: (item.side is OrderSide.BUY, item.symbol),
                )
            )
            if not intents:
                raise SafetyBlock("unresolved run contains no persisted order intents")

            initial_states = tuple(
                self.ledger.current_order_state(intent.client_order_id)
                for intent in intents
            )
            terminal_failure_indexes = tuple(
                index
                for index, state in enumerate(initial_states)
                if state
                in {
                    OrderState.REJECTED,
                    OrderState.CANCELED,
                    OrderState.EXPIRED,
                }
            )
            if terminal_failure_indexes:
                first_failure = min(terminal_failure_indexes)
                self._abort_unsubmitted_intents(
                    tuple(intents[first_failure + 1 :]),
                    now,
                    reason="reconciliation_terminalized_batch_tail",
                )

            confirmed_retry_ids = self._safe_retry_ids(target_run)
            observed_terminal_failure = False
            for intent in intents:
                state = self.ledger.current_order_state(intent.client_order_id)
                if state is OrderState.FILLED:
                    continue
                if state.is_terminal:
                    observed_terminal_failure = True
                    continue
                if state is OrderState.INTENT:
                    if intent.client_order_id in confirmed_retry_ids:
                        continue
                    # INTENT is ambiguous after process/DB failure: Alpaca may
                    # have accepted the POST before the first event committed.
                    # Always query the deterministic client ID and never POST
                    # such a recovered intent automatically.
                    self._require_active_lease(owner)
                    recovered = self.gateway.get_order_by_client_id(
                        intent.client_order_id
                    )
                    if recovered is None:
                        age_minutes = (
                            now - intent.created_at.astimezone(UTC)
                        ).total_seconds() / 60
                        if age_minutes < self.settings.unresolved_order_minutes:
                            raise ReconciliationRequired(
                                f"persisted intent {intent.client_order_id} is absent from "
                                "the broker but the 15-minute confirmation window has not elapsed"
                            )
                        self._mark_intents_safe_to_retry(
                            target_run.run_id,
                            (intent.client_order_id,),
                            now,
                            reason="broker_absence_confirmed_after_15_minutes",
                        )
                        confirmed_retry_ids.add(intent.client_order_id)
                        continue
                    self._append_snapshot(intent, recovered)
                    if recovered.state is OrderState.FILLED:
                        continue
                    if recovered.state.is_terminal:
                        raise SafetyBlock(
                            f"order {intent.client_order_id} ended as {recovered.state.value}"
                        )
                    self._cancel_buy_for_drawdown(
                        intent, recovered, reconciliation_containment, owner
                    )
                    raise ReconciliationRequired(
                        f"recovered order {intent.client_order_id} remains {recovered.state.value}"
                    )

                self._require_active_lease(owner)
                snapshot = self.gateway.get_order_by_client_id(intent.client_order_id)
                if snapshot is None:
                    if state is not OrderState.UNKNOWN:
                        self._append_event(intent, OrderState.UNKNOWN, now)
                    raise ReconciliationRequired(
                        f"order {intent.client_order_id} is still unknown"
                    )
                self._append_snapshot(intent, snapshot)
                if snapshot.state is OrderState.FILLED:
                    continue
                if snapshot.state.is_terminal:
                    raise SafetyBlock(
                        f"order {intent.client_order_id} ended as {snapshot.state.value}"
                    )
                self._cancel_buy_for_drawdown(
                    intent, snapshot, reconciliation_containment, owner
                )
                age_minutes = (now - intent.created_at.astimezone(UTC)).total_seconds() / 60
                if age_minutes >= self.settings.unresolved_order_minutes:
                    self._require_active_lease(owner)
                    self.gateway.cancel_order(snapshot.broker_order_id)
                    raise ReconciliationRequired(
                        f"order {intent.client_order_id} exceeded 15 minutes and was canceled"
                    )
                raise ReconciliationRequired(
                    f"order {intent.client_order_id} remains {snapshot.state.value}"
                )

            if any(
                self.ledger.current_order_state(intent.client_order_id)
                is OrderState.INTENT
                for intent in intents
            ):
                raise SafetyBlock(
                    "broker absence is confirmed for the remaining frozen intents; "
                    "a lease-safe rebalance retry may resume them"
                )
            if observed_terminal_failure:
                raise SafetyBlock(
                    "the frozen order batch contains a terminal reject/cancel/expiry; "
                    "all definitely-unsubmitted trailing intents were terminalized"
                )

            target_weights, plan_metadata, target_drawdown = (
                self._completion_context_from_run(target_run, intents)
            )
            completion = self._capture_paper_completion(
                target_run.run_id,
                target_run.decision_key,
                target_run.target_hash,
                intents,
                now,
                owner,
                target_weights=target_weights,
                plan_metadata=plan_metadata,
                drawdown=max(target_drawdown, reconciliation_drawdown),
            )
            if completion.exposure_breach:
                raise ReconciliationRequired(
                    "post-fill exposure drift still exceeds 50 basis points"
                )
            self.ledger.update_run(
                target_run.run_id,
                RunStatus.COMPLETED,
                now,
                metadata={
                    "reconciled_by": operation_id,
                    **self._completion_metadata(completion),
                },
            )
            return self._finish(
                operation_id,
                operation_key,
                target_run.target_hash,
                RunStatus.COMPLETED,
                0,
                "all paper orders are reconciled and filled",
                now,
                order_client_ids=tuple(intent.client_order_id for intent in intents),
                metadata={"target_run_id": target_run.run_id},
            )
        except (ReconciliationRequired, BrokerUncertainOutcome) as exc:
            self.ledger.update_run(
                target_run.run_id,
                RunStatus.RECONCILIATION_REQUIRED,
                now,
                metadata={"last_reconciliation_error": str(exc)},
            )
            return self._finish(
                operation_id,
                operation_key,
                target_run.target_hash,
                RunStatus.RECONCILIATION_REQUIRED,
                3,
                str(exc),
                now,
                metadata={"target_run_id": target_run.run_id},
            )
        except (SafetyBlock, BrokerReadError, BrokerSubmissionError, ValueError) as exc:
            if self._run_has_live_or_unknown_orders(target_run.run_id):
                self.ledger.update_run(
                    target_run.run_id,
                    RunStatus.RECONCILIATION_REQUIRED,
                    now,
                    metadata={"last_reconciliation_error": str(exc)},
                )
                return self._finish(
                    operation_id,
                    operation_key,
                    target_run.target_hash,
                    RunStatus.RECONCILIATION_REQUIRED,
                    3,
                    f"{exc}; persisted broker order still requires reconciliation",
                    now,
                    metadata={"target_run_id": target_run.run_id},
                )
            self.ledger.update_run(
                target_run.run_id,
                RunStatus.BLOCKED,
                now,
                failure_reason=str(exc),
            )
            return self._finish(
                operation_id,
                operation_key,
                target_run.target_hash,
                RunStatus.BLOCKED,
                2,
                str(exc),
                now,
                metadata={"target_run_id": target_run.run_id},
            )
        except Exception as exc:  # pragma: no cover - last-resort status persistence
            if self._run_has_live_or_unknown_orders(target_run.run_id):
                self.ledger.update_run(
                    target_run.run_id,
                    RunStatus.RECONCILIATION_REQUIRED,
                    now,
                    metadata={"last_reconciliation_error": type(exc).__name__},
                )
                return self._finish(
                    operation_id,
                    operation_key,
                    target_run.target_hash,
                    RunStatus.RECONCILIATION_REQUIRED,
                    3,
                    "unexpected cleanup failure; persisted broker order still requires reconciliation",
                    now,
                    metadata={"target_run_id": target_run.run_id},
                )
            return self._finish(
                operation_id,
                operation_key,
                target_run.target_hash,
                RunStatus.FAILED,
                1,
                f"unexpected reconciliation failure: {type(exc).__name__}",
                now,
                metadata={"target_run_id": target_run.run_id},
            )
        finally:
            self.ledger.release_lease(
                self.settings.account_key, self.settings.mode, owner
            )

    def _paper_preflight(
        self,
        plan: PortfolioPlan,
        now: datetime,
        lease_owner: str,
        *,
        prior_de_risked: bool = False,
        planning_context: PaperPlanningContext | None = None,
    ) -> PaperPreflight:
        assert self.gateway is not None
        if planning_context is None:
            account = self.gateway.get_account()
            self._validate_account_snapshot(account, now)
            positions = self.gateway.get_positions()
            self._validate_long_positions(positions)
            fresh_drawdown = self._account_drawdown(account, fallback=plan.drawdown)
        else:
            account = planning_context.account
            positions = planning_context.positions
            fresh_drawdown = planning_context.fresh_drawdown
            if planning_context.now != now:
                raise SafetyBlock("paper planning context timestamp changed before preflight")
            self._validate_account_snapshot(account, now)
            self._validate_long_positions(positions)
            if abs(plan.drawdown - fresh_drawdown) > Decimal("0.000001"):
                raise SafetyBlock(
                    "paper plan drawdown does not match its lease-protected context"
                )
        self._require_active_lease(lease_owner)
        open_orders = self.gateway.get_open_orders()
        de_risk_plan = self._is_drawdown_kill_plan(plan)
        self._check_open_orders(
            open_orders,
            de_risk_plan or prior_de_risked or fresh_drawdown >= Decimal("0.15"),
            now,
            lease_owner,
        )
        if fresh_drawdown >= Decimal("0.15") and not de_risk_plan:
            raise SafetyBlock(
                "fresh broker equity requires a drawdown-containment plan"
            )
        if prior_de_risked and not de_risk_plan:
            approved = plan.metadata.get("satellite_reentry_approved") is True
            if fresh_drawdown >= Decimal("0.10") or not approved:
                raise SafetyBlock(
                    "satellite re-entry requires drawdown below 10% and manual approval"
                )

        self._require_active_lease(lease_owner)
        clock = self.gateway.get_clock()
        if abs((now - clock.timestamp.astimezone(UTC)).total_seconds()) > 60:
            raise SafetyBlock("Alpaca clock snapshot is stale")
        if not clock.is_open:
            raise _MarketClosed("Alpaca reports that the market is closed")

        local_now = now.astimezone(NEW_YORK)
        month_start = date(local_now.year, local_now.month, 1)
        self._require_active_lease(lease_owner)
        sessions = self.gateway.get_calendar(month_start, local_now.date())
        current_session = next(
            (session for session in sessions if session.session_date == local_now.date()),
            None,
        )
        if current_session is None:
            raise _MarketClosed("current date is not an eligible NYSE session")
        if self._is_drawdown_kill_plan(plan):
            if not (
                current_session.open_at.astimezone(UTC)
                <= now
                <= current_session.close_at.astimezone(UTC)
            ):
                raise _MarketClosed(
                    "drawdown containment is only permitted during regular hours"
                )
        else:
            eligible = sorted(s.session_date for s in sessions)[:3]
            if local_now.date() not in eligible:
                raise SafetyBlock("rebalance is outside the first three NYSE sessions")
            if not (
                EXECUTION_WINDOW_START
                <= local_now.time().replace(tzinfo=None)
                <= EXECUTION_WINDOW_END
            ):
                raise SafetyBlock(
                    "rebalance is outside the 10:05-11:30 ET execution window"
                )

        symbols = sorted(set(plan.target_weights) | {p.symbol for p in positions})
        self._require_active_lease(lease_owner)
        assets = self.gateway.get_assets(symbols)
        self._require_active_lease(lease_owner)
        quotes = self.gateway.get_latest_quotes(symbols)
        if set(assets) != set(symbols):
            raise SafetyBlock("asset response is incomplete")
        if set(quotes) != set(symbols):
            raise SafetyBlock("quote response is incomplete")
        for symbol in symbols:
            asset = assets[symbol]
            quote = quotes[symbol]
            if asset.symbol != symbol or quote.symbol != symbol:
                raise SafetyBlock(f"broker market-data identity mismatch for {symbol}")
            if not asset.tradable:
                raise SafetyBlock(f"{symbol} is not tradable")
            if asset.asset_class.lower() != "us_equity":
                raise SafetyBlock(f"{symbol} is not a US equity")
            age = (now - quote.observed_at.astimezone(UTC)).total_seconds()
            if age < -5 or age > self.settings.quote_max_age_seconds:
                raise SafetyBlock(f"quote for {symbol} is stale or future-dated")
            if quote.adv_dollars_30d <= 0:
                raise SafetyBlock(f"ADV for {symbol} is unavailable")
            _ = quote.midpoint
        return PaperPreflight(
            account=account,
            positions=positions,
            open_orders=open_orders,
            sessions=sessions,
            assets=assets,
            quotes=quotes,
            drawdown=fresh_drawdown,
        )

    def _check_open_orders(
        self,
        orders: Sequence[OpenOrderSnapshot],
        containment_active: bool,
        now: datetime,
        lease_owner: str,
    ) -> None:
        for order in orders:
            if (
                not order.broker_order_id
                or not order.symbol
                or order.symbol != order.symbol.upper().strip()
                or not order.quantity.is_finite()
                or not order.filled_quantity.is_finite()
                or order.quantity < 0
                or order.filled_quantity < 0
                or order.filled_quantity > order.quantity
                or order.submitted_at.tzinfo is None
            ):
                raise SafetyBlock("Alpaca returned an invalid open-order snapshot")
            if order.submitted_at.astimezone(UTC) > now + timedelta(seconds=5):
                raise SafetyBlock("Alpaca returned a future-dated open order")
            if order.state.is_terminal:
                continue
            if not order.client_order_id.startswith("aq3-"):
                raise SafetyBlock(
                    f"manual or unattributed open order exists for {order.symbol}"
                )
            intent = self.ledger.get_intent(order.client_order_id)
            if intent is None:
                raise SafetyBlock(
                    f"unknown aq3-prefixed open order is unattributed: {order.client_order_id}"
                )
            source_run = self.ledger.get_run_by_decision_key(intent.decision_key)
            if (
                source_run is None
                or source_run.account_key != self.settings.account_key
                or source_run.mode is not self.settings.mode
                or source_run.strategy_id != self.settings.strategy_id
                or source_run.strategy_version != self.settings.strategy_version
                or intent.symbol != order.symbol
                or intent.side is not order.side
            ):
                raise SafetyBlock(
                    f"aq3 open order attribution does not match this runtime: {order.client_order_id}"
                )
            if containment_active and order.side is OrderSide.BUY:
                assert self.gateway is not None
                self._require_active_lease(lease_owner)
                try:
                    self.gateway.cancel_order(order.broker_order_id)
                except BrokerUncertainOutcome as exc:
                    raise ReconciliationRequired(
                        f"drawdown cancel outcome is unknown for {order.client_order_id}"
                    ) from exc
                raise ReconciliationRequired(
                    f"drawdown containment canceled open buy {order.client_order_id}"
                )
            age_minutes = (
                now - order.submitted_at.astimezone(UTC)
            ).total_seconds() / 60
            if age_minutes >= self.settings.unresolved_order_minutes:
                assert self.gateway is not None
                self._require_active_lease(lease_owner)
                try:
                    self.gateway.cancel_order(order.broker_order_id)
                except BrokerUncertainOutcome as exc:
                    raise ReconciliationRequired(
                        f"cancel outcome is unknown for {order.client_order_id}"
                    ) from exc
                raise ReconciliationRequired(
                    f"stale v3 order {order.client_order_id} was canceled and must reconcile"
                )
            raise ReconciliationRequired(
                f"v3 order {order.client_order_id} remains nonterminal"
            )

    def _submit_paper_intents(
        self,
        intents: tuple[OrderIntent, ...],
        account: AccountSnapshot,
        now: datetime,
        lease_owner: str,
        *,
        safe_retry_client_ids: set[str] | None = None,
    ) -> None:
        assert self.gateway is not None
        safe_retry_client_ids = safe_retry_client_ids or set()
        states = {
            intent.client_order_id: self.ledger.current_order_state(
                intent.client_order_id
            )
            for intent in intents
        }
        unsafe_existing = tuple(
            client_order_id
            for client_order_id, state in states.items()
            if state not in {OrderState.INTENT, OrderState.FILLED}
        )
        if unsafe_existing:
            raise ReconciliationRequired(
                "persisted orders have broker lifecycle state and cannot be submitted again"
            )
        pending = tuple(
            intent
            for intent in intents
            if states[intent.client_order_id] is OrderState.INTENT
        )
        sells = tuple(i for i in pending if i.side is OrderSide.SELL)
        buys = tuple(i for i in pending if i.side is OrderSide.BUY)

        try:
            for intent in sells:
                self._submit_and_reconcile(
                    intent,
                    now,
                    lease_owner,
                    confirm_absent_before_post=(
                        intent.client_order_id in safe_retry_client_ids
                    ),
                )

            if buys:
                try:
                    self._require_active_lease(lease_owner)
                except SafetyBlock as exc:
                    raise _LeaseLostBeforeSubmission(
                        buys[0].client_order_id
                    ) from exc
                refreshed = self.gateway.get_account()
                self._validate_account_snapshot(refreshed, now)
                available = refreshed.cash * (
                    Decimal("1") - self.settings.buying_power_buffer_fraction
                )
                required = sum(
                    (
                        intent.notional
                        if intent.notional is not None
                        else intent.quantity * intent.arrival_price
                    )
                    for intent in buys
                )
                if required > available:
                    raise SafetyBlock(
                        "aggregate buys exceed refreshed buying power with safety buffer"
                    )
            else:
                available = account.cash
            for intent in buys:
                amount = (
                    intent.notional
                    if intent.notional is not None
                    else intent.quantity * intent.arrival_price
                )
                if amount > available:
                    raise SafetyBlock(
                        f"buying power cannot cover {intent.client_order_id} with safety buffer"
                    )
                self._submit_and_reconcile(
                    intent,
                    now,
                    lease_owner,
                    confirm_absent_before_post=(
                        intent.client_order_id in safe_retry_client_ids
                    ),
                )
                available -= amount
        except _LeaseLostBeforeSubmission:
            self._mark_intents_safe_to_retry(
                intents[0].run_id,
                tuple(
                    intent.client_order_id
                    for intent in pending
                    if self.ledger.current_order_state(intent.client_order_id)
                    is OrderState.INTENT
                ),
                now,
                reason="lease_lost_before_submission",
            )
            raise
        except (BrokerSubmissionError, SafetyBlock):
            if not self._batch_has_live_or_unknown(intents):
                self._abort_unsubmitted_intents(
                    intents,
                    now,
                    reason="batch_aborted_after_terminal_failure",
                )
            raise

    def _validate_account_snapshot(
        self, account: AccountSnapshot, now: datetime
    ) -> None:
        if account.observed_at.tzinfo is None:
            raise SafetyBlock("Alpaca account snapshot is timezone-naive")
        if account.account_key != self.settings.account_key:
            raise SafetyBlock(
                "Alpaca credentials resolved to an unexpected account key"
            )
        if account.status not in {"active", "accountstatus.active"}:
            raise SafetyBlock(f"Alpaca account is not active: {account.status}")
        if not all(
            value.is_finite()
            for value in (account.equity, account.cash, account.buying_power)
        ):
            raise SafetyBlock("Alpaca returned non-finite account balances")
        if account.equity <= 0 or account.cash < 0 or account.buying_power < 0:
            raise SafetyBlock("Alpaca returned invalid account balances")
        if abs((now - account.observed_at.astimezone(UTC)).total_seconds()) > 60:
            raise SafetyBlock("Alpaca account snapshot is stale")

    @staticmethod
    def _validate_long_positions(positions: Sequence[PositionSnapshot]) -> None:
        seen: set[str] = set()
        for position in positions:
            if (
                not position.symbol
                or position.symbol != position.symbol.upper().strip()
                or position.symbol in seen
            ):
                raise SafetyBlock("broker positions contain an invalid or duplicate symbol")
            seen.add(position.symbol)
            if position.asset_class.lower() != "us_equity":
                raise SafetyBlock(
                    f"non-US-equity broker position is forbidden: {position.symbol}"
                )
            if (
                not position.quantity.is_finite()
                or not position.market_price.is_finite()
                or position.quantity <= 0
                or position.market_price <= 0
            ):
                raise SafetyBlock(
                    f"invalid or short broker position is forbidden: {position.symbol}"
                )

    @staticmethod
    def _safe_retry_ids(run: RunRecord) -> set[str]:
        raw = run.metadata.get("safe_retry_client_ids", ())
        if not isinstance(raw, (tuple, list, set, frozenset)):
            return set()
        return {
            str(client_order_id)
            for client_order_id in raw
            if isinstance(client_order_id, str) and client_order_id
        }

    def _run_has_only_confirmed_retry_intents(self, run: RunRecord) -> bool:
        intents = self.ledger.intents_for_run(run.run_id)
        if not intents:
            return False
        safe_ids = self._safe_retry_ids(run)
        has_retry = False
        for intent in intents:
            state = self.ledger.current_order_state(intent.client_order_id)
            if state is OrderState.FILLED:
                continue
            if state is OrderState.INTENT and intent.client_order_id in safe_ids:
                has_retry = True
                continue
            return False
        return has_retry

    def _mark_intents_safe_to_retry(
        self,
        run_id: str,
        client_order_ids: tuple[str, ...],
        now: datetime,
        *,
        reason: str,
    ) -> None:
        if not client_order_ids:
            return
        first_intent = self.ledger.get_intent(client_order_ids[0])
        run = (
            None
            if first_intent is None
            else self.ledger.get_run_by_decision_key(first_intent.decision_key)
        )
        if run is None:
            raise SafetyBlock("cannot attribute retry-safe intents to their execution run")
        if run.run_id != run_id:
            raise SafetyBlock("retry-safe intent belongs to a different execution run")
        if any(
            (intent := self.ledger.get_intent(client_order_id)) is None
            or intent.run_id != run_id
            for client_order_id in client_order_ids
        ):
            raise SafetyBlock("retry-safe intent attribution is inconsistent")
        safe_ids = self._safe_retry_ids(run)
        safe_ids.update(client_order_ids)
        self.ledger.update_run(
            run_id,
            RunStatus.BLOCKED,
            now,
            metadata={
                "safe_retry_client_ids": sorted(safe_ids),
                "safe_retry_reason": reason,
                "safe_retry_confirmed_at": now.isoformat(),
            },
        )

    def _abort_unsubmitted_intents(
        self,
        intents: tuple[OrderIntent, ...],
        now: datetime,
        *,
        reason: str,
    ) -> None:
        for intent in intents:
            if self.ledger.current_order_state(intent.client_order_id) is not OrderState.INTENT:
                continue
            self.ledger.append_order_event(
                OrderEvent(
                    event_id=str(uuid.uuid4()),
                    client_order_id=intent.client_order_id,
                    state=OrderState.CANCELED,
                    observed_at=now,
                    details={"local_terminal_reason": reason, "broker_posted": False},
                )
            )

    def _batch_has_live_or_unknown(
        self, intents: tuple[OrderIntent, ...]
    ) -> bool:
        return any(
            self.ledger.current_order_state(intent.client_order_id)
            in {
                OrderState.ACCEPTED,
                OrderState.PARTIALLY_FILLED,
                OrderState.UNKNOWN,
            }
            for intent in intents
        )

    def _validate_recoverable_intents(
        self,
        frozen: tuple[OrderIntent, ...],
        planned: tuple[OrderIntent, ...],
        now: datetime,
    ) -> None:
        frozen_by_id = {intent.client_order_id: intent for intent in frozen}
        planned_by_id = {intent.client_order_id: intent for intent in planned}
        if set(frozen_by_id) != set(planned_by_id):
            raise SafetyBlock(
                "fresh broker truth no longer matches the frozen retry-safe order batch"
            )
        for client_order_id, intent in frozen_by_id.items():
            candidate = planned_by_id[client_order_id]
            if (
                intent.symbol != candidate.symbol
                or intent.side is not candidate.side
                or intent.quantity != candidate.quantity
                or intent.notional != candidate.notional
                or intent.target_weight != candidate.target_weight
            ):
                raise SafetyBlock("retry-safe frozen order amount changed after recomputation")
            age = (now - intent.created_at.astimezone(UTC)).total_seconds()
            if age < 0 or age > self.settings.quote_max_age_seconds:
                self._abort_unsubmitted_intents(
                    frozen,
                    now,
                    reason="retry_safe_intent_arrival_quote_expired",
                )
                raise SafetyBlock(
                    "retry-safe frozen intent was terminalized because its arrival quote expired"
                )

    def _run_has_live_or_unknown_orders(self, run_id: str) -> bool:
        intents = self.ledger.intents_for_run(run_id)
        if not intents:
            return False
        states = tuple(
            self.ledger.current_order_state(intent.client_order_id)
            for intent in intents
        )
        if any(
            state
            in {
                OrderState.ACCEPTED,
                OrderState.PARTIALLY_FILLED,
                OrderState.UNKNOWN,
            }
            for state in states
        ):
            return True
        # Final account/position/recorder failures after fills still require
        # the idempotent completion phase.  A trailing INTENT becomes
        # ambiguous once another order in the frozen batch crossed the broker
        # boundary and must be looked up, never blindly submitted.
        return all(state is OrderState.FILLED for state in states) or (
            OrderState.INTENT in states
            and any(state is not OrderState.INTENT for state in states)
        )

    def _validate_paper_plan(self, plan: PortfolioPlan) -> None:
        """Bind paper orders to the validated v3 constructor and hard limits."""

        metadata = plan.metadata
        required = {
            "plan_origin",
            "config_sha256",
            "source_target_sha256",
            "source_weight_hash",
            "promotable",
            "benchmark_symbol",
            "cash_weight",
            "portfolio_beta",
            "tracking_error",
            "max_active_sector_deviation",
        }
        missing = required.difference(metadata)
        if missing:
            raise SafetyBlock(
                "paper plan is missing constructor provenance: "
                + ", ".join(sorted(missing))
            )
        if metadata["plan_origin"] != "src.v3.portfolio.PortfolioConstructor":
            raise SafetyBlock("paper plan did not originate from the v3 portfolio constructor")
        config_sha = str(metadata["config_sha256"]).lower()
        if config_sha != self.settings.strategy_config_sha256.lower():
            raise SafetyBlock("paper plan config SHA does not match runtime settings")
        if not re_full_sha256(str(metadata["source_target_sha256"])):
            raise SafetyBlock("paper plan source target SHA is invalid")
        if str(metadata["source_weight_hash"]) != build_target_hash(plan.target_weights):
            raise SafetyBlock("paper plan weights changed after constructor validation")
        if str(metadata["benchmark_symbol"]).upper() != "SPY":
            raise SafetyBlock("paper plan benchmark provenance is not SPY")

        cash_weight = Decimal(str(metadata["cash_weight"]))
        beta = Decimal(str(metadata["portfolio_beta"]))
        tracking_error = Decimal(str(metadata["tracking_error"]))
        sector_deviation = Decimal(str(metadata["max_active_sector_deviation"]))
        if not all(
            value.is_finite()
            for value in (cash_weight, beta, tracking_error, sector_deviation)
        ):
            raise SafetyBlock("paper plan risk metadata must be finite")
        invested = sum(plan.target_weights.values(), Decimal("0"))
        if abs(invested + cash_weight - Decimal("1")) > Decimal("0.000000001"):
            raise SafetyBlock("paper plan invested and cash weights do not sum to one")
        core = plan.target_weights.get("SPY", Decimal("0"))
        direct = {
            symbol: weight
            for symbol, weight in plan.target_weights.items()
            if symbol != "SPY" and weight > 0
        }
        satellite = sum(direct.values(), Decimal("0"))
        if len(direct) > 30:
            raise SafetyBlock("paper plan exceeds 30 direct satellite holdings")
        if any(weight > Decimal("0.02") for weight in direct.values()):
            raise SafetyBlock("paper plan exceeds the 2% direct issuer cap")
        if satellite > Decimal("0.30") or core > Decimal("0.99"):
            raise SafetyBlock("paper plan exceeds core/satellite allocation caps")

        de_risk_plan = self._is_drawdown_kill_plan(plan)
        if plan.drawdown >= Decimal("0.15") and not de_risk_plan:
            raise SafetyBlock("de-risked paper plan is missing de-risk provenance")
        if de_risk_plan:
            if direct or abs(core - Decimal("0.69")) > Decimal("0.000000001"):
                raise SafetyBlock("de-risked paper plan must hold 69% SPY and no satellite")
            if abs(cash_weight - Decimal("0.31")) > Decimal("0.000000001"):
                raise SafetyBlock("de-risked paper plan must hold 31% cash")
        else:
            if metadata["promotable"] is not True:
                raise SafetyBlock(
                    "paper plan is not backed by promotable point-in-time data"
                )
            if not Decimal("0.98") <= invested <= Decimal("1"):
                raise SafetyBlock("normal paper plan must remain 98-100% invested")
            if not Decimal("0") <= cash_weight <= Decimal("0.02"):
                raise SafetyBlock("normal paper plan cash must remain within 0-2%")
            if not Decimal("0.69") <= core <= Decimal("0.99"):
                raise SafetyBlock("normal paper plan SPY core must remain within 69-99%")
            if not Decimal("0.90") <= beta <= Decimal("1.10"):
                raise SafetyBlock("normal paper plan beta must remain within 0.90-1.10")
            if tracking_error < 0 or tracking_error > Decimal("0.06"):
                raise SafetyBlock("paper plan exceeds the 6% tracking-error cap")
            if sector_deviation < 0 or sector_deviation > Decimal("0.05"):
                raise SafetyBlock("paper plan exceeds the 5% active-sector cap")

    @staticmethod
    def _is_drawdown_kill_plan(plan: PortfolioPlan) -> bool:
        return (
            plan.metadata.get("drawdown_kill") is True
            or plan.metadata.get("de_risk_active") is True
        )

    def _run_metadata(
        self, trigger: str, plan: PortfolioPlan | None
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "trigger": trigger,
            "commit_sha": self.settings.commit_sha,
        }
        if plan is not None:
            metadata.update(
                {
                    "target_weights": {
                        symbol: str(weight)
                        for symbol, weight in plan.target_weights.items()
                    },
                    "plan_metadata": _json_safe_value(plan.metadata),
                    "drawdown": str(plan.drawdown),
                }
            )
        return metadata

    @staticmethod
    def _run_drawdown(run: RunRecord) -> Decimal:
        try:
            drawdown = Decimal(str(run.metadata.get("drawdown", "0")))
        except Exception:
            return Decimal("0")
        return drawdown if drawdown.is_finite() and drawdown >= 0 else Decimal("0")

    @staticmethod
    def _run_is_de_risked(run: RunRecord) -> bool:
        raw_metadata = run.metadata.get("plan_metadata")
        if not isinstance(raw_metadata, Mapping):
            return False
        return (
            raw_metadata.get("drawdown_kill") is True
            or raw_metadata.get("de_risk_active") is True
        )

    def _account_drawdown(
        self, account: AccountSnapshot, *, fallback: Decimal
    ) -> Decimal:
        if self.paper_risk_state_provider is None:
            return fallback
        try:
            raw_peak = self.paper_risk_state_provider.current_peak(account.account_key)
        except Exception as exc:
            raise SafetyBlock("durable peak-NAV read failed") from exc
        if raw_peak is None:
            raise SafetyBlock("durable peak NAV is unavailable")
        peak = Decimal(str(raw_peak))
        if not peak.is_finite() or peak <= 0:
            raise SafetyBlock("durable peak NAV is invalid")
        return max(Decimal("0"), (peak - account.equity) / peak)

    def _cancel_buy_for_drawdown(
        self,
        intent: OrderIntent,
        snapshot: BrokerOrderSnapshot,
        containment_active: bool,
        lease_owner: str,
    ) -> None:
        if (
            not containment_active
            or intent.side is not OrderSide.BUY
            or snapshot.state.is_terminal
        ):
            return
        if not snapshot.broker_order_id:
            raise ReconciliationRequired(
                f"drawdown buy {intent.client_order_id} has no broker order id"
            )
        assert self.gateway is not None
        self._require_active_lease(lease_owner)
        self.gateway.cancel_order(snapshot.broker_order_id)
        raise ReconciliationRequired(
            f"drawdown containment canceled open buy {intent.client_order_id}"
        )

    @staticmethod
    def _completion_context_from_run(
        run: RunRecord, intents: tuple[OrderIntent, ...]
    ) -> tuple[dict[str, Decimal], dict[str, object], Decimal]:
        raw_weights = run.metadata.get("target_weights")
        target_weights: dict[str, Decimal] = {}
        if isinstance(raw_weights, Mapping):
            for symbol, raw_weight in raw_weights.items():
                weight = Decimal(str(raw_weight))
                if not weight.is_finite() or weight < 0:
                    raise SafetyBlock("persisted target weights are invalid")
                target_weights[str(symbol).upper()] = weight
        else:
            target_weights = {
                intent.symbol: intent.target_weight
                for intent in intents
                if intent.target_weight > 0
            }
        raw_metadata = run.metadata.get("plan_metadata")
        plan_metadata = (
            dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        return target_weights, plan_metadata, ExecutionCoordinator._run_drawdown(run)

    def _submit_and_reconcile(
        self,
        intent: OrderIntent,
        now: datetime,
        lease_owner: str,
        *,
        confirm_absent_before_post: bool = False,
    ) -> None:
        assert self.gateway is not None
        if self.ledger.current_order_state(intent.client_order_id) is not OrderState.INTENT:
            raise ReconciliationRequired(
                f"order {intent.client_order_id} has already crossed the broker boundary"
            )
        try:
            try:
                self._require_active_lease(lease_owner)
            except SafetyBlock as exc:
                raise _LeaseLostBeforeSubmission(intent.client_order_id) from exc
            if confirm_absent_before_post:
                recovered_before_post = self.gateway.get_order_by_client_id(
                    intent.client_order_id
                )
                if recovered_before_post is not None:
                    self._append_snapshot(intent, recovered_before_post)
                    if recovered_before_post.state is OrderState.FILLED:
                        return
                    if recovered_before_post.state.is_terminal:
                        raise SafetyBlock(
                            f"recovered order {intent.client_order_id} ended as "
                            f"{recovered_before_post.state.value}"
                        )
                    raise ReconciliationRequired(
                        f"recovered order {intent.client_order_id} is "
                        f"{recovered_before_post.state.value}; no retry POST was made"
                    )
                self._require_active_lease(lease_owner)
            snapshot = self.gateway.submit_order(intent)
        except BrokerUncertainOutcome:
            self._append_event(intent, OrderState.UNKNOWN, now)
            self._require_active_lease(lease_owner)
            recovered = self.gateway.get_order_by_client_id(intent.client_order_id)
            if recovered is None:
                raise ReconciliationRequired(
                    f"submission outcome remains unknown for {intent.client_order_id}"
                )
            snapshot = recovered
        except BrokerSubmissionError:
            self._append_event(intent, OrderState.REJECTED, now)
            raise

        try:
            self._append_snapshot(intent, snapshot)
        except Exception as exc:
            raise ReconciliationRequired(
                f"broker response for {intent.client_order_id} could not be durably recorded"
            ) from exc
        if snapshot.state.is_terminal:
            if snapshot.state is not OrderState.FILLED:
                raise SafetyBlock(
                    f"order {intent.client_order_id} ended as {snapshot.state.value}"
                )
            return

        self._require_active_lease(lease_owner)
        reconciled = self.gateway.get_order_by_client_id(intent.client_order_id)
        if reconciled is None:
            self._append_event(intent, OrderState.UNKNOWN, now)
            raise ReconciliationRequired(
                f"accepted order {intent.client_order_id} cannot yet be reconciled"
            )
        self._append_snapshot(intent, reconciled)
        self._await_terminal_order(intent, reconciled, now, lease_owner)

    def _await_terminal_order(
        self,
        intent: OrderIntent,
        snapshot: BrokerOrderSnapshot,
        now: datetime,
        lease_owner: str,
    ) -> None:
        """Poll an accepted paper order to fill or the 15-minute cancel boundary."""

        assert self.gateway is not None
        current = snapshot
        started = self.monotonic_clock()
        deadline = started + self.settings.unresolved_order_minutes * 60
        while True:
            if current.state.is_terminal:
                if current.state is OrderState.FILLED:
                    return
                raise SafetyBlock(
                    f"order {intent.client_order_id} ended as {current.state.value}"
                )

            remaining = deadline - self.monotonic_clock()
            if remaining <= 0:
                if not current.broker_order_id:
                    raise ReconciliationRequired(
                        f"order {intent.client_order_id} reached its timeout without a broker id"
                    )
                self._require_active_lease(lease_owner)
                self.gateway.cancel_order(current.broker_order_id)
                self._require_active_lease(lease_owner)
                try:
                    canceled = self.gateway.get_order_by_client_id(
                        intent.client_order_id
                    )
                except BrokerReadError as exc:
                    raise ReconciliationRequired(
                        f"cancel outcome cannot be reconciled for {intent.client_order_id}"
                    ) from exc
                if canceled is None:
                    if (
                        self.ledger.current_order_state(intent.client_order_id)
                        is not OrderState.UNKNOWN
                    ):
                        self._append_event(intent, OrderState.UNKNOWN, now)
                    raise ReconciliationRequired(
                        f"cancel outcome remains unknown for {intent.client_order_id}"
                    )
                self._append_snapshot(intent, canceled)
                if canceled.state is OrderState.FILLED:
                    return
                if canceled.state.is_terminal:
                    raise SafetyBlock(
                        f"order {intent.client_order_id} ended as {canceled.state.value}"
                    )
                raise ReconciliationRequired(
                    f"cancel requested for {intent.client_order_id}; terminal broker state pending"
                )

            if not self.poll_wait(min(self.poll_interval_seconds, remaining)):
                raise ReconciliationRequired(
                    f"order {intent.client_order_id} is {current.state.value}; polling interrupted"
                )
            self._require_active_lease(lease_owner)
            try:
                observed = self.gateway.get_order_by_client_id(
                    intent.client_order_id
                )
            except BrokerReadError as exc:
                raise ReconciliationRequired(
                    f"accepted order {intent.client_order_id} cannot yet be reconciled"
                ) from exc
            if observed is None:
                if (
                    self.ledger.current_order_state(intent.client_order_id)
                    is not OrderState.UNKNOWN
                ):
                    self._append_event(intent, OrderState.UNKNOWN, now)
                continue
            self._append_snapshot(intent, observed)
            current = observed

    def _require_active_lease(self, owner: str) -> None:
        if not self.ledger.renew_lease(
            self.settings.account_key, self.settings.mode, owner
        ):
            raise SafetyBlock("execution lease was lost; no further broker action is allowed")

    def _capture_paper_completion(
        self,
        run_id: str,
        decision_key: str,
        target_hash: str,
        intents: tuple[OrderIntent, ...],
        now: datetime,
        lease_owner: str,
        *,
        target_weights: Mapping[str, Decimal],
        plan_metadata: Mapping[str, object],
        drawdown: Decimal,
    ) -> PaperCompletionSnapshot:
        """Read and optionally persist final broker truth before completion."""

        assert self.gateway is not None
        self._require_active_lease(lease_owner)
        account = self.gateway.get_account()
        self._validate_account_snapshot(account, now)
        self._require_active_lease(lease_owner)
        positions = self.gateway.get_positions()
        self._validate_long_positions(positions)
        events = tuple(
            event
            for intent in intents
            for event in self.ledger.events_for(intent.client_order_id)
        )
        snapshot = PaperCompletionSnapshot(
            run_id=run_id,
            decision_key=decision_key,
            target_hash=target_hash,
            account=account,
            positions=positions,
            intents=intents,
            events=events,
            observed_at=now,
            target_weights=dict(target_weights),
            plan_metadata=dict(plan_metadata),
            drawdown=drawdown,
        )
        max_drift, invested_drift, exposure_breach = self._post_fill_exposure(
            snapshot
        )
        snapshot = PaperCompletionSnapshot(
            run_id=snapshot.run_id,
            decision_key=snapshot.decision_key,
            target_hash=snapshot.target_hash,
            account=snapshot.account,
            positions=snapshot.positions,
            intents=snapshot.intents,
            events=snapshot.events,
            observed_at=snapshot.observed_at,
            target_weights=snapshot.target_weights,
            plan_metadata=snapshot.plan_metadata,
            drawdown=snapshot.drawdown,
            max_position_drift=max_drift,
            invested_weight_drift=invested_drift,
            exposure_breach=exposure_breach,
        )
        if self.paper_completion_recorder is not None:
            self.paper_completion_recorder.record_paper_completion(snapshot)
        return snapshot

    @staticmethod
    def _completion_metadata(snapshot: PaperCompletionSnapshot) -> dict[str, object]:
        return {
            "final_equity": str(snapshot.account.equity),
            "final_cash": str(snapshot.account.cash),
            "final_position_count": len(snapshot.positions),
            "final_event_count": len(snapshot.events),
            "final_observed_at": snapshot.observed_at.isoformat(),
            "final_drawdown": str(snapshot.drawdown),
            "max_position_drift_bps": str(
                snapshot.max_position_drift * Decimal("10000")
            ),
            "invested_weight_drift_bps": str(
                snapshot.invested_weight_drift * Decimal("10000")
            ),
            "post_fill_exposure_breach": snapshot.exposure_breach,
        }

    @staticmethod
    def _post_fill_exposure(
        snapshot: PaperCompletionSnapshot,
    ) -> tuple[Decimal, Decimal, bool]:
        if snapshot.account.equity <= 0:
            return Decimal("1"), Decimal("1"), True
        actual = {
            position.symbol: (
                position.quantity * position.market_price / snapshot.account.equity
            )
            for position in snapshot.positions
        }
        target = dict(snapshot.target_weights)
        symbols = set(actual) | set(target)
        de_risk = (
            snapshot.plan_metadata.get("drawdown_kill") is True
            or snapshot.plan_metadata.get("de_risk_active") is True
        )
        if de_risk:
            # Containment may not buy an underweight SPY core.  Only residual
            # satellite exposure and an overweight core are breaches.
            satellite = sum(
                (weight for symbol, weight in actual.items() if symbol != "SPY"),
                Decimal("0"),
            )
            spy_overweight = max(
                Decimal("0"),
                actual.get("SPY", Decimal("0"))
                - target.get("SPY", Decimal("0")),
            )
            max_drift = max(satellite, spy_overweight)
            invested_drift = max(
                Decimal("0"),
                sum(actual.values(), Decimal("0"))
                - sum(target.values(), Decimal("0")),
            )
        else:
            max_drift = max(
                (
                    abs(
                        actual.get(symbol, Decimal("0"))
                        - target.get(symbol, Decimal("0"))
                    )
                    for symbol in symbols
                ),
                default=Decimal("0"),
            )
            invested_drift = abs(
                sum(actual.values(), Decimal("0"))
                - sum(target.values(), Decimal("0"))
            )
        breach = (
            max_drift > Decimal("0.005")
            or invested_drift > Decimal("0.005")
        )
        return max_drift, invested_drift, breach

    def _append_snapshot(
        self, intent: OrderIntent, snapshot: BrokerOrderSnapshot
    ) -> None:
        if snapshot.client_order_id != intent.client_order_id:
            raise SafetyBlock("broker returned a mismatched client order id")
        if snapshot.symbol != intent.symbol or snapshot.side is not intent.side:
            raise SafetyBlock("broker returned mismatched order identity fields")
        if snapshot.observed_at.tzinfo is None:
            raise SafetyBlock("broker returned a timezone-naive order observation")
        quantity_tolerance = Decimal("0.000000001")
        notional_tolerance = Decimal("0.01")
        if intent.quantity is not None:
            if (
                snapshot.requested_quantity is None
                or not snapshot.requested_quantity.is_finite()
                or abs(snapshot.requested_quantity - intent.quantity)
                > quantity_tolerance
                or snapshot.requested_notional is not None
            ):
                raise SafetyBlock("broker requested quantity differs from frozen intent")
        else:
            assert intent.notional is not None
            if (
                snapshot.requested_notional is None
                or not snapshot.requested_notional.is_finite()
                or abs(snapshot.requested_notional - intent.notional)
                > notional_tolerance
            ):
                raise SafetyBlock("broker requested notional differs from frozen intent")
            if snapshot.requested_quantity is not None and (
                not snapshot.requested_quantity.is_finite()
                or snapshot.requested_quantity <= 0
            ):
                raise SafetyBlock("broker returned an invalid derived order quantity")
        if not snapshot.filled_quantity.is_finite() or snapshot.filled_quantity < 0:
            raise SafetyBlock("broker returned an invalid filled quantity")
        if snapshot.filled_average_price is not None and (
            not snapshot.filled_average_price.is_finite()
            or snapshot.filled_average_price <= 0
        ):
            raise SafetyBlock("broker returned an invalid fill price")
        if snapshot.filled_quantity > 0 and snapshot.filled_average_price is None:
            raise SafetyBlock("broker returned filled quantity without a fill price")
        prior_filled = max(
            (
                event.filled_quantity
                for event in self.ledger.events_for(intent.client_order_id)
            ),
            default=Decimal("0"),
        )
        if snapshot.filled_quantity + quantity_tolerance < prior_filled:
            raise SafetyBlock("broker cumulative filled quantity moved backwards")
        if intent.quantity is not None and (
            snapshot.filled_quantity > intent.quantity + quantity_tolerance
        ):
            raise SafetyBlock("broker fill exceeds the frozen order quantity")
        if (
            intent.notional is not None
            and snapshot.filled_quantity > 0
            and snapshot.filled_average_price is not None
            and snapshot.filled_quantity * snapshot.filled_average_price
            > intent.notional + notional_tolerance
        ):
            raise SafetyBlock("broker fill exceeds the frozen order notional")
        # A market order can fill between the POST and its response.  Preserve
        # the accepted lifecycle milestone even when the first observed SDK
        # snapshot is already partial/terminal; acceptance itself is never
        # interpreted as evidence of a fill.
        if (
            self.ledger.current_order_state(intent.client_order_id) is OrderState.INTENT
            and snapshot.state
            in {
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.CANCELED,
                OrderState.EXPIRED,
            }
        ):
            self.ledger.append_order_event(
                OrderEvent(
                    event_id=str(uuid.uuid4()),
                    client_order_id=intent.client_order_id,
                    state=OrderState.ACCEPTED,
                    observed_at=snapshot.observed_at,
                    broker_order_id=snapshot.broker_order_id,
                )
            )
        slippage_bps: Decimal | None = None
        if snapshot.filled_quantity > 0:
            assert snapshot.filled_average_price is not None
            price_delta = (
                snapshot.filled_average_price - intent.arrival_price
                if intent.side is OrderSide.BUY
                else intent.arrival_price - snapshot.filled_average_price
            )
            slippage_bps = (
                price_delta / intent.arrival_price * Decimal("10000")
            )
        self.ledger.append_order_event(
            OrderEvent(
                event_id=str(uuid.uuid4()),
                client_order_id=intent.client_order_id,
                state=snapshot.state,
                observed_at=snapshot.observed_at,
                broker_order_id=snapshot.broker_order_id,
                filled_quantity=snapshot.filled_quantity,
                filled_average_price=snapshot.filled_average_price,
                slippage_bps=slippage_bps,
            )
        )

    def _append_event(
        self, intent: OrderIntent, state: OrderState, observed_at: datetime
    ) -> None:
        self.ledger.append_order_event(
            OrderEvent(
                event_id=str(uuid.uuid4()),
                client_order_id=intent.client_order_id,
                state=state,
                observed_at=observed_at,
            )
        )

    def _record_paper_planning_failure(
        self,
        *,
        decision_key: str,
        now: datetime,
        trigger: str,
        message: str,
        status: RunStatus = RunStatus.BLOCKED,
        exit_code: int = 2,
        target_hash: str | None = None,
    ) -> RunResult:
        """Persist a fail-closed planning attempt without reserving the decision key."""

        frozen_hash = target_hash or build_target_hash({})
        run_id = ""
        if (
            getattr(self.ledger, "paper_durable_truth", False) is True
            and self.settings.account_key.strip()
        ):
            attempt_key = (
                f"{decision_key}|planning-attempt|"
                f"{now.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
            try:
                record = self.ledger.create_run(
                    RunRecord(
                        run_id=self._new_run_id(),
                        decision_key=attempt_key,
                        strategy_id=self.settings.strategy_id,
                        strategy_version=self.settings.strategy_version,
                        account_key=self.settings.account_key,
                        mode=self.settings.mode,
                        purpose=RunPurpose.REBALANCE,
                        target_hash=frozen_hash,
                        created_at=now,
                        metadata={
                            "trigger": trigger,
                            "commit_sha": self.settings.commit_sha,
                            "intended_decision_key": decision_key,
                            "planning_failure": True,
                        },
                    )
                )
                run_id = record.run_id
                self.ledger.update_run(
                    run_id,
                    status,
                    now,
                    failure_reason=(
                        message
                        if status in {RunStatus.BLOCKED, RunStatus.FAILED}
                        else ""
                    ),
                    metadata={"planning_failure_reason": message},
                )
            except Exception:
                # The original safety failure remains authoritative.  A broken
                # ledger must not be disguised as a successful or executable run.
                run_id = ""
        return RunResult(
            run_id=run_id,
            status=status,
            exit_code=exit_code,
            message=message,
            decision_key=decision_key,
            target_hash=frozen_hash,
        )

    def _reconciliation_block_for_run(
        self, run: RunRecord, now: datetime
    ) -> RunResult:
        message = "an older account order must reconcile before target construction"
        self.ledger.update_run(
            run.run_id,
            RunStatus.RECONCILIATION_REQUIRED,
            now,
            metadata={"recovery_reason": message},
        )
        return RunResult(
            run_id=run.run_id,
            status=RunStatus.RECONCILIATION_REQUIRED,
            exit_code=3,
            message=message,
            decision_key=run.decision_key,
            target_hash=run.target_hash,
        )

    def _finish(
        self,
        run_id: str,
        decision_key: str,
        target_hash: str,
        status: RunStatus,
        exit_code: int,
        message: str,
        now: datetime,
        *,
        order_client_ids: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> RunResult:
        self.ledger.update_run(
            run_id,
            status,
            now,
            failure_reason=message if status in {RunStatus.BLOCKED, RunStatus.FAILED} else "",
            metadata=metadata,
        )
        return RunResult(
            run_id=run_id,
            status=status,
            exit_code=exit_code,
            message=message,
            decision_key=decision_key,
            target_hash=target_hash,
            order_client_ids=order_client_ids,
            metadata={} if metadata is None else dict(metadata),
        )

    def _validate_plan_identity(self, plan: PortfolioPlan) -> None:
        if plan.strategy_id != self.settings.strategy_id:
            raise ValueError("plan strategy id does not match runtime settings")
        if plan.strategy_version != self.settings.strategy_version:
            raise ValueError("plan strategy version does not match runtime settings")

    @staticmethod
    def _enforce_drawdown_plan(plan: PortfolioPlan) -> None:
        if (
            plan.drawdown >= Decimal("0.15")
            or plan.metadata.get("drawdown_kill") is True
            or plan.metadata.get("de_risk_active") is True
        ):
            active = {
                symbol: weight
                for symbol, weight in plan.target_weights.items()
                if symbol != "SPY" and weight > 0
            }
            if active:
                raise SafetyBlock(
                    "15% drawdown plan must set every non-SPY satellite target to zero"
                )
            if abs(
                plan.target_weights.get("SPY", Decimal("0")) - Decimal("0.69")
            ) > Decimal("0.000000001"):
                raise SafetyBlock("15% drawdown plan must target 69% SPY and 31% cash")

    @staticmethod
    def _new_run_id() -> str:
        return InMemoryLedger.new_run_id()


class _MarketClosed(SafetyBlock):
    pass


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _json_safe_value(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_order_intents(
    *,
    run_id: str,
    decision_key: str,
    plan: PortfolioPlan,
    account: AccountSnapshot,
    positions: Sequence[PositionSnapshot],
    open_orders: Sequence[OpenOrderSnapshot],
    assets: Mapping[str, AssetSnapshot],
    quotes: Mapping[str, QuoteSnapshot],
    settings: RuntimeSettings,
    now: datetime,
) -> tuple[OrderIntent, ...]:
    """Convert target weights into deterministic, long-only frozen deltas."""

    if account.equity <= 0:
        raise SafetyBlock("cannot size orders from non-positive equity")
    current = {position.symbol: position.quantity for position in positions}
    effective = dict(current)
    for order in open_orders:
        signed = order.remaining_quantity * (
            Decimal("1") if order.side is OrderSide.BUY else Decimal("-1")
        )
        effective[order.symbol] = effective.get(order.symbol, Decimal("0")) + signed

    symbols = sorted(set(plan.target_weights) | set(effective))
    intents: list[OrderIntent] = []
    for symbol in symbols:
        quote = quotes[symbol]
        asset = assets[symbol]
        price = quote.midpoint
        target_weight = plan.target_weights.get(symbol, Decimal("0"))
        target_notional = account.equity * target_weight
        current_notional = effective.get(symbol, Decimal("0")) * price
        delta_notional = target_notional - current_notional
        absolute_notional = abs(delta_notional)
        drift = absolute_notional / account.equity
        containment_active = (
            plan.drawdown >= Decimal("0.15")
            or plan.metadata.get("drawdown_kill") is True
            or plan.metadata.get("de_risk_active") is True
        )
        if absolute_notional < settings.min_trade_notional:
            continue
        if drift < settings.min_drift_fraction:
            continue
        if symbol == "SPY" and containment_active and drift < Decimal("0.005"):
            # Off-cycle containment uses a wider core band to avoid churning
            # SPY for small drift while satellite risk is being removed.
            continue
        if absolute_notional > quote.adv_dollars_30d * settings.adv_limit_fraction:
            raise SafetyBlock(f"{symbol} order would exceed 5% of 30-day ADV")

        if delta_notional < 0:
            available_qty = max(Decimal("0"), effective.get(symbol, Decimal("0")))
            quantity = min(absolute_notional / price, available_qty)
            if not asset.fractionable:
                quantity = quantity.to_integral_value(rounding=ROUND_DOWN)
            else:
                quantity = quantity.quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
            if quantity <= 0:
                continue
            side = OrderSide.SELL
            amount = quantity
            intent = OrderIntent(
                client_order_id=build_client_order_id(
                    decision_key, symbol, side, amount
                ),
                run_id=run_id,
                decision_key=decision_key,
                sleeve=plan.sleeve,
                symbol=symbol,
                side=side,
                quantity=quantity,
                notional=None,
                target_weight=target_weight,
                arrival_price=price,
                created_at=now,
            )
        else:
            side = OrderSide.BUY
            # The 15% containment path can reduce exposure but must never add
            # exposure, even when current SPY is below the 69% strategic core.
            if containment_active:
                continue
            if asset.fractionable:
                notional = absolute_notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
                if notional < settings.min_trade_notional:
                    continue
                amount = notional
                intent = OrderIntent(
                    client_order_id=build_client_order_id(
                        decision_key, symbol, side, amount
                    ),
                    run_id=run_id,
                    decision_key=decision_key,
                    sleeve=plan.sleeve,
                    symbol=symbol,
                    side=side,
                    quantity=None,
                    notional=notional,
                    target_weight=target_weight,
                    arrival_price=price,
                    created_at=now,
                )
            else:
                quantity = (absolute_notional / price).to_integral_value(
                    rounding=ROUND_DOWN
                )
                if quantity <= 0 or quantity * price < settings.min_trade_notional:
                    continue
                amount = quantity
                intent = OrderIntent(
                    client_order_id=build_client_order_id(
                        decision_key, symbol, side, amount
                    ),
                    run_id=run_id,
                    decision_key=decision_key,
                    sleeve=plan.sleeve,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    notional=None,
                    target_weight=target_weight,
                    arrival_price=price,
                    created_at=now,
                )
        intents.append(intent)
    return tuple(sorted(intents, key=lambda item: (item.side is OrderSide.BUY, item.symbol)))


__all__ = [
    "ExecutionCoordinator",
    "PaperPreflight",
    "PaperCompletionRecorder",
    "PaperCompletionSnapshot",
    "PaperPlanFactory",
    "PaperPlanningContext",
    "PaperRiskStateProvider",
    "ReconciliationRequired",
    "SafetyBlock",
    "build_order_intents",
]
