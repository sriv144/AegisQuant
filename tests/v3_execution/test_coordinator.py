from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from src.execution.v3 import (
    AccountSnapshot,
    AssetSnapshot,
    BrokerOrderSnapshot,
    BrokerReadError,
    BrokerUncertainOutcome,
    CalendarSession,
    ClockSnapshot,
    ExecutionCoordinator as BaseExecutionCoordinator,
    InMemoryLedger as BaseInMemoryLedger,
    OpenOrderSnapshot,
    OrderEvent,
    OrderIntent,
    OrderSide,
    OrderState,
    PortfolioPlan,
    PositionSnapshot,
    QuoteSnapshot,
    RunStatus,
    RunPurpose,
    RunRecord,
    RuntimeSettings,
    TradingMode,
    build_client_order_id,
    build_decision_key,
    build_order_intents,
    build_target_hash,
)


NOW = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)  # 10:30 America/New_York
CONFIG_SHA = "a" * 64


class InMemoryLedger(BaseInMemoryLedger):
    """Test double for an already-verified PostgreSQL ledger capability."""

    paper_durable_truth = True


class StaticPlanFactory:
    def __init__(self, selected_plan: PortfolioPlan) -> None:
        self.selected_plan = selected_plan

    def construct(self, context):
        selected = self.selected_plan
        return PortfolioPlan(
            selected.strategy_id,
            selected.strategy_version,
            selected.as_of,
            selected.target_weights,
            sleeve=selected.sleeve,
            drawdown=context.fresh_drawdown,
            metadata=selected.metadata,
        )


class ExecutionCoordinator(BaseExecutionCoordinator):
    """Paper test harness that always uses the production factory boundary."""

    def __init__(self, settings, ledger, **kwargs):
        kwargs.setdefault("paper_risk_state_provider", FakeRiskState(Decimal("100000")))
        kwargs.setdefault("poll_wait", lambda _seconds: False)
        super().__init__(settings, ledger, **kwargs)

    def run(self, *, plan=None, paper_plan_factory=None, **kwargs):
        if (
            plan is not None
            and self.settings.mode is TradingMode.PAPER
            and self.settings.purpose is RunPurpose.REBALANCE
        ):
            paper_plan_factory = StaticPlanFactory(plan)
            plan = None
        return super().run(
            plan=plan,
            paper_plan_factory=paper_plan_factory,
            **kwargs,
        )


def paper_settings(**overrides) -> RuntimeSettings:
    values = {
        "mode": "paper",
        "purpose": "rebalance",
        "strategy_config_sha256": CONFIG_SHA,
        "execution_enabled": True,
        "kill_switch": False,
        "database_url": "postgresql://user:pass@db/aegis",
        "alpaca_api_key": "paper-key",
        "alpaca_secret_key": "paper-secret",
        "account_key": "paper-account",
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def plan(weights=None) -> PortfolioPlan:
    target_weights = weights or {"SPY": Decimal("0.97"), "AAPL": Decimal("0.02")}
    invested = sum((Decimal(str(v)) for v in target_weights.values()), Decimal("0"))
    return PortfolioPlan(
        "spy_xsmom_core_satellite",
        "3.0.0",
        NOW,
        target_weights,
        metadata={
            "plan_origin": "src.v3.portfolio.PortfolioConstructor",
            "config_sha256": CONFIG_SHA,
            "source_target_sha256": "b" * 64,
            "source_weight_hash": build_target_hash(target_weights),
            "promotable": True,
            "benchmark_symbol": "SPY",
            "cash_weight": Decimal("1") - invested,
            "portfolio_beta": Decimal("0.99"),
            "tracking_error": Decimal("0.03"),
            "max_active_sector_deviation": Decimal("0.02"),
        },
    )


def de_risk_plan(
    *,
    drawdown: Decimal = Decimal("0.15"),
    promotable: bool = False,
) -> PortfolioPlan:
    target_weights = {"SPY": Decimal("0.69")}
    return PortfolioPlan(
        "spy_xsmom_core_satellite",
        "3.0.0",
        NOW,
        target_weights,
        drawdown=drawdown,
        metadata={
            "plan_origin": "src.v3.portfolio.PortfolioConstructor",
            "config_sha256": CONFIG_SHA,
            "source_target_sha256": "c" * 64,
            "source_weight_hash": build_target_hash(target_weights),
            "promotable": promotable,
            "benchmark_symbol": "SPY",
            "cash_weight": Decimal("0.31"),
            "portfolio_beta": Decimal("0.69"),
            "tracking_error": Decimal("0.40"),
            "max_active_sector_deviation": Decimal("0.40"),
            "de_risk_active": True,
        },
    )


def quote(symbol: str, *, observed_at: datetime = NOW, adv: str = "10000000") -> QuoteSnapshot:
    return QuoteSnapshot(
        symbol=symbol,
        bid_price=Decimal("99.99"),
        ask_price=Decimal("100.01"),
        observed_at=observed_at,
        adv_dollars_30d=Decimal(adv),
    )


class FakeGateway:
    def __init__(self) -> None:
        self.calls = Counter()
        self.fail_read: str | None = None
        self.market_open = True
        self.positions: tuple[PositionSnapshot, ...] = ()
        self.open_orders: tuple[OpenOrderSnapshot, ...] = ()
        self.quotes: dict[str, QuoteSnapshot] = {
            "SPY": quote("SPY"),
            "AAPL": quote("AAPL"),
        }
        self.submit_state = OrderState.FILLED
        self.reconcile_state: OrderState | None = OrderState.FILLED
        self.uncertain = False
        self.buying_power = Decimal("100000")
        self.cash = Decimal("100000")
        self.fractionable = True
        self.asset_class = "us_equity"
        self.fill_price_delta = Decimal("0")
        self.submitted = []
        self.cancelled = []
        self.applied_fills: set[str] = set()

    def _called(self, name: str) -> None:
        self.calls[name] += 1
        if self.fail_read == name:
            raise BrokerReadError(f"forced {name} failure")

    def get_account(self) -> AccountSnapshot:
        self._called("get_account")
        return AccountSnapshot(
            "paper-account",
            Decimal("100000"),
            self.cash,
            self.buying_power,
            "active",
            NOW,
        )

    def get_positions(self) -> tuple[PositionSnapshot, ...]:
        self._called("get_positions")
        return self.positions

    def get_open_orders(self) -> tuple[OpenOrderSnapshot, ...]:
        self._called("get_open_orders")
        return self.open_orders

    def get_clock(self) -> ClockSnapshot:
        self._called("get_clock")
        return ClockSnapshot(self.market_open, NOW, NOW, NOW + timedelta(hours=5))

    def get_calendar(self, start: date, end: date) -> tuple[CalendarSession, ...]:
        self._called("get_calendar")
        return (CalendarSession(date(2026, 7, 1), NOW, NOW + timedelta(hours=6)),)

    def get_assets(self, symbols):
        self._called("get_assets")
        return {
            symbol: AssetSnapshot(
                symbol,
                tradable=True,
                fractionable=self.fractionable,
                asset_class=self.asset_class,
            )
            for symbol in symbols
        }

    def get_latest_quotes(self, symbols):
        self._called("get_latest_quotes")
        return {symbol: self.quotes[symbol] for symbol in symbols}

    def submit_order(self, intent):
        self.calls["submit_order"] += 1
        self.submitted.append(intent)
        if self.uncertain:
            raise BrokerUncertainOutcome("forced timeout")
        snapshot = self._snapshot(intent, self.submit_state)
        self._apply_fill(intent, snapshot)
        return snapshot

    def get_order_by_client_id(self, client_order_id):
        self._called("get_order_by_client_id")
        if self.reconcile_state is None:
            return None
        intent = next(
            (item for item in self.submitted if item.client_order_id == client_order_id),
            None,
        )
        if intent is None:
            return None
        snapshot = self._snapshot(intent, self.reconcile_state)
        self._apply_fill(intent, snapshot)
        return snapshot

    def cancel_order(self, broker_order_id):
        self.calls["cancel_order"] += 1
        self.cancelled.append(broker_order_id)

    def _snapshot(self, intent, state: OrderState) -> BrokerOrderSnapshot:
        quantity = intent.quantity
        fill_price = None
        if state is OrderState.FILLED:
            fill_price = (
                intent.arrival_price + self.fill_price_delta
                if intent.side is OrderSide.BUY
                else intent.arrival_price - self.fill_price_delta
            )
            if quantity is None:
                quantity = intent.notional / fill_price
        return BrokerOrderSnapshot(
            broker_order_id=f"broker-{intent.client_order_id}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            state=state,
            requested_quantity=intent.quantity,
            requested_notional=intent.notional,
            filled_quantity=quantity if state is OrderState.FILLED else Decimal("0"),
            filled_average_price=fill_price,
            observed_at=NOW,
        )

    def _apply_fill(self, intent, snapshot: BrokerOrderSnapshot) -> None:
        if (
            snapshot.state is not OrderState.FILLED
            or intent.client_order_id in self.applied_fills
        ):
            return
        self.applied_fills.add(intent.client_order_id)
        positions = {position.symbol: position for position in self.positions}
        prior = positions.get(intent.symbol)
        prior_qty = Decimal("0") if prior is None else prior.quantity
        signed = snapshot.filled_quantity * (
            Decimal("1") if intent.side is OrderSide.BUY else Decimal("-1")
        )
        ending = prior_qty + signed
        if ending > 0:
            positions[intent.symbol] = PositionSnapshot(
                intent.symbol,
                ending,
                snapshot.filled_average_price or intent.arrival_price,
            )
        else:
            positions.pop(intent.symbol, None)
        self.positions = tuple(sorted(positions.values(), key=lambda item: item.symbol))
        fill_value = snapshot.filled_quantity * (
            snapshot.filled_average_price or intent.arrival_price
        )
        self.cash += fill_value * (
            Decimal("-1") if intent.side is OrderSide.BUY else Decimal("1")
        )


class FakeRiskState:
    def __init__(
        self,
        peak: Decimal,
        *,
        de_risked: bool = False,
        de_risked_at: datetime | None = None,
    ) -> None:
        self.peak = peak
        self.de_risked = de_risked
        self.de_risked_at = de_risked_at or (
            datetime(2026, 6, 1, 14, 30, tzinfo=UTC) if de_risked else None
        )
        self.calls = Counter()

    def current_peak(self, account_key: str) -> Decimal:
        self.calls["current_peak"] += 1
        assert account_key == "paper-account"
        return self.peak

    def is_de_risked(self, account_key: str) -> bool:
        self.calls["is_de_risked"] += 1
        assert account_key == "paper-account"
        return self.de_risked

    def de_risked_since(self, account_key: str) -> datetime | None:
        self.calls["de_risked_since"] += 1
        assert account_key == "paper-account"
        return self.de_risked_at


def test_paper_prerequisite_failure_is_blocked_and_makes_zero_broker_calls() -> None:
    settings = RuntimeSettings(mode="paper", purpose="rebalance")
    gateway = FakeGateway()
    result = ExecutionCoordinator(settings, InMemoryLedger(), gateway=gateway).run(
        plan=plan(), now=NOW
    )
    assert result.status is RunStatus.BLOCKED
    assert result.exit_code == 2
    assert sum(gateway.calls.values()) == 0
    assert "not explicitly enabled" in result.message


def test_precomputed_paper_plan_is_rejected_and_persisted_without_broker_reads() -> None:
    ledger = InMemoryLedger()
    gateway = FakeGateway()
    result = BaseExecutionCoordinator(
        paper_settings(),
        ledger,
        gateway=gateway,
        paper_risk_state_provider=FakeRiskState(Decimal("100000")),
    ).run(plan=plan(), now=NOW)

    assert result.status is RunStatus.BLOCKED
    assert result.run_id
    assert "PaperPlanFactory" in result.message
    assert sum(gateway.calls.values()) == 0
    assert any(run.run_id == result.run_id for run in ledger._runs.values())


def test_factory_static_gate_failure_persists_blocked_attempt_without_broker_reads() -> None:
    ledger = InMemoryLedger()
    gateway = FakeGateway()
    result = BaseExecutionCoordinator(
        paper_settings(execution_enabled=False),
        ledger,
        gateway=gateway,
        paper_risk_state_provider=FakeRiskState(Decimal("100000")),
    ).run(paper_plan_factory=StaticPlanFactory(plan()), now=NOW)

    assert result.status is RunStatus.BLOCKED
    assert result.run_id
    assert sum(gateway.calls.values()) == 0
    persisted = next(run for run in ledger._runs.values() if run.run_id == result.run_id)
    assert persisted.status is RunStatus.BLOCKED
    assert persisted.metadata["planning_failure"] is True


def test_postgres_url_string_cannot_make_inmemory_ledger_paper_durable() -> None:
    gateway = FakeGateway()
    result = ExecutionCoordinator(
        paper_settings(), BaseInMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "verified durable PostgreSQL ledger" in result.message
    assert sum(gateway.calls.values()) == 0


def test_unprovenanced_or_nonpromotable_plan_is_blocked_before_broker_read() -> None:
    gateway = FakeGateway()
    untrusted = PortfolioPlan(
        "spy_xsmom_core_satellite", "3.0.0", NOW, {"SPY": Decimal("0.99")}
    )
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=untrusted, now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "constructor provenance" in result.message
    assert gateway.calls["submit_order"] == 0


def test_direct_issuer_cap_is_checked_before_broker_read() -> None:
    gateway = FakeGateway()
    over_cap = plan({"SPY": Decimal("0.96"), "AAPL": Decimal("0.03")})
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=over_cap, now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "2% direct issuer cap" in result.message
    assert gateway.calls["submit_order"] == 0


def test_broker_read_failure_is_blocked_and_never_submits() -> None:
    gateway = FakeGateway()
    gateway.fail_read = "get_positions"
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert result.exit_code == 2
    assert gateway.calls["submit_order"] == 0


def test_broker_account_fingerprint_must_match_configured_idempotency_namespace() -> None:
    gateway = FakeGateway()
    gateway.get_account = lambda: AccountSnapshot(
        "different-account",
        Decimal("100000"),
        Decimal("100000"),
        Decimal("100000"),
        "active",
        NOW,
    )
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "unexpected account key" in result.message
    assert gateway.calls["submit_order"] == 0


def test_empty_positions_is_a_valid_successful_read_not_a_read_failure() -> None:
    gateway = FakeGateway()
    gateway.positions = ()
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.COMPLETED
    assert gateway.calls["submit_order"] == 2


def test_closed_market_is_valid_skip_with_no_submission() -> None:
    gateway = FakeGateway()
    gateway.market_open = False
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.SKIPPED_MARKET_CLOSED
    assert result.exit_code == 0
    assert gateway.calls["submit_order"] == 0


def test_stale_quote_blocks_before_intents_or_submission() -> None:
    gateway = FakeGateway()
    gateway.quotes["AAPL"] = quote("AAPL", observed_at=NOW - timedelta(seconds=61))
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "stale" in result.message
    assert gateway.calls["submit_order"] == 0


def test_manual_open_order_for_target_symbol_blocks_submission() -> None:
    gateway = FakeGateway()
    gateway.open_orders = (
        OpenOrderSnapshot(
            "manual-id",
            "user-manual-order",
            "SPY",
            OrderSide.BUY,
            Decimal("1"),
            Decimal("0"),
            OrderState.ACCEPTED,
            NOW - timedelta(minutes=1),
        ),
    )
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "manual or unattributed" in result.message
    assert gateway.calls["submit_order"] == 0


def test_unknown_aq3_prefix_is_unattributed_and_never_cancelled_or_trusted() -> None:
    gateway = FakeGateway()
    gateway.open_orders = (
        OpenOrderSnapshot(
            "broker-old",
            "aq3-p-202607-oldoldoldoldoldoldold0",
            "SPY",
            OrderSide.BUY,
            Decimal("1"),
            Decimal("0"),
            OrderState.ACCEPTED,
            NOW - timedelta(minutes=16),
        ),
    )
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert result.exit_code == 2
    assert "unknown aq3-prefixed" in result.message
    assert gateway.cancelled == []
    assert gateway.calls["submit_order"] == 0


def test_adv_limit_blocks_oversized_order() -> None:
    gateway = FakeGateway()
    gateway.quotes["AAPL"] = quote("AAPL", adv="10000")
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "5%" in result.message
    assert gateway.calls["submit_order"] == 0


def test_aggregate_cash_is_checked_before_first_buy_post_and_margin_is_ignored() -> None:
    gateway = FakeGateway()
    gateway.cash = Decimal("50000")
    gateway.buying_power = Decimal("1000000")
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "aggregate buys" in result.message
    assert gateway.calls["submit_order"] == 0


def test_intents_are_persisted_and_sells_fill_before_buys() -> None:
    gateway = FakeGateway()
    gateway.positions = (
        PositionSnapshot("OLD", Decimal("100"), Decimal("100")),
    )
    gateway.quotes["OLD"] = quote("OLD")
    ledger = InMemoryLedger()
    result = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=plan(), now=NOW
    )
    assert result.status is RunStatus.COMPLETED
    assert [intent.side for intent in gateway.submitted] == [
        OrderSide.SELL,
        OrderSide.BUY,
        OrderSide.BUY,
    ]
    assert gateway.calls["get_account"] == 3  # preflight, post-sell, final truth
    persisted = ledger.intents_for_run(result.run_id)
    assert len(persisted) == 3
    assert all(ledger.current_order_state(i.client_order_id) is OrderState.FILLED for i in persisted)


def test_acceptance_is_not_treated_as_fill_and_halts_following_orders() -> None:
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    gateway.reconcile_state = OrderState.ACCEPTED
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.RECONCILIATION_REQUIRED
    assert result.exit_code == 3
    assert gateway.calls["submit_order"] == 1
    assert gateway.calls["get_order_by_client_id"] == 1


def test_accepted_order_is_polled_to_fill_before_next_submission() -> None:
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    states = iter((OrderState.ACCEPTED, OrderState.FILLED))

    def scripted_lookup(client_order_id):
        gateway._called("get_order_by_client_id")
        intent = next(
            item for item in gateway.submitted if item.client_order_id == client_order_id
        )
        snapshot = gateway._snapshot(intent, next(states))
        gateway._apply_fill(intent, snapshot)
        return snapshot

    gateway.get_order_by_client_id = scripted_lookup
    result = ExecutionCoordinator(
        paper_settings(),
        InMemoryLedger(),
        gateway=gateway,
        poll_wait=lambda _seconds: True,
        monotonic_clock=lambda: 0.0,
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)

    assert result.status is RunStatus.COMPLETED
    assert gateway.calls["get_order_by_client_id"] == 2
    assert gateway.calls["cancel_order"] == 0


def test_accepted_order_is_canceled_at_timeout_and_batch_halts() -> None:
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    states = iter((OrderState.ACCEPTED, OrderState.CANCELED))

    def scripted_lookup(client_order_id):
        gateway._called("get_order_by_client_id")
        intent = next(
            item for item in gateway.submitted if item.client_order_id == client_order_id
        )
        return gateway._snapshot(intent, next(states))

    clock_values = iter((0.0, 901.0))
    gateway.get_order_by_client_id = scripted_lookup
    result = ExecutionCoordinator(
        paper_settings(unresolved_order_minutes=15),
        InMemoryLedger(),
        gateway=gateway,
        poll_wait=lambda _seconds: True,
        monotonic_clock=lambda: next(clock_values),
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)

    assert result.status is RunStatus.BLOCKED
    assert result.exit_code == 2
    assert gateway.calls["cancel_order"] == 1
    assert gateway.calls["submit_order"] == 1


def test_read_failure_after_acceptance_preserves_reconciliation_required_status() -> None:
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    gateway.fail_read = "get_order_by_client_id"
    ledger = InMemoryLedger()
    result = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=plan({"SPY": Decimal("0.99")}),
        now=NOW,
    )
    assert result.status is RunStatus.RECONCILIATION_REQUIRED
    assert result.exit_code == 3
    assert ledger.get_run_by_decision_key(result.decision_key).status is RunStatus.RECONCILIATION_REQUIRED


def test_existing_durable_intents_are_never_reposted_after_failed_status() -> None:
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    gateway.reconcile_state = OrderState.ACCEPTED
    ledger = InMemoryLedger()
    coordinator = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway)
    first = coordinator.run(plan=plan(), now=NOW)
    assert first.status is RunStatus.RECONCILIATION_REQUIRED
    posts = gateway.calls["submit_order"]
    ledger.update_run(first.run_id, RunStatus.FAILED, NOW, failure_reason="simulated crash")

    retry = coordinator.run(plan=plan(), now=NOW + timedelta(seconds=1))
    assert retry.status is RunStatus.RECONCILIATION_REQUIRED
    assert gateway.calls["submit_order"] == posts
    assert ledger.get_run_by_decision_key(first.decision_key).status is RunStatus.RECONCILIATION_REQUIRED


def test_uncertain_submission_queries_by_client_id_without_retrying_post() -> None:
    gateway = FakeGateway()
    gateway.uncertain = True
    gateway.reconcile_state = OrderState.FILLED
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)
    assert result.status is RunStatus.COMPLETED
    assert gateway.calls["submit_order"] == 1
    assert gateway.calls["get_order_by_client_id"] == 1


def test_unknown_submission_outcome_records_reconciliation_required() -> None:
    gateway = FakeGateway()
    gateway.uncertain = True
    gateway.reconcile_state = None
    ledger = InMemoryLedger()
    result = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=plan({"SPY": Decimal("0.99")}), now=NOW
    )
    assert result.status is RunStatus.RECONCILIATION_REQUIRED
    assert gateway.calls["submit_order"] == 1
    assert gateway.calls["get_order_by_client_id"] == 1
    intent = ledger.intents_for_run(result.run_id)[0]
    assert ledger.current_order_state(intent.client_order_id) is OrderState.UNKNOWN


def test_same_month_changed_target_is_blocked_as_conflicting_frozen_decision() -> None:
    settings = RuntimeSettings(mode="shadow", purpose="rebalance")
    ledger = InMemoryLedger()
    coordinator = ExecutionCoordinator(settings, ledger)
    from src.execution.v3 import ShadowAccount

    account = ShadowAccount(settings.account_key, Decimal("100000"))
    quotes = {"SPY": quote("SPY"), "AAPL": quote("AAPL")}
    first = coordinator.run(
        plan=plan({"SPY": 0.99}), now=NOW, shadow_account=account, shadow_quotes=quotes
    )
    second = coordinator.run(
        plan=plan({"SPY": 0.69, "AAPL": 0.30}),
        now=NOW,
        shadow_account=account,
        shadow_quotes=quotes,
    )
    assert first.status is RunStatus.COMPLETED
    assert second.status is RunStatus.BLOCKED
    assert "different target" in second.message


def test_health_operation_does_not_freeze_or_conflict_with_monthly_rebalance() -> None:
    ledger = InMemoryLedger()
    health = ExecutionCoordinator(
        RuntimeSettings(mode="shadow", purpose="health"), ledger
    ).run(now=NOW)
    assert health.status is RunStatus.SKIPPED_NOT_DUE
    assert "|health|2026-07-01" in health.decision_key

    rebalance_settings = RuntimeSettings(mode="shadow", purpose="rebalance")
    from src.execution.v3 import ShadowAccount

    rebalance = ExecutionCoordinator(rebalance_settings, ledger).run(
        plan=plan({"SPY": 0.99}),
        now=NOW,
        shadow_account=ShadowAccount(rebalance_settings.account_key, Decimal("100000")),
        shadow_quotes={"SPY": quote("SPY")},
    )
    assert rebalance.status is RunStatus.COMPLETED
    assert rebalance.decision_key.endswith("|shadow|2026-07")


def test_reconcile_never_auto_posts_a_trailing_persisted_intent() -> None:
    ledger = InMemoryLedger()
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    gateway.reconcile_state = OrderState.ACCEPTED
    first = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=plan(), now=NOW
    )
    assert first.status is RunStatus.RECONCILIATION_REQUIRED
    assert gateway.calls["submit_order"] == 1

    gateway.submit_state = OrderState.FILLED
    gateway.reconcile_state = OrderState.FILLED
    reconciliation = ExecutionCoordinator(
        paper_settings(purpose="reconcile"), ledger, gateway=gateway
    ).run(now=NOW + timedelta(seconds=30))
    assert reconciliation.status is RunStatus.RECONCILIATION_REQUIRED
    assert "|reconcile|2026-07-01" in reconciliation.decision_key
    # The first accepted intent was looked up, never POSTed twice. The second
    # persisted INTENT is ambiguous after recovery and is never auto-submitted.
    assert len({intent.client_order_id for intent in gateway.submitted}) == 1
    assert gateway.calls["submit_order"] == 1
    assert ledger.get_run_by_decision_key(first.decision_key).status is RunStatus.RECONCILIATION_REQUIRED


def test_reconcile_with_no_unresolved_run_is_a_valid_noop() -> None:
    result = ExecutionCoordinator(
        paper_settings(purpose="reconcile"), InMemoryLedger(), gateway=FakeGateway()
    ).run(now=NOW)
    assert result.status is RunStatus.SKIPPED_NOT_DUE
    assert result.exit_code == 0


def test_kill_switch_allows_cleanup_of_accepted_order_but_no_new_post() -> None:
    ledger = InMemoryLedger()
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    gateway.reconcile_state = OrderState.ACCEPTED
    single = plan({"SPY": Decimal("0.99")})
    first = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=single, now=NOW
    )
    assert first.status is RunStatus.RECONCILIATION_REQUIRED
    posts = gateway.calls["submit_order"]

    gateway.reconcile_state = OrderState.FILLED
    cleanup_settings = paper_settings(
        purpose="reconcile", execution_enabled=False, kill_switch=True
    )
    cleanup = ExecutionCoordinator(
        cleanup_settings, ledger, gateway=gateway
    ).run(now=NOW)
    assert cleanup.status is RunStatus.COMPLETED
    assert gateway.calls["submit_order"] == posts


def test_nonfractionable_buy_uses_quantity_without_assertion_or_notional_post() -> None:
    gateway = FakeGateway()
    gateway.fractionable = False
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(
        plan=plan({"SPY": Decimal("0.99")}),
        now=NOW,
    )
    assert result.status is RunStatus.COMPLETED
    assert gateway.submitted[0].quantity == Decimal("990")
    assert gateway.submitted[0].notional is None


def test_crash_after_post_before_first_event_recovers_by_client_id_without_second_post() -> None:
    class AppendFailsOnceLedger(InMemoryLedger):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next_append = True

        def append_order_event(self, event):
            if self.fail_next_append:
                self.fail_next_append = False
                raise RuntimeError("simulated DB failure after broker response")
            return super().append_order_event(event)

    ledger = AppendFailsOnceLedger()
    gateway = FakeGateway()
    single = plan({"SPY": Decimal("0.99")})
    failed = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=single, now=NOW
    )
    assert failed.status is RunStatus.RECONCILIATION_REQUIRED
    assert failed.exit_code == 3
    assert gateway.calls["submit_order"] == 1
    persisted = ledger.intents_for_run(failed.run_id)
    assert ledger.current_order_state(persisted[0].client_order_id) is OrderState.INTENT

    recovered = ExecutionCoordinator(
        paper_settings(purpose="reconcile"), ledger, gateway=gateway
    ).run(now=NOW + timedelta(seconds=1))
    assert recovered.status is RunStatus.COMPLETED
    assert gateway.calls["submit_order"] == 1
    assert gateway.calls["get_order_by_client_id"] >= 1
    assert ledger.current_order_state(persisted[0].client_order_id) is OrderState.FILLED


def test_lost_lease_blocks_before_first_order_post() -> None:
    class LeaseLosingLedger(InMemoryLedger):
        def renew_lease(self, account_key, mode, owner):
            return False

    ledger = LeaseLosingLedger()
    gateway = FakeGateway()
    result = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=plan(), now=NOW
    )
    assert result.status is RunStatus.BLOCKED
    assert result.exit_code == 2
    assert "lease was lost" in result.message
    assert gateway.calls["submit_order"] == 0


def test_older_cross_month_unresolved_intent_blocks_new_month_before_broker_read() -> None:
    ledger = InMemoryLedger()
    june = datetime(2026, 6, 2, 14, 30, tzinfo=UTC)
    decision = build_decision_key(
        "spy_xsmom_core_satellite",
        "3.0.0",
        "paper-account",
        TradingMode.PAPER,
        june,
    )
    run = RunRecord(
        run_id="june-run",
        decision_key=decision,
        strategy_id="spy_xsmom_core_satellite",
        strategy_version="3.0.0",
        account_key="paper-account",
        mode=TradingMode.PAPER,
        purpose=RunPurpose.REBALANCE,
        target_hash="c" * 64,
        created_at=june,
        status=RunStatus.FAILED,
    )
    ledger.create_run(run)
    client_id = build_client_order_id(
        decision, "SPY", OrderSide.BUY, Decimal("99000")
    )
    intent = OrderIntent(
        client_order_id=client_id,
        run_id=run.run_id,
        decision_key=decision,
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.99"),
        arrival_price=Decimal("100"),
        created_at=june,
        notional=Decimal("99000"),
    )
    ledger.add_intents((intent,))
    ledger.append_order_event(
        OrderEvent("june-unknown", client_id, OrderState.UNKNOWN, june)
    )
    gateway = FakeGateway()
    result = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway).run(
        plan=plan({"SPY": Decimal("0.99")}), now=NOW
    )
    assert result.status is RunStatus.RECONCILIATION_REQUIRED
    assert result.run_id == "june-run"
    assert "older account order" in result.message
    assert sum(gateway.calls.values()) == 0


def test_final_account_positions_and_events_are_recorded_before_completed() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.snapshots = []

        def record_paper_completion(self, snapshot) -> None:
            self.snapshots.append(snapshot)

    recorder = Recorder()
    gateway = FakeGateway()
    result = ExecutionCoordinator(
        paper_settings(),
        InMemoryLedger(),
        gateway=gateway,
        paper_completion_recorder=recorder,
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)
    assert result.status is RunStatus.COMPLETED
    assert len(recorder.snapshots) == 1
    snapshot = recorder.snapshots[0]
    assert snapshot.run_id == result.run_id
    assert snapshot.account.account_key == "paper-account"
    assert snapshot.intents
    assert snapshot.events
    assert result.metadata["final_event_count"] == len(snapshot.events)


def test_final_positions_read_failure_prevents_completed_status_after_fills() -> None:
    class FinalReadFailsGateway(FakeGateway):
        def get_positions(self):
            self.calls["get_positions"] += 1
            if self.calls["get_positions"] == 2:
                raise BrokerReadError("final positions unavailable")
            return self.positions

    gateway = FinalReadFailsGateway()
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)
    assert gateway.calls["submit_order"] == 1
    assert result.status is RunStatus.RECONCILIATION_REQUIRED
    assert result.exit_code == 3
    assert "final positions unavailable" in result.message


def test_nonpromotable_drawdown_plan_ignores_normal_risk_caps_and_only_sells() -> None:
    gateway = FakeGateway()
    gateway.positions = (
        PositionSnapshot("SPY", Decimal("500"), Decimal("100")),
        PositionSnapshot("AAPL", Decimal("200"), Decimal("100")),
    )
    ledger = InMemoryLedger()
    result = ExecutionCoordinator(
        paper_settings(), ledger, gateway=gateway
    ).run(plan=de_risk_plan(), now=NOW)

    assert result.status is RunStatus.COMPLETED
    assert "|risk|2026-07-01" in result.decision_key
    assert [(intent.symbol, intent.side) for intent in gateway.submitted] == [
        ("AAPL", OrderSide.SELL)
    ]


def test_lease_safe_factory_receives_fresh_drawdown_and_constructs_containment() -> None:
    class Factory:
        def __init__(self) -> None:
            self.contexts = []

        def construct(self, context):
            self.contexts.append(context)
            return de_risk_plan(drawdown=context.fresh_drawdown)

    gateway = FakeGateway()
    gateway.positions = (
        PositionSnapshot("SPY", Decimal("500"), Decimal("100")),
        PositionSnapshot("AAPL", Decimal("200"), Decimal("100")),
    )
    risk = FakeRiskState(Decimal("120000"))
    factory = Factory()
    result = ExecutionCoordinator(
        paper_settings(),
        InMemoryLedger(),
        gateway=gateway,
        paper_risk_state_provider=risk,
    ).run(paper_plan_factory=factory, now=NOW)

    assert result.status is RunStatus.COMPLETED
    assert len(factory.contexts) == 1
    context = factory.contexts[0]
    assert context.fresh_drawdown == Decimal("0.1666666666666666666666666667")
    assert context.current_holdings == frozenset({"SPY", "AAPL"})
    assert [(intent.symbol, intent.side) for intent in gateway.submitted] == [
        ("AAPL", OrderSide.SELL)
    ]
    assert risk.calls["current_peak"] == 1


def test_prior_de_risk_state_blocks_unapproved_reentry_below_ten_percent() -> None:
    class Factory:
        def construct(self, context):
            base = plan()
            return PortfolioPlan(
                base.strategy_id,
                base.strategy_version,
                base.as_of,
                base.target_weights,
                drawdown=context.fresh_drawdown,
                metadata=base.metadata,
            )

    gateway = FakeGateway()
    risk = FakeRiskState(Decimal("105000"), de_risked=True)
    result = ExecutionCoordinator(
        paper_settings(),
        InMemoryLedger(),
        gateway=gateway,
        paper_risk_state_provider=risk,
    ).run(paper_plan_factory=Factory(), now=NOW)

    assert result.status is RunStatus.BLOCKED
    assert "manual approval" in result.message
    assert gateway.calls["submit_order"] == 0


def test_non_us_equity_position_or_target_blocks_with_zero_posts() -> None:
    position_gateway = FakeGateway()
    position_gateway.positions = (
        PositionSnapshot(
            "BTCUSD", Decimal("1"), Decimal("50000"), asset_class="crypto"
        ),
    )
    position_result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=position_gateway
    ).run(plan=plan(), now=NOW)
    assert position_result.status is RunStatus.BLOCKED
    assert "non-US-equity" in position_result.message
    assert position_gateway.calls["submit_order"] == 0

    target_gateway = FakeGateway()
    target_gateway.asset_class = "crypto"
    target_result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=target_gateway
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)
    assert target_result.status is RunStatus.BLOCKED
    assert "not a US equity" in target_result.message
    assert target_gateway.calls["submit_order"] == 0


def test_fill_events_record_signed_arrival_slippage_and_plan_provenance() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.snapshot = None

        def record_paper_completion(self, snapshot) -> None:
            self.snapshot = snapshot

    gateway = FakeGateway()
    gateway.fill_price_delta = Decimal("1")
    recorder = Recorder()
    selected_plan = plan({"SPY": Decimal("0.99")})
    result = ExecutionCoordinator(
        paper_settings(),
        InMemoryLedger(),
        gateway=gateway,
        paper_completion_recorder=recorder,
    ).run(plan=selected_plan, now=NOW)

    assert result.status is RunStatus.COMPLETED
    assert recorder.snapshot is not None
    filled = [
        event
        for event in recorder.snapshot.events
        if event.state is OrderState.FILLED
    ]
    assert [event.slippage_bps for event in filled] == [Decimal("100")]
    assert recorder.snapshot.target_weights == selected_plan.target_weights
    assert recorder.snapshot.plan_metadata["config_sha256"] == CONFIG_SHA
    assert recorder.snapshot.drawdown == Decimal("0")


def test_completion_recorder_crash_reconciles_all_fills_without_second_post() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.fail = True
            self.snapshots = []

        def record_paper_completion(self, snapshot) -> None:
            if self.fail:
                self.fail = False
                raise RuntimeError("simulated completion transaction failure")
            self.snapshots.append(snapshot)

    ledger = InMemoryLedger()
    gateway = FakeGateway()
    recorder = Recorder()
    first = ExecutionCoordinator(
        paper_settings(),
        ledger,
        gateway=gateway,
        paper_completion_recorder=recorder,
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)
    assert first.status is RunStatus.RECONCILIATION_REQUIRED
    assert first.exit_code == 3
    assert gateway.calls["submit_order"] == 1

    cleanup = ExecutionCoordinator(
        paper_settings(purpose="reconcile"),
        ledger,
        gateway=gateway,
        paper_completion_recorder=recorder,
    ).run(now=NOW + timedelta(seconds=1))
    assert cleanup.status is RunStatus.COMPLETED
    assert gateway.calls["submit_order"] == 1
    assert ledger.get_run_by_decision_key(first.decision_key).status is RunStatus.COMPLETED
    assert len(recorder.snapshots) == 1


def test_lease_loss_before_post_is_confirmed_absent_then_retried_once() -> None:
    class LeaseLedger(InMemoryLedger):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next_renew = False

        def renew_lease(self, account_key, mode, owner):
            if self.fail_next_renew:
                self.fail_next_renew = False
                return False
            return super().renew_lease(account_key, mode, owner)

    class LoseAtPostGateway(FakeGateway):
        def __init__(self, ledger) -> None:
            super().__init__()
            self.ledger = ledger

        def get_account(self):
            snapshot = super().get_account()
            # The second account read is the refreshed-cash read immediately
            # before the first buy. Lose the lease at the following POST gate.
            if self.calls["get_account"] == 2:
                self.ledger.fail_next_renew = True
            return snapshot

    ledger = LeaseLedger()
    gateway = LoseAtPostGateway(ledger)
    coordinator = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway)
    selected = plan({"SPY": Decimal("0.99")})

    first = coordinator.run(plan=selected, now=NOW)
    assert first.status is RunStatus.BLOCKED
    assert gateway.calls["submit_order"] == 0
    frozen = ledger.intents_for_run(first.run_id)
    assert len(frozen) == 1
    assert frozen[0].client_order_id in ledger.get_run_by_decision_key(
        first.decision_key
    ).metadata["safe_retry_client_ids"]

    retry = coordinator.run(plan=selected, now=NOW + timedelta(seconds=1))
    assert retry.status is RunStatus.COMPLETED
    assert gateway.calls["get_order_by_client_id"] >= 1
    assert gateway.calls["submit_order"] == 1
    assert ledger.current_order_state(frozen[0].client_order_id) is OrderState.FILLED


def test_terminal_reject_aborts_trailing_unsubmitted_intents_without_deadlock() -> None:
    ledger = InMemoryLedger()
    gateway = FakeGateway()
    gateway.submit_state = OrderState.REJECTED
    coordinator = ExecutionCoordinator(paper_settings(), ledger, gateway=gateway)
    selected = plan()

    first = coordinator.run(plan=selected, now=NOW)
    assert first.status is RunStatus.BLOCKED
    assert gateway.calls["submit_order"] == 1
    intents = ledger.intents_for_run(first.run_id)
    states = {
        intent.symbol: ledger.current_order_state(intent.client_order_id)
        for intent in intents
    }
    assert sorted(states.values(), key=lambda state: state.value) == sorted(
        [OrderState.REJECTED, OrderState.CANCELED],
        key=lambda state: state.value,
    )
    canceled = next(
        intent for intent in intents if states[intent.symbol] is OrderState.CANCELED
    )
    assert ledger.events_for(canceled.client_order_id)[-1].details["broker_posted"] is False
    assert ledger.oldest_run_requiring_reconciliation(
        "paper-account", TradingMode.PAPER
    ) is None

    retry = coordinator.run(plan=selected, now=NOW + timedelta(seconds=1))
    assert retry.status is RunStatus.BLOCKED
    assert "terminal order failure" in retry.message
    assert gateway.calls["submit_order"] == 1

    cleanup = ExecutionCoordinator(
        paper_settings(purpose="reconcile"), ledger, gateway=gateway
    ).run(now=NOW + timedelta(seconds=2))
    assert cleanup.status is RunStatus.SKIPPED_NOT_DUE


def test_reconciled_reject_also_terminalizes_the_unsubmitted_batch_tail() -> None:
    ledger = InMemoryLedger()
    gateway = FakeGateway()
    gateway.submit_state = OrderState.ACCEPTED
    gateway.reconcile_state = OrderState.REJECTED
    result = ExecutionCoordinator(
        paper_settings(), ledger, gateway=gateway
    ).run(plan=plan(), now=NOW)

    assert result.status is RunStatus.BLOCKED
    assert gateway.calls["submit_order"] == 1
    assert sorted(
        (
            ledger.current_order_state(intent.client_order_id)
            for intent in ledger.intents_for_run(result.run_id)
        ),
        key=lambda state: state.value,
    ) == sorted(
        [OrderState.REJECTED, OrderState.CANCELED],
        key=lambda state: state.value,
    )


def test_reconcile_repairs_legacy_rejected_head_and_intent_tail_deadlock() -> None:
    ledger = InMemoryLedger()
    selected = plan()
    decision = build_decision_key(
        selected.strategy_id,
        selected.strategy_version,
        "paper-account",
        TradingMode.PAPER,
        NOW,
    )
    run = RunRecord(
        run_id="legacy-terminal-batch",
        decision_key=decision,
        strategy_id=selected.strategy_id,
        strategy_version=selected.strategy_version,
        account_key="paper-account",
        mode=TradingMode.PAPER,
        purpose=RunPurpose.REBALANCE,
        target_hash=build_target_hash(selected.target_weights),
        created_at=NOW - timedelta(minutes=20),
        status=RunStatus.RECONCILIATION_REQUIRED,
    )
    ledger.create_run(run)
    intents = build_order_intents(
        run_id=run.run_id,
        decision_key=decision,
        plan=selected,
        account=AccountSnapshot(
            "paper-account",
            Decimal("100000"),
            Decimal("100000"),
            Decimal("100000"),
            "active",
            NOW,
        ),
        positions=(),
        open_orders=(),
        assets={
            "AAPL": AssetSnapshot("AAPL", True, True),
            "SPY": AssetSnapshot("SPY", True, True),
        },
        quotes={"AAPL": quote("AAPL"), "SPY": quote("SPY")},
        settings=paper_settings(),
        now=NOW - timedelta(minutes=20),
    )
    ledger.add_intents(intents)
    ledger.append_order_event(
        OrderEvent(
            "legacy-reject",
            intents[0].client_order_id,
            OrderState.REJECTED,
            NOW - timedelta(minutes=19),
        )
    )

    result = ExecutionCoordinator(
        paper_settings(purpose="reconcile"), ledger, gateway=FakeGateway()
    ).run(now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert ledger.current_order_state(intents[1].client_order_id) is OrderState.CANCELED
    assert ledger.oldest_run_requiring_reconciliation(
        "paper-account", TradingMode.PAPER
    ) is None


def test_same_month_satellite_reentry_is_blocked_even_with_manual_approval() -> None:
    class ReentryFactory:
        def construct(self, context):
            selected = plan()
            metadata = dict(selected.metadata)
            metadata["satellite_reentry_approved"] = True
            return PortfolioPlan(
                selected.strategy_id,
                selected.strategy_version,
                selected.as_of,
                selected.target_weights,
                drawdown=context.fresh_drawdown,
                metadata=metadata,
            )

    gateway = FakeGateway()
    risk = FakeRiskState(
        Decimal("105000"),
        de_risked=True,
        de_risked_at=datetime(2026, 7, 1, 13, 0, tzinfo=UTC),
    )
    result = ExecutionCoordinator(
        paper_settings(),
        InMemoryLedger(),
        gateway=gateway,
        paper_risk_state_provider=risk,
    ).run(paper_plan_factory=ReentryFactory(), now=NOW)

    assert result.status is RunStatus.BLOCKED
    assert "next monthly rebalance" in result.message
    assert gateway.calls["submit_order"] == 0


def test_underweight_spy_drawdown_containment_never_buys_core() -> None:
    gateway = FakeGateway()
    gateway.positions = (
        PositionSnapshot("SPY", Decimal("300"), Decimal("100")),
        PositionSnapshot("AAPL", Decimal("200"), Decimal("100")),
    )
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=de_risk_plan(), now=NOW)

    assert result.status is RunStatus.COMPLETED
    assert [(intent.symbol, intent.side) for intent in gateway.submitted] == [
        ("AAPL", OrderSide.SELL)
    ]


def test_post_fill_exposure_breach_is_persisted_and_requires_reconciliation() -> None:
    class NoPositionUpdateGateway(FakeGateway):
        def _apply_fill(self, intent, snapshot):
            self.applied_fills.add(intent.client_order_id)

    class Recorder:
        def __init__(self) -> None:
            self.snapshot = None

        def record_paper_completion(self, snapshot) -> None:
            self.snapshot = snapshot

    gateway = NoPositionUpdateGateway()
    recorder = Recorder()
    result = ExecutionCoordinator(
        paper_settings(),
        InMemoryLedger(),
        gateway=gateway,
        paper_completion_recorder=recorder,
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)

    assert result.status is RunStatus.RECONCILIATION_REQUIRED
    assert "exposure drift" in result.message
    assert recorder.snapshot is not None
    assert recorder.snapshot.exposure_breach is True
    assert recorder.snapshot.max_position_drift == Decimal("0.99")


def test_broker_requested_amount_and_overfill_must_match_frozen_intent() -> None:
    class MismatchedAmountGateway(FakeGateway):
        def _snapshot(self, intent, state):
            snapshot = super()._snapshot(intent, state)
            if snapshot.requested_notional is not None:
                return replace(
                    snapshot,
                    requested_notional=snapshot.requested_notional + Decimal("1"),
                )
            return snapshot

    mismatch = MismatchedAmountGateway()
    mismatch_result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=mismatch
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)
    assert mismatch_result.status is RunStatus.RECONCILIATION_REQUIRED
    assert mismatch.calls["submit_order"] == 1

    class OverfillGateway(FakeGateway):
        def _snapshot(self, intent, state):
            snapshot = super()._snapshot(intent, state)
            if state is OrderState.FILLED:
                return replace(
                    snapshot,
                    filled_quantity=snapshot.filled_quantity + Decimal("1"),
                )
            return snapshot

    overfill = OverfillGateway()
    overfill_result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=overfill
    ).run(plan=plan({"SPY": Decimal("0.99")}), now=NOW)
    assert overfill_result.status is RunStatus.RECONCILIATION_REQUIRED
    assert overfill.calls["submit_order"] == 1


def test_invalid_open_order_quantities_fail_closed_before_submission() -> None:
    gateway = FakeGateway()
    gateway.open_orders = (
        OpenOrderSnapshot(
            "broker-open",
            "manual-order",
            "SPY",
            OrderSide.BUY,
            Decimal("1"),
            Decimal("2"),
            OrderState.PARTIALLY_FILLED,
            NOW,
        ),
    )
    result = ExecutionCoordinator(
        paper_settings(), InMemoryLedger(), gateway=gateway
    ).run(plan=plan(), now=NOW)
    assert result.status is RunStatus.BLOCKED
    assert "invalid open-order" in result.message
    assert gateway.calls["submit_order"] == 0
