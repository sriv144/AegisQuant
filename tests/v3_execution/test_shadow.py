from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.execution.v3 import (
    ExecutionCoordinator,
    InMemoryLedger,
    PortfolioPlan,
    QuoteSnapshot,
    RunPurpose,
    RunStatus,
    RuntimeSettings,
    ShadowAccount,
    ShadowExecutor,
    ShadowPosition,
    TradingMode,
)


NOW = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)


def quote(symbol: str, price: str = "100") -> QuoteSnapshot:
    mid = Decimal(price)
    return QuoteSnapshot(
        symbol=symbol,
        bid_price=mid - Decimal("0.01"),
        ask_price=mid + Decimal("0.01"),
        observed_at=NOW,
        adv_dollars_30d=Decimal("10000000"),
    )


class ExplodingGateway:
    calls = 0

    def __getattribute__(self, name: str):
        if name.startswith("get_") or name in {"submit_order", "cancel_order"}:
            object.__setattr__(self, "calls", object.__getattribute__(self, "calls") + 1)
            raise AssertionError("shadow mode reached the broker boundary")
        return object.__getattribute__(self, name)


def test_shadow_coordinator_has_structural_zero_broker_call_invariant() -> None:
    settings = RuntimeSettings(mode="shadow", purpose="rebalance", account_key="shadow-1")
    gateway = ExplodingGateway()
    ledger = InMemoryLedger()
    account = ShadowAccount("shadow-1", Decimal("100000"))
    plan = PortfolioPlan(
        settings.strategy_id,
        settings.strategy_version,
        NOW,
        {"SPY": Decimal("0.69"), "AAPL": Decimal("0.30")},
    )
    quotes = {"SPY": quote("SPY"), "AAPL": quote("AAPL", "200")}
    result = ExecutionCoordinator(settings, ledger, gateway=gateway).run(
        plan=plan,
        now=NOW,
        shadow_account=account,
        shadow_quotes=quotes,
    )
    assert result.status is RunStatus.COMPLETED
    assert result.exit_code == 0
    assert gateway.calls == 0
    assert set(account.positions) == {"AAPL", "SPY"}
    assert account.cash >= 0
    assert Decimal(result.metadata["ending_nav"]) < Decimal("100000")


def test_shadow_executor_sells_before_buys_and_remains_self_financing() -> None:
    executor = ShadowExecutor(one_way_cost_bps=5)
    account = ShadowAccount(
        "shadow-1",
        Decimal("0"),
        {"OLD": ShadowPosition("OLD", Decimal("1000"), Decimal("90"))},
    )
    plan = PortfolioPlan("s", "v", NOW, {"SPY": Decimal("0.99")})
    quotes = {"OLD": quote("OLD"), "SPY": quote("SPY")}
    result = executor.execute(
        account=account,
        plan=plan,
        quotes=quotes,
        decision_key="s|v|shadow-1|shadow|2026-07",
        now=NOW,
    )
    assert [fill.side.value for fill in result.fills] == ["sell", "buy"]
    assert "OLD" not in account.positions
    assert account.cash >= 0
    assert result.ending_nav < Decimal("100000")


def test_completed_monthly_shadow_decision_is_idempotent() -> None:
    settings = RuntimeSettings(mode=TradingMode.SHADOW, purpose=RunPurpose.REBALANCE)
    ledger = InMemoryLedger()
    account = ShadowAccount(settings.account_key, Decimal("100000"))
    plan = PortfolioPlan(settings.strategy_id, settings.strategy_version, NOW, {"SPY": 0.99})
    quotes = {"SPY": quote("SPY")}
    coordinator = ExecutionCoordinator(settings, ledger)
    first = coordinator.run(
        plan=plan, now=NOW, shadow_account=account, shadow_quotes=quotes
    )
    cash_after_first = account.cash
    second = coordinator.run(
        plan=plan, now=NOW, shadow_account=account, shadow_quotes=quotes
    )
    assert first.status is RunStatus.COMPLETED
    assert second.status is RunStatus.SKIPPED_NOT_DUE
    assert second.run_id == first.run_id
    assert account.cash == cash_after_first


def test_drawdown_plan_blocks_non_spy_satellite_targets() -> None:
    settings = RuntimeSettings(mode="shadow", purpose="rebalance")
    account = ShadowAccount(settings.account_key, Decimal("100000"))
    plan = PortfolioPlan(
        settings.strategy_id,
        settings.strategy_version,
        NOW,
        {"SPY": 0.69, "AAPL": 0.30},
        drawdown=0.15,
    )
    result = ExecutionCoordinator(settings, InMemoryLedger()).run(
        plan=plan,
        now=NOW,
        shadow_account=account,
        shadow_quotes={"SPY": quote("SPY"), "AAPL": quote("AAPL")},
    )
    assert result.status is RunStatus.BLOCKED
    assert result.exit_code == 2
    assert not account.positions


def test_shadow_account_namespace_mismatch_is_blocked() -> None:
    settings = RuntimeSettings(mode="shadow", purpose="rebalance", account_key="expected")
    result = ExecutionCoordinator(settings, InMemoryLedger()).run(
        plan=PortfolioPlan(settings.strategy_id, settings.strategy_version, NOW, {"SPY": 0.99}),
        now=NOW,
        shadow_account=ShadowAccount("different", Decimal("100000")),
        shadow_quotes={"SPY": quote("SPY")},
    )
    assert result.status is RunStatus.BLOCKED
    assert "account key" in result.message
