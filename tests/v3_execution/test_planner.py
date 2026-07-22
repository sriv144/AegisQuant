from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.execution.v3 import (
    AccountSnapshot,
    AssetSnapshot,
    OpenOrderSnapshot,
    OrderSide,
    OrderState,
    PortfolioPlan,
    PositionSnapshot,
    QuoteSnapshot,
    RuntimeSettings,
    build_order_intents,
)
from src.execution.v3.ids import build_decision_key


NOW = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)
SETTINGS = RuntimeSettings(mode="paper", purpose="rebalance")
DECISION = build_decision_key(
    SETTINGS.strategy_id,
    SETTINGS.strategy_version,
    SETTINGS.account_key,
    SETTINGS.mode,
    NOW,
)


def account() -> AccountSnapshot:
    return AccountSnapshot(
        "acct", Decimal("100000"), Decimal("100000"), Decimal("100000"), "active", NOW
    )


def quote(symbol: str, price: str = "100") -> QuoteSnapshot:
    value = Decimal(price)
    return QuoteSnapshot(
        symbol,
        value - Decimal("0.01"),
        value + Decimal("0.01"),
        NOW,
        Decimal("10000000"),
    )


def make_plan(weights, *, drawdown=Decimal("0"), metadata=None) -> PortfolioPlan:
    return PortfolioPlan(
        SETTINGS.strategy_id,
        SETTINGS.strategy_version,
        NOW,
        weights,
        drawdown=drawdown,
        metadata=metadata or {},
    )


def build(plan, positions=(), open_orders=(), *, fractionable=True):
    symbols = set(plan.target_weights) | {position.symbol for position in positions}
    return build_order_intents(
        run_id="run",
        decision_key=DECISION,
        plan=plan,
        account=account(),
        positions=positions,
        open_orders=open_orders,
        assets={s: AssetSnapshot(s, True, fractionable) for s in symbols},
        quotes={s: quote(s) for s in symbols},
        settings=SETTINGS,
        now=NOW,
    )


def test_effective_quantity_includes_signed_unfilled_open_order_quantity() -> None:
    target = make_plan({"SPY": Decimal("0.01")})  # 10 shares at $100
    positions = (PositionSnapshot("SPY", Decimal("5"), Decimal("100")),)
    open_orders = (
        OpenOrderSnapshot(
            "broker",
            "aq3-p-202607-existingexisting00",
            "SPY",
            OrderSide.BUY,
            Decimal("3"),
            Decimal("1"),
            OrderState.ACCEPTED,
            NOW,
        ),
    )
    intents = build(target, positions, open_orders)
    assert len(intents) == 1
    assert intents[0].side is OrderSide.BUY
    assert intents[0].notional == Decimal("300.00")


def test_minimum_notional_and_twenty_basis_point_drift_are_both_required() -> None:
    assert build(make_plan({"SPY": Decimal("0.0019")})) == ()
    # $200 is exactly 20 bps of $100k and therefore eligible.
    intents = build(make_plan({"SPY": Decimal("0.002")}))
    assert len(intents) == 1
    assert intents[0].notional == Decimal("200.00")


def test_sell_delta_is_capped_at_long_quantity_and_cannot_create_short() -> None:
    positions = (PositionSnapshot("OLD", Decimal("2.5"), Decimal("100")),)
    intents = build(make_plan({}), positions)
    assert len(intents) == 1
    assert intents[0].side is OrderSide.SELL
    assert intents[0].quantity == Decimal("2.500000000")


def test_nonfractionable_assets_use_whole_share_quantity_for_buys() -> None:
    intents = build(make_plan({"SPY": Decimal("0.0125")}), fractionable=False)
    assert len(intents) == 1
    assert intents[0].side is OrderSide.BUY
    assert intents[0].quantity == Decimal("12")
    assert intents[0].notional is None


def test_off_cycle_drawdown_core_uses_fifty_basis_point_no_trade_band() -> None:
    target = make_plan(
        {"SPY": Decimal("0.69")},
        drawdown=Decimal("0.15"),
        metadata={"drawdown_kill": True},
    )
    positions = (PositionSnapshot("SPY", Decimal("693"), Decimal("100")),)

    assert build(target, positions) == ()
