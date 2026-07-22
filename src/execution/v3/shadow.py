"""Self-financing shadow execution with no broker dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Mapping

from .contracts import OrderSide, PortfolioPlan, QuoteSnapshot, to_decimal
from .ids import build_client_order_id


@dataclass(slots=True)
class ShadowPosition:
    symbol: str
    quantity: Decimal
    average_cost: Decimal


@dataclass(slots=True)
class ShadowAccount:
    account_key: str
    cash: Decimal
    positions: dict[str, ShadowPosition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cash = to_decimal(self.cash)
        if not self.cash.is_finite() or self.cash < 0:
            raise ValueError("shadow cash cannot be negative")
        for symbol, position in self.positions.items():
            if (
                position.quantity < 0
                or not position.quantity.is_finite()
                or not position.average_cost.is_finite()
                or position.average_cost < 0
            ):
                raise ValueError(f"invalid long-only shadow position: {symbol}")

    def net_asset_value(self, quotes: Mapping[str, QuoteSnapshot]) -> Decimal:
        nav = self.cash
        for symbol, position in self.positions.items():
            try:
                price = quotes[symbol].midpoint
            except KeyError as exc:
                raise ValueError(f"missing shadow valuation quote for {symbol}") from exc
            nav += position.quantity * price
        return nav


@dataclass(frozen=True, slots=True)
class ShadowFill:
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    arrival_price: Decimal
    fill_price: Decimal
    filled_at: datetime
    transaction_cost: Decimal
    frozen_order_amount: Decimal


@dataclass(frozen=True, slots=True)
class ShadowExecutionResult:
    fills: tuple[ShadowFill, ...]
    ending_nav: Decimal
    ending_cash: Decimal


class ShadowExecutor:
    """Execute a target plan against an isolated hypothetical account."""

    def __init__(
        self,
        *,
        one_way_cost_bps: Decimal | int | float | str = Decimal("5"),
        min_trade_notional: Decimal | int | float | str = Decimal("100"),
        min_drift_fraction: Decimal | int | float | str = Decimal("0.002"),
    ) -> None:
        self.cost_rate = to_decimal(one_way_cost_bps) / Decimal("10000")
        self.min_trade_notional = to_decimal(min_trade_notional)
        self.min_drift_fraction = to_decimal(min_drift_fraction)
        if self.cost_rate < 0:
            raise ValueError("transaction cost cannot be negative")

    def execute(
        self,
        *,
        account: ShadowAccount,
        plan: PortfolioPlan,
        quotes: Mapping[str, QuoteSnapshot],
        decision_key: str,
        now: datetime,
    ) -> ShadowExecutionResult:
        if now.tzinfo is None:
            raise ValueError("execution timestamp must be timezone-aware")
        required = set(plan.target_weights) | set(account.positions)
        missing = required - set(quotes)
        if missing:
            raise ValueError(f"missing shadow quotes: {', '.join(sorted(missing))}")

        starting_nav = account.net_asset_value(quotes)
        if starting_nav <= 0:
            raise ValueError("shadow account NAV must be positive")
        threshold = max(self.min_trade_notional, starting_nav * self.min_drift_fraction)

        deltas: list[tuple[OrderSide, str, Decimal, Decimal]] = []
        for symbol in sorted(required):
            price = quotes[symbol].midpoint
            target_notional = starting_nav * plan.target_weights.get(symbol, Decimal("0"))
            current_qty = account.positions.get(
                symbol, ShadowPosition(symbol, Decimal("0"), Decimal("0"))
            ).quantity
            target_qty = target_notional / price
            delta_qty = target_qty - current_qty
            if abs(delta_qty * price) < threshold:
                continue
            de_risk_active = (
                plan.drawdown >= Decimal("0.15")
                or plan.metadata.get("drawdown_kill") is True
                or plan.metadata.get("de_risk_active") is True
            )
            if de_risk_active and delta_qty > 0:
                # Operational de-risking never increases exposure; a core
                # shortfall remains cash until a later approved rebalance.
                continue
            side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
            deltas.append((side, symbol, abs(delta_qty), price))

        fills: list[ShadowFill] = []
        for side, symbol, requested_qty, arrival_price in sorted(
            deltas, key=lambda item: (item[0] is OrderSide.BUY, item[1])
        ):
            if side is OrderSide.SELL:
                position = account.positions[symbol]
                quantity = min(requested_qty, position.quantity)
                fill_price = arrival_price * (Decimal("1") - self.cost_rate)
                proceeds = quantity * fill_price
                account.cash += proceeds
                position.quantity -= quantity
                if position.quantity <= Decimal("0.000000001"):
                    del account.positions[symbol]
                frozen_amount = quantity
                transaction_cost = quantity * arrival_price - proceeds
            else:
                fill_price = arrival_price * (Decimal("1") + self.cost_rate)
                affordable = (account.cash / fill_price).quantize(
                    Decimal("0.000000001"), rounding=ROUND_DOWN
                )
                quantity = min(requested_qty, affordable)
                if quantity <= 0:
                    continue
                cost = quantity * fill_price
                account.cash -= cost
                current = account.positions.get(symbol)
                if current is None:
                    account.positions[symbol] = ShadowPosition(symbol, quantity, fill_price)
                else:
                    old_cost = current.quantity * current.average_cost
                    current.quantity += quantity
                    current.average_cost = (old_cost + cost) / current.quantity
                frozen_amount = requested_qty * arrival_price
                transaction_cost = cost - quantity * arrival_price

            fills.append(
                ShadowFill(
                    client_order_id=build_client_order_id(
                        decision_key, symbol, side, frozen_amount
                    ),
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    arrival_price=arrival_price,
                    fill_price=fill_price,
                    filled_at=now,
                    transaction_cost=transaction_cost,
                    frozen_order_amount=frozen_amount,
                )
            )

        if account.cash < Decimal("-0.000001"):
            raise AssertionError("shadow executor violated the self-financing cash invariant")
        return ShadowExecutionResult(
            fills=tuple(fills),
            ending_nav=account.net_asset_value(quotes),
            ending_cash=account.cash,
        )
