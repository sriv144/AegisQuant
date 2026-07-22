"""Append-only order lifecycle rules."""

from __future__ import annotations

from .contracts import OrderState


class InvalidOrderTransition(ValueError):
    pass


_ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    OrderState.INTENT: frozenset(
        {
            OrderState.ACCEPTED,
            OrderState.REJECTED,
            OrderState.CANCELED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.ACCEPTED: frozenset(
        {
            OrderState.ACCEPTED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.UNKNOWN,
            OrderState.ACCEPTED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.CANCELED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


def validate_order_transition(previous: OrderState, current: OrderState) -> None:
    if current not in _ALLOWED[previous]:
        raise InvalidOrderTransition(
            f"invalid order lifecycle transition: {previous.value} -> {current.value}"
        )
