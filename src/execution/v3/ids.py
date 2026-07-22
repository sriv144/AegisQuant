"""Deterministic, provenance-bearing identifiers for v3 execution."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from .contracts import OrderSide, RunPurpose, TradingMode, to_decimal


def build_decision_key(
    strategy_id: str,
    strategy_version: str,
    account_key: str,
    mode: TradingMode,
    decision_time: datetime,
) -> str:
    if decision_time.tzinfo is None:
        raise ValueError("decision time must be timezone-aware")
    return "|".join(
        (
            strategy_id.strip(),
            strategy_version.strip(),
            account_key.strip(),
            mode.value,
            decision_time.strftime("%Y-%m"),
        )
    )


def build_risk_decision_key(
    strategy_id: str,
    strategy_version: str,
    account_key: str,
    mode: TradingMode,
    decision_time: datetime,
) -> str:
    """Daily idempotency key for an off-cycle drawdown containment run."""

    if decision_time.tzinfo is None:
        raise ValueError("decision time must be timezone-aware")
    return "|".join(
        (
            strategy_id.strip(),
            strategy_version.strip(),
            account_key.strip(),
            mode.value,
            "risk",
            decision_time.strftime("%Y-%m-%d"),
        )
    )


def _canonical_amount(amount: Decimal | int | float | str) -> str:
    value = to_decimal(amount)
    if value <= 0:
        raise ValueError("frozen order amount must be positive")
    return format(value.normalize(), "f")


def build_client_order_id(
    decision_key: str,
    symbol: str,
    side: OrderSide,
    frozen_order_amount: Decimal | int | float | str,
) -> str:
    """Return an Alpaca-safe ID that is stable for an identical intent."""

    parts = decision_key.split("|")
    if len(parts) == 5:
        year_month = parts[4].replace("-", "")
    elif len(parts) == 6 and parts[4] == "risk":
        year_month = parts[5][:7].replace("-", "")
    else:
        raise ValueError("decision key does not have the v3 canonical shape")
    mode = TradingMode(parts[3])
    payload = "|".join(
        (
            decision_key,
            symbol.upper().strip(),
            side.value,
            _canonical_amount(frozen_order_amount),
        )
    ).encode("utf-8")
    digest = base64.b32encode(hashlib.sha256(payload).digest()).decode("ascii")
    return f"aq3-{mode.value[0]}-{year_month}-{digest[:20].lower()}"


def build_operational_key(
    strategy_id: str,
    strategy_version: str,
    account_key: str,
    mode: TradingMode,
    purpose: RunPurpose,
    operation_time: datetime,
) -> str:
    """Daily idempotency key for non-rebalance operational runs.

    Operational probes must never reserve or conflict with the canonical
    monthly rebalance decision key.
    """

    if purpose is RunPurpose.REBALANCE:
        raise ValueError("rebalance must use build_decision_key")
    if operation_time.tzinfo is None:
        raise ValueError("operation time must be timezone-aware")
    return "|".join(
        (
            strategy_id.strip(),
            strategy_version.strip(),
            account_key.strip(),
            mode.value,
            purpose.value,
            operation_time.strftime("%Y-%m-%d"),
        )
    )


def build_target_hash(target_weights: Mapping[str, Decimal | int | float | str]) -> str:
    canonical = {
        symbol.upper().strip(): _canonical_weight(weight)
        for symbol, weight in sorted(target_weights.items())
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_weight(weight: Decimal | int | float | str) -> str:
    value = to_decimal(weight)
    if value < 0:
        raise ValueError("target weights cannot be negative")
    return format(value.normalize(), "f")
