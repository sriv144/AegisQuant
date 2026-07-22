"""Typed contracts for the AegisQuant v3 execution boundary.

The module deliberately contains no broker SDK imports.  Shadow execution can
therefore use the same portfolio and order contracts without gaining a code
path capable of submitting an order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class SettingsValidationError(ValueError):
    """Raised when a runtime configuration violates a safety invariant."""


class TradingMode(str, Enum):
    SHADOW = "shadow"
    PAPER = "paper"


class RunPurpose(str, Enum):
    HEALTH = "health"
    EOD = "eod"
    REBALANCE = "rebalance"
    RECONCILE = "reconcile"
    BOOTSTRAP = "bootstrap"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED_NOT_DUE = "skipped_not_due"
    SKIPPED_MARKET_CLOSED = "skipped_market_closed"
    BLOCKED = "blocked"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    FAILED = "failed"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderState(str, Enum):
    INTENT = "intent"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELED,
            OrderState.EXPIRED,
        }


def to_decimal(value: Decimal | int | float | str) -> Decimal:
    """Convert numeric input without inheriting binary float representation."""

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Execution settings with a deliberately closed set of trading modes.

    Paper-specific prerequisites are exposed through ``paper_gate_errors`` so
    the coordinator can persist a BLOCKED run rather than silently switching
    modes.  Invalid modes and non-paper broker URLs are rejected at creation.
    """

    mode: TradingMode | str = TradingMode.SHADOW
    purpose: RunPurpose | str = RunPurpose.HEALTH
    strategy_id: str = "spy_xsmom_core_satellite"
    strategy_version: str = "3.0.0"
    strategy_config_sha256: str = ""
    benchmark: str = "SPY"
    commit_sha: str = "unknown"
    database_url: str = ""
    execution_enabled: bool = False
    kill_switch: bool = True
    alpaca_base_url: str = PAPER_BASE_URL
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    account_key: str = "configured-paper-account"
    quote_max_age_seconds: int = 60
    unresolved_order_minutes: int = 15
    min_trade_notional: Decimal | int | float | str = Decimal("100")
    min_drift_fraction: Decimal | int | float | str = Decimal("0.002")
    adv_limit_fraction: Decimal | int | float | str = Decimal("0.05")
    buying_power_buffer_fraction: Decimal | int | float | str = Decimal("0.005")

    def __post_init__(self) -> None:
        try:
            mode = self.mode if isinstance(self.mode, TradingMode) else TradingMode(self.mode)
        except ValueError as exc:
            raise SettingsValidationError(
                f"unsupported trading mode {self.mode!r}; only shadow and paper are valid"
            ) from exc
        try:
            purpose = (
                self.purpose
                if isinstance(self.purpose, RunPurpose)
                else RunPurpose(self.purpose)
            )
        except ValueError as exc:
            raise SettingsValidationError(f"unsupported run purpose {self.purpose!r}") from exc

        base_url = self.alpaca_base_url.rstrip("/")
        if base_url != PAPER_BASE_URL:
            raise SettingsValidationError(
                "only the exact Alpaca paper endpoint is allowed; live and proxy endpoints are rejected"
            )
        if not self.strategy_id.strip() or not self.strategy_version.strip():
            raise SettingsValidationError("strategy id and version are required")
        if self.benchmark.upper().strip() != "SPY":
            raise SettingsValidationError("v3 benchmark must be SPY")
        commit_sha = self.commit_sha.lower().strip()
        if commit_sha != "unknown" and not re.fullmatch(r"[0-9a-f]{7,64}", commit_sha):
            raise SettingsValidationError("commit SHA must be 7-64 hexadecimal characters")
        if self.quote_max_age_seconds <= 0 or self.unresolved_order_minutes <= 0:
            raise SettingsValidationError("freshness and reconciliation timeouts must be positive")

        min_notional = to_decimal(self.min_trade_notional)
        min_drift = to_decimal(self.min_drift_fraction)
        adv_limit = to_decimal(self.adv_limit_fraction)
        bp_buffer = to_decimal(self.buying_power_buffer_fraction)
        if not all(
            value.is_finite()
            for value in (min_notional, min_drift, adv_limit, bp_buffer)
        ):
            raise SettingsValidationError("numeric execution settings must be finite")
        if not isinstance(self.execution_enabled, bool) or not isinstance(
            self.kill_switch, bool
        ):
            raise SettingsValidationError("execution enable and kill switch must be booleans")
        if min_notional <= 0:
            raise SettingsValidationError("minimum trade notional must be positive")
        if not Decimal("0") < min_drift < Decimal("1"):
            raise SettingsValidationError("minimum drift must be between zero and one")
        if not Decimal("0") < adv_limit <= Decimal("0.05"):
            raise SettingsValidationError("ADV limit must be positive and no greater than 5%")
        if not Decimal("0") <= bp_buffer < Decimal("1"):
            raise SettingsValidationError("buying-power buffer must be in [0, 1)")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "alpaca_base_url", base_url)
        object.__setattr__(self, "benchmark", "SPY")
        object.__setattr__(self, "commit_sha", commit_sha)
        object.__setattr__(self, "min_trade_notional", min_notional)
        object.__setattr__(self, "min_drift_fraction", min_drift)
        object.__setattr__(self, "adv_limit_fraction", adv_limit)
        object.__setattr__(self, "buying_power_buffer_fraction", bp_buffer)

    def paper_gate_errors(self) -> tuple[str, ...]:
        """Return all fail-closed paper prerequisites that are not satisfied."""

        if self.mode is not TradingMode.PAPER:
            return ()
        errors: list[str] = []
        if not self.execution_enabled:
            errors.append("broker execution is not explicitly enabled")
        if self.kill_switch:
            errors.append("kill switch is active")
        errors.extend(self.paper_reconciliation_gate_errors())
        if not re.fullmatch(r"[0-9a-f]{64}", self.strategy_config_sha256.lower()):
            errors.append("tracked strategy config SHA-256 is missing or invalid")
        return tuple(errors)

    def paper_reconciliation_gate_errors(self) -> tuple[str, ...]:
        """Credentials/storage gates needed for read/cancel cleanup.

        Execution enablement and the kill switch intentionally do not appear:
        they prohibit new order POSTs, not reconciliation of existing exposure.
        """

        if self.mode is not TradingMode.PAPER:
            return ()
        errors: list[str] = []
        if not re.match(
            r"^postgresql(?:\+[a-z0-9_]+)?://\S+$", self.database_url.lower()
        ):
            errors.append("paper execution requires durable PostgreSQL")
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            errors.append("Alpaca paper credentials are missing")
        if not self.account_key.strip():
            errors.append("a non-empty account key is required")
        return tuple(errors)


@dataclass(frozen=True, slots=True)
class PortfolioPlan:
    strategy_id: str
    strategy_version: str
    as_of: datetime
    target_weights: Mapping[str, Decimal | int | float | str]
    sleeve: str = "spy_xsmom_core_satellite"
    drawdown: Decimal | int | float | str = Decimal("0")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("portfolio plan timestamp must be timezone-aware")
        weights: dict[str, Decimal] = {}
        for raw_symbol, raw_weight in self.target_weights.items():
            symbol = raw_symbol.upper().strip()
            weight = to_decimal(raw_weight)
            if not symbol:
                raise ValueError("target symbols cannot be empty")
            if not weight.is_finite():
                raise ValueError("target weights must be finite")
            if weight < 0:
                raise ValueError("v3 is long-only; negative target weights are invalid")
            weights[symbol] = weight
        total = sum(weights.values(), Decimal("0"))
        if total > Decimal("1"):
            raise ValueError("target weights cannot exceed 100%")
        drawdown = to_decimal(self.drawdown)
        if not drawdown.is_finite() or not Decimal("0") <= drawdown <= Decimal("1"):
            raise ValueError("drawdown is a positive magnitude between zero and one")
        object.__setattr__(self, "target_weights", MappingProxyType(dict(sorted(weights.items()))))
        object.__setattr__(self, "drawdown", drawdown)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    account_key: str
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    status: str
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    symbol: str
    quantity: Decimal
    market_price: Decimal
    asset_class: str = "us_equity"


@dataclass(frozen=True, slots=True)
class OpenOrderSnapshot:
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    filled_quantity: Decimal
    state: OrderState
    submitted_at: datetime

    @property
    def remaining_quantity(self) -> Decimal:
        return max(Decimal("0"), self.quantity - self.filled_quantity)


@dataclass(frozen=True, slots=True)
class ClockSnapshot:
    is_open: bool
    timestamp: datetime
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True, slots=True)
class CalendarSession:
    session_date: date
    open_at: datetime
    close_at: datetime


@dataclass(frozen=True, slots=True)
class AssetSnapshot:
    symbol: str
    tradable: bool
    fractionable: bool
    asset_class: str = "us_equity"


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    observed_at: datetime
    adv_dollars_30d: Decimal

    @property
    def midpoint(self) -> Decimal:
        if (
            not self.bid_price.is_finite()
            or not self.ask_price.is_finite()
            or self.bid_price <= 0
            or self.ask_price <= 0
        ):
            raise ValueError(f"invalid quote for {self.symbol}")
        return (self.bid_price + self.ask_price) / Decimal("2")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    client_order_id: str
    run_id: str
    decision_key: str
    sleeve: str
    symbol: str
    side: OrderSide
    target_weight: Decimal
    arrival_price: Decimal
    created_at: datetime
    quantity: Decimal | None = None
    notional: Decimal | None = None

    def __post_init__(self) -> None:
        if (self.quantity is None) == (self.notional is None):
            raise ValueError("an order intent must contain exactly one of quantity or notional")
        if self.quantity is not None and (
            not self.quantity.is_finite() or self.quantity <= 0
        ):
            raise ValueError("order quantity must be positive")
        if self.notional is not None and (
            not self.notional.is_finite() or self.notional <= 0
        ):
            raise ValueError("order notional must be positive")
        if not self.arrival_price.is_finite() or self.arrival_price <= 0:
            raise ValueError("arrival price must be finite and positive")
        if not self.target_weight.is_finite() or self.target_weight < 0:
            raise ValueError("target weight must be finite and nonnegative")
        if self.created_at.tzinfo is None:
            raise ValueError("order intent timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    state: OrderState
    requested_quantity: Decimal | None
    requested_notional: Decimal | None
    filled_quantity: Decimal
    filled_average_price: Decimal | None
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: RunStatus
    exit_code: int
    message: str
    decision_key: str
    target_hash: str
    order_client_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
