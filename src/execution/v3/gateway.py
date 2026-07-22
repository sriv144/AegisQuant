"""Fail-closed Alpaca paper gateway.

All SDK calls are isolated here.  Reads either return a typed snapshot or raise
``BrokerReadError``; they never fabricate cash, equity, positions, or quotes.
The concrete gateway always constructs ``TradingClient`` with ``paper=True``
and the exact paper URL.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Callable, Protocol, Sequence, TypeVar, runtime_checkable

import requests

from .contracts import (
    PAPER_BASE_URL,
    AccountSnapshot,
    AssetSnapshot,
    BrokerOrderSnapshot,
    CalendarSession,
    ClockSnapshot,
    OpenOrderSnapshot,
    OrderIntent,
    OrderSide,
    OrderState,
    PositionSnapshot,
    QuoteSnapshot,
    RuntimeSettings,
    to_decimal,
)


class BrokerError(RuntimeError):
    pass


class BrokerReadError(BrokerError):
    pass


class BrokerSubmissionError(BrokerError):
    pass


class BrokerUncertainOutcome(BrokerError):
    """The request may have reached Alpaca; reconciliation is mandatory."""


@runtime_checkable
class AlpacaGateway(Protocol):
    def get_account(self) -> AccountSnapshot: ...

    def get_positions(self) -> tuple[PositionSnapshot, ...]: ...

    def get_open_orders(self) -> tuple[OpenOrderSnapshot, ...]: ...

    def get_order_history(self) -> tuple[BrokerOrderSnapshot, ...]: ...

    def get_clock(self) -> ClockSnapshot: ...

    def get_calendar(self, start: date, end: date) -> tuple[CalendarSession, ...]: ...

    def get_assets(self, symbols: Sequence[str]) -> dict[str, AssetSnapshot]: ...

    def get_latest_quotes(self, symbols: Sequence[str]) -> dict[str, QuoteSnapshot]: ...

    def submit_order(self, intent: OrderIntent) -> BrokerOrderSnapshot: ...

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrderSnapshot | None: ...

    def cancel_order(self, broker_order_id: str) -> None: ...


_T = TypeVar("_T")


def _enum_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise BrokerReadError("Alpaca returned a timezone-naive timestamp")
    return value


def _map_order_state(value: object) -> OrderState:
    text = _enum_text(value)
    aliases = {
        "new": OrderState.ACCEPTED,
        "pending_new": OrderState.ACCEPTED,
        "accepted": OrderState.ACCEPTED,
        "accepted_for_bidding": OrderState.ACCEPTED,
        "calculated": OrderState.ACCEPTED,
        "held": OrderState.ACCEPTED,
        "partially_filled": OrderState.PARTIALLY_FILLED,
        "filled": OrderState.FILLED,
        "rejected": OrderState.REJECTED,
        "canceled": OrderState.CANCELED,
        "expired": OrderState.EXPIRED,
        "done_for_day": OrderState.EXPIRED,
        "stopped": OrderState.CANCELED,
        "suspended": OrderState.UNKNOWN,
        "pending_cancel": OrderState.ACCEPTED,
        "pending_replace": OrderState.ACCEPTED,
        "replaced": OrderState.CANCELED,
    }
    return aliases.get(text, OrderState.UNKNOWN)


class AlpacaPyGateway:
    """alpaca-py implementation with injectable clients for isolated tests."""

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        trading_client: object | None = None,
        data_client: object | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if settings.alpaca_base_url != PAPER_BASE_URL:
            raise ValueError("AlpacaPyGateway only permits the exact paper endpoint")
        self.settings = settings
        self._now = now or (lambda: datetime.now(UTC))
        if trading_client is None or data_client is None:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.trading.client import TradingClient
            except ImportError as exc:  # pragma: no cover - dependency is installed in CI
                raise RuntimeError("alpaca-py is required for the paper gateway") from exc
            trading_client = trading_client or TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=True,
                url_override=PAPER_BASE_URL,
            )
            data_client = data_client or StockHistoricalDataClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
        # alpaca-py retries selected responses by default, including POSTs.
        # A submission must instead return to our client-ID reconciliation path.
        if hasattr(trading_client, "_retry_codes"):
            trading_client._retry_codes = []
        self._trading = trading_client
        self._data = data_client

    def _read(self, operation: str, fn: Callable[[], _T]) -> _T:
        try:
            return fn()
        except BrokerReadError:
            raise
        except Exception as exc:
            raise BrokerReadError(f"Alpaca {operation} failed") from exc

    def get_account(self) -> AccountSnapshot:
        def load() -> AccountSnapshot:
            account = self._trading.get_account()
            raw_id = str(getattr(account, "id", ""))
            if not raw_id:
                raise BrokerReadError("Alpaca account response has no id")
            account_key = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
            observed_at = self._now()
            return AccountSnapshot(
                account_key=account_key,
                equity=to_decimal(account.equity),
                cash=to_decimal(account.cash),
                buying_power=to_decimal(account.buying_power),
                status=_enum_text(account.status),
                observed_at=_aware(observed_at),
            )

        return self._read("account read", load)

    def get_positions(self) -> tuple[PositionSnapshot, ...]:
        def load() -> tuple[PositionSnapshot, ...]:
            positions = self._trading.get_all_positions()
            if positions is None:
                raise BrokerReadError("Alpaca positions response was null")
            return tuple(
                PositionSnapshot(
                    symbol=str(position.symbol).upper(),
                    quantity=to_decimal(position.qty),
                    market_price=to_decimal(position.current_price),
                    asset_class=_enum_text(getattr(position, "asset_class", "")),
                )
                for position in positions
            )

        return self._read("positions read", load)

    def get_open_orders(self) -> tuple[OpenOrderSnapshot, ...]:
        def load() -> tuple[OpenOrderSnapshot, ...]:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            orders = self._trading.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            if orders is None:
                raise BrokerReadError("Alpaca open-orders response was null")
            return tuple(self._to_open_order(order) for order in orders)

        return self._read("open-orders read", load)

    def get_order_history(self) -> tuple[BrokerOrderSnapshot, ...]:
        """Return the broker's closed-and-open order audit window.

        Alpaca limits a single order-list response to 500 rows.  Bootstrap uses
        this read-only snapshot for attribution/audit only; it never infers a
        fill from a missing row and never submits or cancels an order here.
        """

        def load() -> tuple[BrokerOrderSnapshot, ...]:
            from alpaca.common.enums import Sort
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            orders = self._trading.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.ALL,
                    limit=500,
                    direction=Sort.DESC,
                    nested=False,
                )
            )
            if orders is None:
                raise BrokerReadError("Alpaca order-history response was null")
            return tuple(self._to_broker_order(order) for order in orders)

        return self._read("order-history read", load)

    def get_clock(self) -> ClockSnapshot:
        def load() -> ClockSnapshot:
            clock = self._trading.get_clock()
            return ClockSnapshot(
                is_open=bool(clock.is_open),
                timestamp=_aware(clock.timestamp),
                next_open=_aware(clock.next_open),
                next_close=_aware(clock.next_close),
            )

        return self._read("clock read", load)

    def get_calendar(self, start: date, end: date) -> tuple[CalendarSession, ...]:
        def load() -> tuple[CalendarSession, ...]:
            from alpaca.trading.requests import GetCalendarRequest

            rows = self._trading.get_calendar(
                filters=GetCalendarRequest(start=start, end=end)
            )
            if rows is None:
                raise BrokerReadError("Alpaca calendar response was null")
            sessions: list[CalendarSession] = []
            for row in rows:
                session_date = row.date
                open_at = row.open
                close_at = row.close
                if isinstance(open_at, str) or isinstance(close_at, str):
                    raise BrokerReadError("Alpaca calendar returned unparsed timestamps")
                sessions.append(
                    CalendarSession(
                        session_date=session_date,
                        open_at=_aware(open_at),
                        close_at=_aware(close_at),
                    )
                )
            return tuple(sessions)

        return self._read("calendar read", load)

    def get_assets(self, symbols: Sequence[str]) -> dict[str, AssetSnapshot]:
        def load() -> dict[str, AssetSnapshot]:
            result: dict[str, AssetSnapshot] = {}
            for symbol in sorted(set(symbols)):
                asset = self._trading.get_asset(symbol)
                result[symbol] = AssetSnapshot(
                    symbol=symbol,
                    tradable=bool(asset.tradable),
                    fractionable=bool(asset.fractionable),
                    asset_class=_enum_text(getattr(asset, "asset_class", "")),
                )
            return result

        return self._read("asset read", load)

    def get_latest_quotes(self, symbols: Sequence[str]) -> dict[str, QuoteSnapshot]:
        def load() -> dict[str, QuoteSnapshot]:
            from alpaca.data.enums import Adjustment, DataFeed
            from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
            from alpaca.data.timeframe import TimeFrame

            unique = sorted(set(symbols))
            if not unique:
                return {}
            quotes = self._data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=unique, feed=DataFeed.IEX)
            )
            now = self._now()
            bars = self._data.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=unique,
                    timeframe=TimeFrame.Day,
                    start=now - timedelta(days=50),
                    end=now,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX,
                )
            )
            result: dict[str, QuoteSnapshot] = {}
            for symbol in unique:
                if symbol not in quotes:
                    raise BrokerReadError(f"Alpaca returned no latest quote for {symbol}")
                quote = quotes[symbol]
                symbol_bars = list(bars[symbol])[-30:]
                if len(symbol_bars) < 20:
                    raise BrokerReadError(f"insufficient ADV history for {symbol}")
                dollar_volumes = [
                    to_decimal(bar.volume) * to_decimal(bar.vwap or bar.close)
                    for bar in symbol_bars
                ]
                adv = sum(dollar_volumes, Decimal("0")) / Decimal(len(dollar_volumes))
                result[symbol] = QuoteSnapshot(
                    symbol=symbol,
                    bid_price=to_decimal(quote.bid_price),
                    ask_price=to_decimal(quote.ask_price),
                    observed_at=_aware(quote.timestamp),
                    adv_dollars_30d=adv,
                )
            return result

        return self._read("quote/ADV read", load)

    def submit_order(self, intent: OrderIntent) -> BrokerOrderSnapshot:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import OrderType, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=intent.symbol,
            qty=None if intent.quantity is None else float(intent.quantity),
            notional=None if intent.notional is None else float(intent.notional),
            side=AlpacaSide.BUY if intent.side is OrderSide.BUY else AlpacaSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=intent.client_order_id,
            extended_hours=False,
        )
        try:
            order = self._trading.submit_order(order_data=request)
        except (
            TimeoutError,
            ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            raise BrokerUncertainOutcome(
                f"submission outcome unknown for {intent.client_order_id}"
            ) from exc
        except Exception as exc:
            # Only deterministic request/auth/validation failures are known not
            # to have produced an order.  Everything else is reconciled by the
            # client ID because an unfamiliar transport/API exception may have
            # happened after Alpaca accepted the POST.
            if getattr(exc, "status_code", None) in {400, 401, 403, 404, 405, 422}:
                raise BrokerSubmissionError(
                    f"Alpaca rejected submission for {intent.client_order_id}"
                ) from exc
            raise BrokerUncertainOutcome(
                f"submission outcome unknown for {intent.client_order_id}"
            ) from exc
        return self._to_broker_order(order)

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrderSnapshot | None:
        try:
            order = self._trading.get_order_by_client_id(client_order_id)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 404:
                return None
            raise BrokerReadError(
                f"Alpaca order reconciliation failed for {client_order_id}"
            ) from exc
        return None if order is None else self._to_broker_order(order)

    def cancel_order(self, broker_order_id: str) -> None:
        try:
            self._trading.cancel_order_by_id(broker_order_id)
        except Exception as exc:
            # A cancel timeout/error says nothing about the order's terminal
            # state.  The caller must query it again rather than assuming the
            # cancel succeeded or failed.
            raise BrokerUncertainOutcome(
                f"cancel outcome unknown for {broker_order_id}"
            ) from exc

    def _to_open_order(self, order: object) -> OpenOrderSnapshot:
        submitted_at = _aware(getattr(order, "submitted_at"))
        return OpenOrderSnapshot(
            broker_order_id=str(getattr(order, "id")),
            client_order_id=str(getattr(order, "client_order_id", "")),
            symbol=str(getattr(order, "symbol")).upper(),
            side=OrderSide(_enum_text(getattr(order, "side"))),
            quantity=to_decimal(getattr(order, "qty") or 0),
            filled_quantity=to_decimal(getattr(order, "filled_qty") or 0),
            state=_map_order_state(getattr(order, "status")),
            submitted_at=submitted_at,
        )

    def _to_broker_order(self, order: object) -> BrokerOrderSnapshot:
        observed_at = self._now()
        return BrokerOrderSnapshot(
            broker_order_id=str(getattr(order, "id")),
            client_order_id=str(getattr(order, "client_order_id")),
            symbol=str(getattr(order, "symbol")).upper(),
            side=OrderSide(_enum_text(getattr(order, "side"))),
            state=_map_order_state(getattr(order, "status")),
            requested_quantity=(
                None
                if getattr(order, "qty", None) is None
                else to_decimal(getattr(order, "qty"))
            ),
            requested_notional=(
                None
                if getattr(order, "notional", None) is None
                else to_decimal(getattr(order, "notional"))
            ),
            filled_quantity=to_decimal(getattr(order, "filled_qty", 0) or 0),
            filled_average_price=(
                None
                if getattr(order, "filled_avg_price", None) is None
                else to_decimal(getattr(order, "filled_avg_price"))
            ),
            observed_at=_aware(observed_at),
        )


__all__ = [
    "AlpacaGateway",
    "AlpacaPyGateway",
    "BrokerError",
    "BrokerReadError",
    "BrokerSubmissionError",
    "BrokerUncertainOutcome",
]
