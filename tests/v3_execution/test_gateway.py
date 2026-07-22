from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.execution.v3 import (
    AlpacaPyGateway,
    BrokerReadError,
    BrokerSubmissionError,
    BrokerUncertainOutcome,
    OrderIntent,
    OrderSide,
    RuntimeSettings,
)


NOW = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)


def settings() -> RuntimeSettings:
    return RuntimeSettings(
        mode="paper",
        purpose="rebalance",
        execution_enabled=True,
        kill_switch=False,
        database_url="postgresql://db/aegis",
        alpaca_api_key="key",
        alpaca_secret_key="secret",
    )


class TradingStub:
    def __init__(self) -> None:
        self.positions = []
        self.orders = []
        self.last_order_filter = None
        self.submit_exception = None
        self.cancel_exception = None

    def get_account(self):
        return SimpleNamespace(
            id="raw-account-id",
            equity="100000",
            cash="99000",
            buying_power="99000",
            status="ACTIVE",
        )

    def get_all_positions(self):
        return self.positions

    def get_orders(self, filter):
        self.last_order_filter = filter
        return self.orders

    def submit_order(self, order_data):
        if self.submit_exception:
            raise self.submit_exception
        return SimpleNamespace(
            id="broker-id",
            client_order_id=order_data.client_order_id,
            symbol=order_data.symbol,
            side=order_data.side,
            status="new",
            qty=order_data.qty,
            notional=order_data.notional,
            filled_qty="0",
            filled_avg_price=None,
        )

    def cancel_order_by_id(self, broker_order_id):
        if self.cancel_exception:
            raise self.cancel_exception


class DataStub:
    pass


class RetryableServerError(Exception):
    status_code = 504


class DeterministicValidationError(Exception):
    status_code = 422


def test_gateway_account_and_empty_positions_are_typed_successes() -> None:
    gateway = AlpacaPyGateway(
        settings(), trading_client=TradingStub(), data_client=DataStub(), now=lambda: NOW
    )
    account = gateway.get_account()
    assert account.account_key != "raw-account-id"
    assert account.equity == Decimal("100000")
    assert gateway.get_positions() == ()


def test_gateway_null_positions_fail_closed_instead_of_returning_empty() -> None:
    trading = TradingStub()
    trading.positions = None
    gateway = AlpacaPyGateway(
        settings(), trading_client=trading, data_client=DataStub(), now=lambda: NOW
    )
    with pytest.raises(BrokerReadError, match="positions"):
        gateway.get_positions()


def test_gateway_order_history_is_read_only_typed_and_requests_all_orders() -> None:
    trading = TradingStub()
    trading.orders = [
        SimpleNamespace(
            id="historical-order",
            client_order_id="aq3-p-202606-abcdefghijklmnopqrst",
            symbol="spy",
            side="buy",
            status="filled",
            qty="1.5",
            notional=None,
            filled_qty="1.5",
            filled_avg_price="100.25",
        )
    ]
    gateway = AlpacaPyGateway(
        settings(), trading_client=trading, data_client=DataStub(), now=lambda: NOW
    )

    history = gateway.get_order_history()
    assert len(history) == 1
    assert history[0].symbol == "SPY"
    assert history[0].filled_quantity == Decimal("1.5")
    assert str(trading.last_order_filter.status).lower().endswith("all")
    assert trading.last_order_filter.limit == 500


def test_gateway_null_order_history_fails_closed() -> None:
    trading = TradingStub()
    trading.orders = None
    gateway = AlpacaPyGateway(
        settings(), trading_client=trading, data_client=DataStub(), now=lambda: NOW
    )
    with pytest.raises(BrokerReadError, match="order-history"):
        gateway.get_order_history()


def test_gateway_timeout_is_an_uncertain_outcome_not_a_rejection() -> None:
    trading = TradingStub()
    trading.submit_exception = TimeoutError("timed out")
    gateway = AlpacaPyGateway(
        settings(), trading_client=trading, data_client=DataStub(), now=lambda: NOW
    )
    intent = OrderIntent(
        client_order_id="aq3-p-202607-abcdefghijklmnopqrst",
        run_id="run",
        decision_key="s|v|a|paper|2026-07",
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.69"),
        arrival_price=Decimal("100"),
        created_at=NOW,
        notional=Decimal("1000"),
    )
    with pytest.raises(BrokerUncertainOutcome):
        gateway.submit_order(intent)


def test_gateway_retryable_http_failure_is_uncertain_and_sdk_retries_are_disabled() -> None:
    trading = TradingStub()
    trading._retry_codes = [429, 504]
    trading.submit_exception = RetryableServerError("gateway timeout")
    gateway = AlpacaPyGateway(
        settings(), trading_client=trading, data_client=DataStub(), now=lambda: NOW
    )
    assert trading._retry_codes == []
    intent = OrderIntent(
        client_order_id="aq3-p-202607-abcdefghijklmnopqrst",
        run_id="run",
        decision_key="s|v|a|paper|2026-07",
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.69"),
        arrival_price=Decimal("100"),
        created_at=NOW,
        notional=Decimal("1000"),
    )
    with pytest.raises(BrokerUncertainOutcome):
        gateway.submit_order(intent)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("unclassified SDK failure"), BrokerUncertainOutcome),
        (DeterministicValidationError("bad request"), BrokerSubmissionError),
    ],
)
def test_gateway_only_classifies_proven_validation_failures_as_rejections(error, expected) -> None:
    trading = TradingStub()
    trading.submit_exception = error
    gateway = AlpacaPyGateway(
        settings(), trading_client=trading, data_client=DataStub(), now=lambda: NOW
    )
    intent = OrderIntent(
        client_order_id="aq3-p-202607-abcdefghijklmnopqrst",
        run_id="run",
        decision_key="s|v|a|paper|2026-07",
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.69"),
        arrival_price=Decimal("100"),
        created_at=NOW,
        notional=Decimal("1000"),
    )
    with pytest.raises(expected):
        gateway.submit_order(intent)


def test_cancel_transport_failure_is_uncertain_and_requires_reconciliation() -> None:
    trading = TradingStub()
    trading.cancel_exception = TimeoutError("cancel timed out")
    gateway = AlpacaPyGateway(
        settings(), trading_client=trading, data_client=DataStub(), now=lambda: NOW
    )
    with pytest.raises(BrokerUncertainOutcome):
        gateway.cancel_order("broker-id")
