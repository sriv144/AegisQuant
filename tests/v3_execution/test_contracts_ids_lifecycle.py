from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.execution.v3 import (
    InMemoryLedger,
    InvalidOrderTransition,
    Ledger,
    OrderEvent,
    OrderIntent,
    OrderSide,
    OrderState,
    PortfolioPlan,
    RunPurpose,
    RunRecord,
    RunStatus,
    RuntimeSettings,
    SettingsValidationError,
    TradingMode,
    build_client_order_id,
    build_decision_key,
    build_operational_key,
    build_target_hash,
    validate_order_transition,
)


NOW = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)


def test_runtime_defaults_to_shadow_and_rejects_live_or_nonpaper_urls() -> None:
    assert RuntimeSettings().mode is TradingMode.SHADOW
    with pytest.raises(SettingsValidationError, match="only shadow and paper"):
        RuntimeSettings(mode="live")
    with pytest.raises(SettingsValidationError, match="exact Alpaca paper endpoint"):
        RuntimeSettings(alpaca_base_url="https://api.alpaca.markets")
    with pytest.raises(SettingsValidationError, match="exact Alpaca paper endpoint"):
        RuntimeSettings(alpaca_base_url="https://proxy.invalid")


def test_paper_gate_reports_every_missing_prerequisite_without_mode_fallback() -> None:
    settings = RuntimeSettings(mode="paper", purpose="rebalance")
    assert settings.mode is TradingMode.PAPER
    assert settings.paper_gate_errors() == (
        "broker execution is not explicitly enabled",
        "kill switch is active",
        "paper execution requires durable PostgreSQL",
        "Alpaca paper credentials are missing",
        "tracked strategy config SHA-256 is missing or invalid",
    )


def test_valid_paper_settings_require_postgres_credentials_and_explicit_enable() -> None:
    settings = RuntimeSettings(
        mode="paper",
        purpose="rebalance",
        execution_enabled=True,
        kill_switch=False,
        database_url="postgresql+psycopg2://user:pass@db/aegis",
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        strategy_config_sha256="a" * 64,
    )
    assert settings.paper_gate_errors() == ()


def test_runtime_commit_sha_is_normalized_and_validated() -> None:
    assert RuntimeSettings(commit_sha="ABCDEF1234567").commit_sha == "abcdef1234567"
    with pytest.raises(SettingsValidationError, match="commit SHA"):
        RuntimeSettings(commit_sha="not-a-git-sha")


def test_postgres_looking_but_malformed_database_urls_do_not_pass_the_gate() -> None:
    settings = RuntimeSettings(
        mode="paper",
        execution_enabled=True,
        kill_switch=False,
        database_url="postgresql+not-a-url",
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        strategy_config_sha256="a" * 64,
    )
    assert "paper execution requires durable PostgreSQL" in settings.paper_gate_errors()


def test_portfolio_plan_is_long_only_normalizes_drawdown_and_freezes_weights() -> None:
    source = {"spy": 0.69, "aapl": "0.30"}
    plan = PortfolioPlan("spy_xsmom_core_satellite", "3.0.0", NOW, source, drawdown="0.07")
    source["spy"] = 0
    assert plan.target_weights == {"AAPL": Decimal("0.30"), "SPY": Decimal("0.69")}
    assert plan.drawdown == Decimal("0.07")
    with pytest.raises(ValueError, match="long-only"):
        PortfolioPlan("s", "v", NOW, {"SPY": -0.1})
    with pytest.raises(ValueError, match="100%"):
        PortfolioPlan("s", "v", NOW, {"SPY": 1.01})


def test_decision_client_and_target_ids_are_deterministic_and_sensitive() -> None:
    key = build_decision_key("strategy", "3.0.0", "acct", TradingMode.PAPER, NOW)
    assert key == "strategy|3.0.0|acct|paper|2026-07"
    first = build_client_order_id(key, "spy", OrderSide.BUY, Decimal("100.00"))
    second = build_client_order_id(key, "SPY", OrderSide.BUY, 100)
    changed = build_client_order_id(key, "SPY", OrderSide.BUY, 101)
    assert first == second
    assert first.startswith("aq3-p-202607-")
    assert len(first) <= 48
    assert changed != first
    assert build_target_hash({"SPY": Decimal("0.690")}) == build_target_hash({"spy": 0.69})
    operational = build_operational_key(
        "strategy", "3.0.0", "acct", TradingMode.PAPER, RunPurpose.EOD, NOW
    )
    assert operational == "strategy|3.0.0|acct|paper|eod|2026-07-01"
    assert operational != key
    with pytest.raises(ValueError, match="rebalance"):
        build_operational_key(
            "strategy", "3.0.0", "acct", TradingMode.PAPER, RunPurpose.REBALANCE, NOW
        )


def _intent(client_order_id: str = "aq3-s-202607-abcdefghijklmnopqrst") -> OrderIntent:
    return OrderIntent(
        client_order_id=client_order_id,
        run_id="run-1",
        decision_key="s|v|a|shadow|2026-07",
        sleeve="core",
        symbol="SPY",
        side=OrderSide.BUY,
        target_weight=Decimal("0.69"),
        arrival_price=Decimal("100"),
        created_at=NOW,
        notional=Decimal("69000"),
    )


def test_order_lifecycle_acceptance_is_not_a_fill_and_terminal_states_are_final() -> None:
    validate_order_transition(OrderState.INTENT, OrderState.ACCEPTED)
    assert not OrderState.ACCEPTED.is_terminal
    validate_order_transition(OrderState.ACCEPTED, OrderState.PARTIALLY_FILLED)
    validate_order_transition(OrderState.PARTIALLY_FILLED, OrderState.FILLED)
    assert OrderState.FILLED.is_terminal
    with pytest.raises(InvalidOrderTransition):
        validate_order_transition(OrderState.FILLED, OrderState.ACCEPTED)


def test_inmemory_ledger_matches_protocol_enforces_lease_and_append_only_events() -> None:
    ledger = InMemoryLedger()
    assert isinstance(ledger, Ledger)
    assert ledger.acquire_lease("acct", TradingMode.SHADOW, "owner-1")
    assert not ledger.acquire_lease("acct", TradingMode.SHADOW, "owner-2")
    ledger.release_lease("acct", TradingMode.SHADOW, "owner-2")
    assert not ledger.acquire_lease("acct", TradingMode.SHADOW, "owner-2")
    ledger.release_lease("acct", TradingMode.SHADOW, "owner-1")
    assert ledger.acquire_lease("acct", TradingMode.SHADOW, "owner-2")

    record = RunRecord(
        run_id="run-1",
        decision_key="s|v|acct|shadow|2026-07",
        strategy_id="s",
        strategy_version="v",
        account_key="acct",
        mode=TradingMode.SHADOW,
        purpose=RunPurpose.REBALANCE,
        target_hash="hash",
        created_at=NOW,
    )
    assert ledger.create_run(record) is record
    assert ledger.create_run(record) is record
    ledger.add_intents((_intent(),))
    ledger.append_order_event(
        OrderEvent("e1", _intent().client_order_id, OrderState.ACCEPTED, NOW)
    )
    assert ledger.current_order_state(_intent().client_order_id) is OrderState.ACCEPTED
    ledger.append_order_event(
        OrderEvent("e2", _intent().client_order_id, OrderState.FILLED, NOW)
    )
    with pytest.raises(InvalidOrderTransition):
        ledger.append_order_event(
            OrderEvent("e3", _intent().client_order_id, OrderState.ACCEPTED, NOW)
        )
    ledger.update_run("run-1", RunStatus.COMPLETED, NOW)
    assert ledger.get_run_by_decision_key(record.decision_key).status is RunStatus.COMPLETED
