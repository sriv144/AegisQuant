from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.v3.data import DataManifest
from src.v3.portfolio import DataFailure, PortfolioConstructor, SecurityMetadata

from .conftest import make_portfolio_inputs


def _diagnostics(plan):
    return dict(plan.diagnostics)


def test_constructor_builds_locked_69_30_1_portfolio(portfolio_inputs):
    plan = PortfolioConstructor().construct(portfolio_inputs)
    weights = dict(plan.target_weights)

    assert weights["SPY"] == pytest.approx(0.69)
    assert sum(weight for symbol, weight in weights.items() if symbol != "SPY") == pytest.approx(0.30)
    assert plan.cash_weight == pytest.approx(0.01)
    assert plan.invested_weight == pytest.approx(0.99)
    assert len(plan.selected_symbols) == 30
    assert max(weight for symbol, weight in weights.items() if symbol != "SPY") <= 0.02
    assert 0.90 <= plan.portfolio_beta <= 1.10
    assert plan.tracking_error <= 0.06
    assert plan.max_active_sector_deviation <= 0.05
    assert plan.promotable
    assert _diagnostics(plan)["construction"] == "normal"


def test_signal_uses_252_sessions_ending_before_21_session_skip(portfolio_inputs):
    constructor = PortfolioConstructor()
    baseline = constructor.construct(portfolio_inputs)
    baseline_signal = next(signal for signal in baseline.signals if signal.symbol == "S399")

    changed_prices = portfolio_inputs.total_return_prices.copy(deep=True)
    changed_prices.loc[changed_prices.index[-21]:, "S399"] *= np.linspace(1.0, 50.0, 21)
    changed = constructor.construct(replace(portfolio_inputs, total_return_prices=changed_prices))
    changed_signal = next(signal for signal in changed.signals if signal.symbol == "S399")

    assert changed_signal.momentum_return == pytest.approx(baseline_signal.momentum_return)
    assert changed_signal.path_smoothness == pytest.approx(baseline_signal.path_smoothness)
    assert changed_signal.rank == baseline_signal.rank


def test_higher_adv_share_class_wins_issuer_deduplication(portfolio_inputs):
    prices = portfolio_inputs.total_return_prices.copy(deep=True)
    prices["S399.B"] = prices["S399"]
    duplicate = SecurityMetadata(
        symbol="S399.B",
        issuer_id="ISSUER-399",
        sector="Health Care",
        adv_30d=9_000_000.0,
    )
    plan = PortfolioConstructor().construct(
        replace(portfolio_inputs, total_return_prices=prices, securities=portfolio_inputs.securities + (duplicate,))
    )
    assert "S399.B" in {signal.symbol for signal in plan.signals}
    assert "S399" not in {signal.symbol for signal in plan.signals}
    assert not ({"S399", "S399.B"} <= set(plan.selected_symbols))


def test_retention_band_keeps_incumbent_ranked_between_10_and_20_percent(portfolio_inputs):
    constructor = PortfolioConstructor()
    initial = constructor.construct(portfolio_inputs)
    incumbent = next(signal for signal in initial.signals if 0.14 <= signal.percentile <= 0.16)
    assert incumbent.symbol not in initial.selected_symbols

    retained = constructor.construct(replace(portfolio_inputs, current_holdings=frozenset({incumbent.symbol})))
    retained_signal = next(signal for signal in retained.signals if signal.symbol == incumbent.symbol)
    assert retained_signal.retained
    assert incumbent.symbol in retained.selected_symbols
    assert len(retained.selected_symbols) == 30


def test_target_hash_is_deterministic_and_changes_with_strategy_state(portfolio_inputs):
    constructor = PortfolioConstructor()
    first = constructor.construct(portfolio_inputs)
    second = constructor.construct(portfolio_inputs)
    assert first.target_sha256 == second.target_sha256
    assert first.weight_sha256 == second.weight_sha256
    assert first.target_weights == second.target_weights

    later = constructor.construct(replace(portfolio_inputs, signal_date=portfolio_inputs.signal_date + pd.Timedelta(days=1)))
    assert later.target_weights == first.target_weights
    assert later.weight_sha256 == first.weight_sha256
    assert later.target_sha256 != first.target_sha256


def test_missing_price_coverage_fails_closed_instead_of_buying_spy(portfolio_inputs):
    prices = portfolio_inputs.total_return_prices.drop(columns=[f"S{i:03d}" for i in range(20)])
    with pytest.raises(DataFailure) as error:
        PortfolioConstructor().construct(replace(portfolio_inputs, total_return_prices=prices))
    assert error.value.code == "insufficient_price_coverage"


def test_lookahead_manifest_fails_closed(portfolio_inputs):
    manifest = portfolio_inputs.manifests[0]
    future = DataManifest(
        dataset=manifest.dataset,
        source=manifest.source,
        availability_at=manifest.availability_at + timedelta(days=1),
        frozen_at=manifest.frozen_at + timedelta(days=1),
        row_count=manifest.row_count,
        coverage=manifest.coverage,
        sha256=manifest.sha256,
        tier=manifest.tier,
        warnings=manifest.warnings,
    )
    manifests = (future,) + portfolio_inputs.manifests[1:]
    with pytest.raises(DataFailure) as error:
        PortfolioConstructor().construct(replace(portfolio_inputs, manifests=manifests))
    assert error.value.code == "lookahead_manifest"


def test_honest_execution_quote_after_month_end_signal_is_not_lookahead(portfolio_inputs):
    signal_manifest = portfolio_inputs.manifests[0]
    execution_quote = DataManifest(
        dataset="quotes",
        source="arrival-quote-fixture",
        availability_at=signal_manifest.availability_at + timedelta(days=1),
        frozen_at=signal_manifest.frozen_at + timedelta(days=1),
        row_count=30,
        coverage=1.0,
        sha256="f" * 64,
        tier=signal_manifest.tier,
    )
    plan = PortfolioConstructor().construct(
        replace(portfolio_inputs, manifests=portfolio_inputs.manifests + (execution_quote,))
    )
    assert plan.cash_weight == pytest.approx(0.01)


def test_extreme_tracking_error_steps_satellite_to_spy_fallback():
    # A common +/-20% active shock cannot fit below the 6% TE cap even at a
    # five-point satellite allocation; the valid deterministic fallback is SPY.
    common = np.where(np.arange(300) % 2 == 0, 0.20, -0.20)
    inputs = make_portfolio_inputs(common_active_return=common)
    plan = PortfolioConstructor().construct(inputs)
    assert plan.target_weights == (("SPY", 0.99),)
    assert plan.cash_weight == pytest.approx(0.01)
    assert plan.selected_symbols == ()
    assert _diagnostics(plan)["construction"] == "spy_risk_fallback"
    assert plan.tracking_error < 0.06


def test_moderate_tracking_error_scales_satellite_in_five_point_steps():
    common = np.where(np.arange(300) % 2 == 0, 0.02, -0.02)
    plan = PortfolioConstructor().construct(make_portfolio_inputs(common_active_return=common))
    diagnostics = _diagnostics(plan)
    assert diagnostics["construction"] == "satellite_scaled"
    risk_budget = float(diagnostics["risk_budget_satellite_weight"])
    assert risk_budget in {0.05, 0.10, 0.15, 0.20, 0.25}
    assert plan.tracking_error <= 0.06
    assert 0.69 < dict(plan.target_weights)["SPY"] < 0.99


def test_drawdown_de_risk_holds_released_satellite_as_cash(portfolio_inputs):
    plan = PortfolioConstructor().construct(replace(portfolio_inputs, current_drawdown=0.15))
    assert plan.target_weights == (("SPY", 0.69),)
    assert plan.cash_weight == pytest.approx(0.31)
    assert plan.selected_symbols == ()
    assert _diagnostics(plan)["construction"] == "drawdown_de_risk"


def test_drawdown_de_risk_remains_available_during_signal_data_outage(portfolio_inputs):
    plan = PortfolioConstructor().construct(
        replace(
            portfolio_inputs,
            current_drawdown=0.15,
            total_return_prices=portfolio_inputs.total_return_prices.iloc[0:0],
            securities=(),
            benchmark_sector_weights={},
            manifests=(),
        )
    )
    assert plan.target_weights == (("SPY", 0.69),)
    assert plan.cash_weight == pytest.approx(0.31)
    assert plan.promotable is False


def test_prior_de_risk_requires_both_recovery_and_manual_approval(portfolio_inputs):
    blocked = PortfolioConstructor().construct(
        replace(portfolio_inputs, current_drawdown=0.05, prior_de_risked=True, satellite_reentry_approved=False)
    )
    approved = PortfolioConstructor().construct(
        replace(portfolio_inputs, current_drawdown=0.05, prior_de_risked=True, satellite_reentry_approved=True)
    )
    assert blocked.cash_weight == pytest.approx(0.31)
    assert approved.cash_weight == pytest.approx(0.01)
    assert len(approved.selected_symbols) == 30


def test_missing_manifests_allows_research_but_marks_plan_non_promotable(portfolio_inputs):
    plan = PortfolioConstructor().construct(replace(portfolio_inputs, manifests=()))
    assert plan.promotable is False
    assert plan.promotion_blockers == ("missing:data_manifests",)
