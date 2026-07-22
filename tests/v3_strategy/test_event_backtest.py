from __future__ import annotations

import pandas as pd
import pytest

from src.execution.v3.ids import build_target_hash
from src.v3.backtest import BacktestDataError, EventDrivenBacktester


def test_target_executes_on_next_session_and_shares_drift_without_smoothing():
    dates = pd.bdate_range("2026-01-02", periods=4)
    prices = pd.DataFrame({"SPY": [100.0, 100.0, 110.0, 121.0]}, index=dates)
    result = EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=0).run(
        prices, {dates[0]: {"SPY": 0.99}}
    )

    assert result.executed_signal_dates == (dates[0],)
    assert result.trades[0].execution_date == dates[1]
    assert result.accounts[0].positions == ()
    assert result.accounts[1].positions[0][1] == pytest.approx(99.0)
    # 99% invested at entry drifts above 99% after SPY appreciates; the engine
    # does not synthesize constant-weight daily returns.
    weight_day_three = result.accounts[2].market_value / result.accounts[2].nav
    assert weight_day_three > 0.99
    assert result.daily_returns.iloc[2] == pytest.approx(0.099)


def test_transaction_costs_are_self_financing_and_monotonic():
    dates = pd.bdate_range("2026-01-02", periods=4)
    prices = pd.DataFrame({"SPY": [100.0, 100.0, 100.0, 100.0]}, index=dates)
    targets = {dates[0]: {"SPY": 0.99}}
    free = EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=0).run(prices, targets)
    costly = EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=15).run(prices, targets)

    assert costly.total_transaction_cost > 0
    assert costly.nav.iloc[-1] < free.nav.iloc[-1]
    assert all(account.cash >= 0 for account in costly.accounts)
    assert costly.one_way_turnover > 0


def test_dividends_and_splits_are_booked_on_actual_shares():
    dates = pd.bdate_range("2026-01-02", periods=4)
    prices = pd.DataFrame({"AAA": [100.0, 100.0, 50.0, 50.0]}, index=dates)
    dividends = pd.DataFrame({"AAA": [0.0, 0.0, 0.0, 1.0]}, index=dates)
    splits = pd.DataFrame({"AAA": [1.0, 1.0, 2.0, 1.0]}, index=dates)
    result = EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=0).run(
        prices,
        {dates[0]: {"AAA": 0.99}},
        dividends=dividends,
        splits=splits,
    )

    assert result.accounts[1].positions == (("AAA", 99.0),)
    assert result.accounts[2].positions == (("AAA", 198.0),)
    assert result.nav.iloc[2] == pytest.approx(result.nav.iloc[1])
    assert result.nav.iloc[3] - result.nav.iloc[2] == pytest.approx(198.0)


def test_delisting_recovery_liquidates_position():
    dates = pd.bdate_range("2026-01-02", periods=4)
    prices = pd.DataFrame({"AAA": [100.0, 100.0, 80.0, 80.0]}, index=dates)
    delistings = pd.DataFrame({"AAA": [None, None, 75.0, None]}, index=dates)
    result = EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=0).run(
        prices, {dates[0]: {"AAA": 0.99}}, delistings=delistings
    )
    assert result.accounts[2].positions == ()
    assert result.ending_positions == {}
    assert result.accounts[2].cash == pytest.approx(100.0 + 99.0 * 75.0)


def test_missing_mark_or_multiple_signals_fail_loudly():
    dates = pd.bdate_range("2026-01-02", periods=4)
    prices = pd.DataFrame({"AAA": [100.0, 100.0, None, 100.0]}, index=dates)
    with pytest.raises(BacktestDataError, match="missing non-positive"):
        EventDrivenBacktester(transaction_cost_bps=0).run(prices, {dates[0]: {"AAA": 0.99}})

    good = prices.fillna(100.0)
    with pytest.raises(BacktestDataError, match="multiple signals"):
        EventDrivenBacktester().run(
            good,
            {
                dates[0] - pd.Timedelta(days=2): {"AAA": 0.5},
                dates[0] - pd.Timedelta(days=1): {"AAA": 0.6},
            },
        )


@pytest.mark.parametrize("bad_weight", [-0.01, float("nan"), float("inf")])
def test_invalid_target_is_rejected_before_zero_weight_filtering(bad_weight):
    dates = pd.bdate_range("2026-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [100.0, 100.0, 100.0]}, index=dates)

    with pytest.raises(BacktestDataError, match="finite and non-negative"):
        EventDrivenBacktester().run(prices, {dates[0]: {"AAA": bad_weight}})


def test_symbol_change_preserves_shares_and_old_symbol_targets_resolve_to_successor():
    dates = pd.bdate_range("2026-01-02", periods=5)
    prices = pd.DataFrame(
        {
            "OLD": [100.0, 100.0, float("nan"), float("nan"), float("nan")],
            "NEW": [100.0, 100.0, 100.0, 105.0, 110.0],
        },
        index=dates,
    )
    changes = pd.DataFrame(
        [{"effective_date": dates[2], "old_symbol": "OLD", "new_symbol": "NEW", "ratio": 1.0}]
    )
    targets = {dates[0]: {"OLD": 0.99}, dates[2]: {"OLD": 0.99}}

    result = EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=0).run(
        prices, targets, symbol_changes=changes
    )

    assert result.accounts[2].positions == (("NEW", 99.0),)
    assert set(result.ending_positions) == {"NEW"}
    assert result.executed_target_hashes == (
        (dates[0], build_target_hash({"OLD": 0.99})),
        (dates[2], build_target_hash({"OLD": 0.99})),
    )


def test_delisting_return_uses_pre_event_mark_and_blocks_reentry():
    dates = pd.bdate_range("2026-01-02", periods=5)
    prices = pd.DataFrame({"AAA": [100.0, 100.0, 80.0, 80.0, 80.0]}, index=dates)
    delisting_returns = pd.DataFrame(
        {"AAA": [None, None, -0.25, None, None]}, index=dates
    )
    result = EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=0).run(
        prices,
        {dates[0]: {"AAA": 0.99}},
        delisting_returns=delisting_returns,
    )
    assert result.accounts[2].cash == pytest.approx(100.0 + 99.0 * 100.0 * 0.75)
    assert result.ending_positions == {}

    with pytest.raises(BacktestDataError, match="delisted symbol"):
        EventDrivenBacktester(initial_cash=10_000, transaction_cost_bps=0).run(
            prices,
            {dates[0]: {"AAA": 0.99}, dates[2]: {"AAA": 0.99}},
            delisting_returns=delisting_returns,
        )


def test_nonfractionable_assets_use_whole_share_floor_without_borrowing():
    dates = pd.bdate_range("2026-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [333.0, 333.0, 333.0]}, index=dates)
    result = EventDrivenBacktester(initial_cash=1_000, transaction_cost_bps=15).run(
        prices,
        {dates[0]: {"AAA": 0.99}},
        fractionable={"AAA": False},
    )
    assert result.accounts[1].positions == (("AAA", 2.0),)
    assert result.accounts[1].cash >= 0
