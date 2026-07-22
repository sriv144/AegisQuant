from __future__ import annotations

import numpy as np
import pytest

from src.v3.metrics import (
    compute_spy_relative_metrics,
    deflated_sharpe_ratio,
    max_drawdown_magnitude,
    probability_backtest_overfit,
    probability_sharpe_ratio,
)


def test_spy_relative_metrics_use_excess_returns_and_positive_drawdown():
    rng = np.random.default_rng(7)
    spy = rng.normal(0.00035, 0.01, 756)
    portfolio = 0.98 * spy + rng.normal(0.00010, 0.001, 756)
    metrics = compute_spy_relative_metrics(portfolio, spy, n_trials=3)

    assert metrics.observations == 756
    assert metrics.net_annualized_excess_return > 0
    assert metrics.information_ratio > 0
    assert 0.90 <= metrics.beta <= 1.10
    assert metrics.tracking_error > 0
    assert 0 <= metrics.portfolio_max_drawdown <= 1
    assert 0 <= metrics.psr <= 1
    assert 0 <= metrics.dsr <= 1


def test_drawdown_normalization_is_positive_magnitude():
    returns = np.array([0.10, -0.20, 0.05])
    assert max_drawdown_magnitude(returns) == pytest.approx(0.20)


def test_psr_and_dsr_penalize_noise_and_multiple_trials():
    rng = np.random.default_rng(42)
    edge = rng.normal(0.001, 0.01, 600)
    noise = rng.normal(0.0, 0.01, 600)
    assert probability_sharpe_ratio(edge) > probability_sharpe_ratio(noise)
    assert deflated_sharpe_ratio(edge, n_trials=500) < deflated_sharpe_ratio(edge, n_trials=1)
    with pytest.raises(ValueError, match="every attempted"):
        deflated_sharpe_ratio(edge, n_trials=0)


def test_pbo_foundation_reports_dominant_strategy_below_coin_flip():
    rng = np.random.default_rng(11)
    matrix = rng.normal(0, 0.01, (480, 6))
    matrix[:, 0] += 0.003
    pbo = probability_backtest_overfit(matrix, n_splits=8)
    assert 0 <= pbo < 0.5
    combined = compute_spy_relative_metrics(
        matrix[:, 0], np.zeros(480) + rng.normal(0, 0.001, 480), n_trials=6, trial_returns=matrix
    )
    assert combined.pbo == pytest.approx(pbo)


def test_pbo_rejects_underpowered_or_malformed_experiments():
    with pytest.raises(ValueError, match="shape"):
        probability_backtest_overfit(np.ones(100))
    with pytest.raises(ValueError, match="four observations"):
        probability_backtest_overfit(np.ones((20, 3)), n_splits=8)
