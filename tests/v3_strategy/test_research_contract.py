from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.v3.metrics import SpyRelativeMetrics, compute_spy_relative_metrics
from src.v3.research import ExperimentEvidence, PreRegisteredStudy, evaluate_promotion


def test_preregistered_windows_references_neighbors_and_annual_embargo_are_locked():
    study = PreRegisteredStudy.from_config()
    sessions = pd.bdate_range("2005-01-03", "2026-06-30")
    folds = study.anchored_folds(sessions)

    assert study.discovery.start == pd.Timestamp("2005-01-01")
    assert study.validation.start == pd.Timestamp("2015-01-01")
    assert study.holdout.end == pd.Timestamp("2026-06-30")
    assert study.embargo_sessions == 21
    assert {reference.name for reference in study.references} == {
        "spy",
        "core_satellite_80_20",
        "core_satellite_60_40",
        "current_65_percent_invested",
        "active_100",
    }
    assert len(study.neighboring_allocations) == 2
    assert study.required_regime_slices == (
        "bull",
        "bear",
        "high_volatility",
        "low_volatility",
    )
    assert len(folds) == 12
    first = folds[0]
    assert first.test.start.year == 2015
    train_position = sessions.get_loc(first.train.end)
    test_position = sessions.get_loc(first.test.start)
    assert test_position - train_position - 1 == 21


def _passing_evidence() -> ExperimentEvidence:
    metrics = SpyRelativeMetrics(
        observations=1000,
        portfolio_annualized_return=0.12,
        spy_annualized_return=0.10,
        net_annualized_excess_return=0.02,
        portfolio_annualized_volatility=0.15,
        tracking_error=0.04,
        information_ratio=0.50,
        beta=1.0,
        portfolio_max_drawdown=0.20,
        spy_max_drawdown=0.20,
        positive_rolling_12m_fraction=0.60,
        psr=0.96,
        dsr=0.96,
        pbo=0.10,
    )
    return ExperimentEvidence(
        metrics=metrics,
        point_in_time_validated=True,
        survivorship_safe=True,
        leakage_checks_passed=True,
        pit_warnings=(),
        required_price_coverage=0.99,
        target_hash_parity=True,
        fold_net_excess_returns=(0.01, 0.02, 0.01),
        fold_alpha_contributions=(0.01, 0.02, 0.01),
        annual_one_way_turnover=1.0,
        stress_net_annualized_excess_return=0.001,
        neighboring_net_excess_returns=(0.01, 0.005),
        attempted_trials=7,
        recorded_trials=7,
        holdout_evaluations=1,
    )


def test_typed_promotion_gate_cannot_skip_hash_stress_or_trial_ledger():
    assert evaluate_promotion(_passing_evidence()).promotable
    failed = replace(
        _passing_evidence(),
        target_hash_parity=False,
        stress_net_annualized_excess_return=0.0,
        recorded_trials=6,
    )
    names = {gate.name for gate in evaluate_promotion(failed).failures}
    assert {"target_hash_parity", "stress_cost_alpha", "trial_ledger"}.issubset(names)


def test_metrics_require_trial_count_to_cover_pbo_matrix():
    rng = np.random.default_rng(12)
    trials = rng.normal(0.0, 0.01, (240, 3))
    with pytest.raises(ValueError, match="cannot omit"):
        compute_spy_relative_metrics(
            trials[:, 0],
            rng.normal(0.0, 0.01, 240),
            n_trials=2,
            trial_returns=trials,
        )
