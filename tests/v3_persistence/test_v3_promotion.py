from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import pytest

from src.db.v3_models import ExperimentRun, V3Base
from src.v3 import load_strategy_config
from src.v3.promotion import ExperimentRegistry, evaluate_promotion_metrics


def _passing_metrics() -> dict:
    return {
        "pit_leakage_survivorship_warnings": [],
        "required_price_coverage": 0.99,
        "target_hash_parity": True,
        "oos_annualized_excess_return": 0.02,
        "information_ratio": 0.5,
        "beta": 1.0,
        "tracking_error": 0.04,
        "max_drawdown": 0.20,
        "spy_max_drawdown": 0.20,
        "positive_rolling_12m_fraction": 0.60,
        "positive_oos_fold_fraction": 0.75,
        "annual_one_way_turnover": 1.20,
        "stress_15bps_excess_return": 0.005,
        "max_fold_alpha_contribution": 0.40,
        "psr_probability": 0.96,
        "dsr_probability": 0.96,
        "pbo": 0.15,
        "neighboring_parameters_preserve_excess_sign": True,
        "pre_registered_study": True,
        "holdout_end": "2026-06-30",
        "related_trial_count": 1,
    }


def test_locked_promotion_thresholds_fail_closed() -> None:
    metrics = _passing_metrics()
    metrics["information_ratio"] = 0.39
    metrics["stress_15bps_excess_return"] = 0.0

    evaluation = evaluate_promotion_metrics(metrics)

    assert evaluation.passed is False
    assert "information_ratio_below_0_40" in evaluation.failures
    assert "nonpositive_15bps_stress_excess" in evaluation.failures


def test_registry_requires_exact_config_data_and_commit_evidence() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    config = load_strategy_config()
    registry = ExperimentRegistry(engine)
    data_hash = "a" * 64
    commit = "b" * 40

    registry.register_preregistration(
        config=config,
        data_manifest_sha256=data_hash,
        commit_sha=commit,
        trial_family="preregistered_v3",
        parameters={"allocation": "70/30", "holdout_end": "2026-06-30"},
    )
    row = registry.record_trial(
        config=config,
        data_manifest_sha256=data_hash,
        commit_sha=commit,
        trial_family="preregistered_v3",
        split_name="final_holdout",
        parameters={"allocation": "70/30", "holdout_end": "2026-06-30"},
        metrics=_passing_metrics(),
    )

    assert row.promotion_status == "passed"
    assert registry.has_passing_evidence(
        config=config, data_manifest_sha256=data_hash, commit_sha=commit
    )
    assert not registry.has_passing_evidence(
        config=config, data_manifest_sha256="c" * 64, commit_sha=commit
    )
    assert not registry.has_passing_evidence(
        config=config,
        data_manifest_sha256=data_hash,
        commit_sha=commit,
        require_trusted_runner=True,
    )
    with pytest.raises(ValueError, match="already been attempted"):
        registry.record_trial(
            config=config,
            data_manifest_sha256=data_hash,
            commit_sha=commit,
            trial_family="preregistered_v3",
            split_name="final_holdout",
            parameters={"allocation": "70/30", "holdout_end": "2026-06-30"},
            metrics=_passing_metrics(),
        )
    with Session(engine) as session:
        assert len(session.scalars(select(ExperimentRun)).all()) == 2


def test_callers_cannot_self_attest_trusted_study_metrics() -> None:
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    config = load_strategy_config()
    registry = ExperimentRegistry(engine)
    metrics = _passing_metrics()
    metrics.update(
        {
            "trusted_study_runner": True,
            "study_attestation_sha256": "f" * 64,
        }
    )

    with pytest.raises(ValueError, match="trusted study evidence"):
        registry.record_trial(
            config=config,
            data_manifest_sha256="d" * 64,
            commit_sha="e" * 40,
            trial_family="spoofed",
            split_name="final_holdout",
            parameters={},
            metrics=metrics,
        )
