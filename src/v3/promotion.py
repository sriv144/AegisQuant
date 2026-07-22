"""Research-promotion gates and durable evidence registry."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from src.db.v3_models import ExperimentRun

from .config import StrategyConfig


@dataclass(frozen=True, slots=True)
class PromotionEvaluation:
    passed: bool
    failures: tuple[str, ...]


def evaluate_promotion_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_related_trials: int = 1,
) -> PromotionEvaluation:
    """Apply the locked v3 economic, risk and robustness thresholds."""

    failures: list[str] = []

    def number(name: str) -> float:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"missing_or_invalid:{name}")
            return math.nan
        if not math.isfinite(value):
            failures.append(f"non_finite:{name}")
        return value

    warnings = metrics.get("pit_leakage_survivorship_warnings")
    if not isinstance(warnings, list) or warnings:
        failures.append("pit_leakage_or_survivorship_warning")
    if number("required_price_coverage") < 0.98:
        failures.append("price_coverage_below_98pct")
    if metrics.get("target_hash_parity") is not True:
        failures.append("target_hash_parity_failed")
    if number("oos_annualized_excess_return") < 0.015:
        failures.append("oos_excess_return_below_1_5pct")
    if number("information_ratio") < 0.40:
        failures.append("information_ratio_below_0_40")
    beta = number("beta")
    if math.isfinite(beta) and not 0.90 <= beta <= 1.10:
        failures.append("beta_outside_0_90_1_10")
    if number("tracking_error") > 0.06:
        failures.append("tracking_error_above_6pct")
    drawdown = number("max_drawdown")
    spy_drawdown = number("spy_max_drawdown")
    if drawdown > 0.25:
        failures.append("max_drawdown_above_25pct")
    if math.isfinite(drawdown) and math.isfinite(spy_drawdown) and drawdown > spy_drawdown + 0.02:
        failures.append("drawdown_more_than_2pct_worse_than_spy")
    if number("positive_rolling_12m_fraction") < 0.55:
        failures.append("rolling_12m_excess_consistency_below_55pct")
    if number("positive_oos_fold_fraction") < 2 / 3:
        failures.append("positive_oos_folds_below_two_thirds")
    if number("annual_one_way_turnover") > 1.50:
        failures.append("annual_turnover_above_150pct")
    if number("stress_15bps_excess_return") <= 0:
        failures.append("nonpositive_15bps_stress_excess")
    if number("max_fold_alpha_contribution") > 0.50:
        failures.append("single_fold_alpha_contribution_above_50pct")
    if number("psr_probability") < 0.95:
        failures.append("psr_below_95pct")
    if number("dsr_probability") < 0.95:
        failures.append("dsr_below_95pct")
    if number("pbo") > 0.20:
        failures.append("pbo_above_20pct")
    if metrics.get("neighboring_parameters_preserve_excess_sign") is not True:
        failures.append("neighboring_parameter_sign_failed")
    if metrics.get("pre_registered_study") is not True:
        failures.append("study_not_pre_registered")
    if metrics.get("holdout_end") != "2026-06-30":
        failures.append("holdout_window_mismatch")
    try:
        included_trials = int(metrics["related_trial_count"])
    except (KeyError, TypeError, ValueError):
        included_trials = 0
        failures.append("missing_or_invalid:related_trial_count")
    if included_trials < expected_related_trials:
        failures.append("dsr_does_not_include_all_related_trials")

    return PromotionEvaluation(False if failures else True, tuple(dict.fromkeys(failures)))


class ExperimentRegistry:
    """Append every attempted trial and query exact passing evidence."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def register_preregistration(
        self,
        *,
        config: StrategyConfig,
        data_manifest_sha256: str,
        commit_sha: str,
        trial_family: str,
        parameters: Mapping[str, Any],
    ) -> ExperimentRun:
        """Durably freeze the study before its final holdout is recorded."""

        if len(data_manifest_sha256) != 64 or len(commit_sha) < 7:
            raise ValueError("data and commit identities are required")

        identity = self._study_identity(
            config, data_manifest_sha256, trial_family, "preregistration"
        )
        with Session(self.engine) as session:
            existing = session.get(ExperimentRun, identity)
            if existing is not None:
                if (
                    existing.commit_sha != commit_sha
                    or dict(existing.parameters_json) != dict(parameters)
                ):
                    raise ValueError("preregistration is immutable")
                return existing
            row = ExperimentRun(
                experiment_id=identity,
                strategy_id=config.strategy_id,
                strategy_version=config.version,
                config_sha256=config.sha256,
                data_manifest_sha256=data_manifest_sha256,
                trial_family=trial_family,
                split_name="preregistration",
                parameters_json=dict(parameters),
                metrics_json={},
                warnings_json=[],
                promotion_status="preregistered",
                gate_failures_json=[],
                commit_sha=commit_sha,
                attempted_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def require_preregistration(
        self,
        *,
        config: StrategyConfig,
        data_manifest_sha256: str,
        commit_sha: str,
        trial_family: str,
        parameters: Mapping[str, Any],
    ) -> ExperimentRun:
        """Fail unless the exact study was durably frozen before evaluation."""

        identity = self._study_identity(
            config, data_manifest_sha256, trial_family, "preregistration"
        )
        with Session(self.engine) as session:
            row = session.get(ExperimentRun, identity)
            if (
                row is None
                or row.commit_sha != commit_sha
                or dict(row.parameters_json) != dict(parameters)
                or row.promotion_status != "preregistered"
            ):
                raise ValueError(
                    "exact config/data/commit study must be preregistered before evaluation"
                )
            return row

    def record_trial(
        self,
        *,
        config: StrategyConfig,
        data_manifest_sha256: str,
        commit_sha: str,
        trial_family: str,
        split_name: str,
        parameters: Mapping[str, Any],
        metrics: Mapping[str, Any],
        warnings: Sequence[str] = (),
    ) -> ExperimentRun:
        """Record an ordinary trial; callers cannot self-attest trusted evidence."""

        if (
            metrics.get("trusted_study_runner") is True
            or "study_attestation_sha256" in metrics
        ):
            raise ValueError(
                "trusted study evidence must come from record_trusted_final_holdout"
            )
        return self._record_trial(
            config=config,
            data_manifest_sha256=data_manifest_sha256,
            commit_sha=commit_sha,
            trial_family=trial_family,
            split_name=split_name,
            parameters=parameters,
            metrics=metrics,
            warnings=warnings,
        )

    def record_trusted_final_holdout(
        self,
        *,
        config: StrategyConfig,
        commit_sha: str,
        trial_family: str,
        result: Any,
    ) -> ExperimentRun:
        """Persist final evidence only after recomputing the study attestation."""

        from .study import TrustedStudyResult, verify_study_attestation

        if not isinstance(result, TrustedStudyResult):
            raise ValueError("trusted evidence requires a TrustedStudyResult")
        if not verify_study_attestation(result, config):
            raise ValueError("trusted study attestation verification failed")
        parameters = {
            "runner_version": str(result.promotion_metrics.get("runner_version", ""))
            or "aegisquant-trusted-study-v1",
            "registration_sha256": result.study.registration_sha256,
            "champion": result.study.champion.name,
            "holdout_end": result.study.holdout.end.date().isoformat(),
        }
        expected_related = self.related_trial_count(
            config=config,
            data_manifest_sha256=result.research_data_sha256,
        ) + 1
        if int(result.promotion_metrics.get("related_trial_count", 0)) != expected_related:
            raise ValueError("trusted study did not include every related attempted trial")
        return self._record_trial(
            config=config,
            data_manifest_sha256=result.research_data_sha256,
            commit_sha=commit_sha,
            trial_family=trial_family,
            split_name="final_holdout",
            parameters=parameters,
            metrics=result.promotion_metrics,
            warnings=(),
        )

    def _record_trial(
        self,
        *,
        config: StrategyConfig,
        data_manifest_sha256: str,
        commit_sha: str,
        trial_family: str,
        split_name: str,
        parameters: Mapping[str, Any],
        metrics: Mapping[str, Any],
        warnings: Sequence[str] = (),
    ) -> ExperimentRun:
        if len(data_manifest_sha256) != 64 or len(commit_sha) < 7:
            raise ValueError("data and commit identities are required")
        with Session(self.engine) as session:
            if split_name == "final_holdout":
                preregistration_id = self._study_identity(
                    config,
                    data_manifest_sha256,
                    trial_family,
                    "preregistration",
                )
                preregistration = session.get(ExperimentRun, preregistration_id)
                if preregistration is None or preregistration.commit_sha != commit_sha:
                    raise ValueError(
                        "final holdout requires a prior config/data/commit-bound preregistration"
                    )
                preregistered_parameters = dict(preregistration.parameters_json)
                if dict(parameters) != preregistered_parameters:
                    raise ValueError("final holdout parameters were not preregistered")
                prior_holdout = session.scalar(
                    select(ExperimentRun.experiment_id)
                    .where(
                        ExperimentRun.strategy_id == config.strategy_id,
                        ExperimentRun.strategy_version == config.version,
                        ExperimentRun.config_sha256 == config.sha256,
                        ExperimentRun.data_manifest_sha256 == data_manifest_sha256,
                        ExperimentRun.split_name == "final_holdout",
                    )
                    .limit(1)
                )
                if prior_holdout is not None:
                    raise ValueError("the final holdout has already been attempted")
                experiment_id = self._study_identity(
                    config, data_manifest_sha256, trial_family, "final_holdout"
                )
            else:
                experiment_id = str(uuid.uuid4())
            prior_count = session.scalar(
                select(func.count())
                .select_from(ExperimentRun)
                .where(
                    ExperimentRun.strategy_id == config.strategy_id,
                    ExperimentRun.strategy_version == config.version,
                    ExperimentRun.config_sha256 == config.sha256,
                    ExperimentRun.data_manifest_sha256 == data_manifest_sha256,
                    ExperimentRun.split_name != "preregistration",
                )
            ) or 0
            evaluation = evaluate_promotion_metrics(
                metrics, expected_related_trials=int(prior_count) + 1
            )
            combined_failures = tuple(evaluation.failures) + tuple(
                f"research_warning:{warning}" for warning in warnings
            )
            row = ExperimentRun(
                experiment_id=experiment_id,
                strategy_id=config.strategy_id,
                strategy_version=config.version,
                config_sha256=config.sha256,
                data_manifest_sha256=data_manifest_sha256,
                trial_family=trial_family,
                split_name=split_name,
                parameters_json=dict(parameters),
                metrics_json=dict(metrics),
                warnings_json=list(warnings),
                promotion_status="passed" if not combined_failures else "failed",
                gate_failures_json=list(combined_failures),
                commit_sha=commit_sha,
                attempted_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    @staticmethod
    def _study_identity(
        config: StrategyConfig,
        data_manifest_sha256: str,
        trial_family: str,
        split_name: str,
    ) -> str:
        payload = "|".join(
            (
                "aegisquant-v3-study",
                config.strategy_id,
                config.version,
                config.sha256,
                data_manifest_sha256,
                trial_family,
                split_name,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def related_trial_count(
        self,
        *,
        config: StrategyConfig,
        data_manifest_sha256: str,
    ) -> int:
        """Count every attempted related trial across caller-chosen families."""

        with Session(self.engine) as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(ExperimentRun)
                    .where(
                        ExperimentRun.strategy_id == config.strategy_id,
                        ExperimentRun.strategy_version == config.version,
                        ExperimentRun.config_sha256 == config.sha256,
                        ExperimentRun.data_manifest_sha256 == data_manifest_sha256,
                        ExperimentRun.split_name != "preregistration",
                    )
                )
                or 0
            )

    def has_passing_evidence(
        self,
        *,
        config: StrategyConfig,
        data_manifest_sha256: str,
        commit_sha: str,
        require_trusted_runner: bool = False,
    ) -> bool:
        if not commit_sha or commit_sha == "unknown":
            return False
        with Session(self.engine) as session:
            rows = session.scalars(
                select(ExperimentRun)
                .where(
                    ExperimentRun.strategy_id == config.strategy_id,
                    ExperimentRun.strategy_version == config.version,
                    ExperimentRun.config_sha256 == config.sha256,
                    ExperimentRun.data_manifest_sha256 == data_manifest_sha256,
                    ExperimentRun.commit_sha == commit_sha,
                    ExperimentRun.split_name == "final_holdout",
                    ExperimentRun.promotion_status == "passed",
                )
                .order_by(ExperimentRun.attempted_at.desc())
                .limit(1)
            ).all()
            if not rows:
                return False
            if not require_trusted_runner:
                return True
            metrics = dict(rows[0].metrics_json or {})
            attestation = str(metrics.get("study_attestation_sha256", ""))
            return metrics.get("trusted_study_runner") is True and len(attestation) == 64


__all__ = [
    "ExperimentRegistry",
    "PromotionEvaluation",
    "evaluate_promotion_metrics",
]
