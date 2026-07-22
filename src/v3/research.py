"""Pre-registered study design and deterministic promotion gates for v3."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import pandas as pd

from .config import StrategyConfig, load_strategy_config
from .metrics import SpyRelativeMetrics


class StudyPhase(str, Enum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


@dataclass(frozen=True, slots=True)
class StudyPeriod:
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", pd.Timestamp(self.start).normalize())
        object.__setattr__(self, "end", pd.Timestamp(self.end).normalize())
        if self.end < self.start:
            raise ValueError("study period end precedes start")


@dataclass(frozen=True, slots=True)
class AnchoredFold:
    phase: StudyPhase
    train: StudyPeriod
    test: StudyPeriod
    embargo_sessions: int


@dataclass(frozen=True, slots=True)
class ReferencePortfolio:
    name: str
    core_weight: float
    satellite_weight: float
    cash_weight: float

    def __post_init__(self) -> None:
        if min(self.core_weight, self.satellite_weight, self.cash_weight) < 0:
            raise ValueError("reference weights cannot be negative")
        if not math.isclose(
            self.core_weight + self.satellite_weight + self.cash_weight,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("reference weights must sum to one")


@dataclass(frozen=True, slots=True)
class PreRegisteredStudy:
    strategy_id: str
    strategy_version: str
    config_sha256: str
    discovery: StudyPeriod
    validation: StudyPeriod
    holdout: StudyPeriod
    embargo_sessions: int
    base_cost_bps_one_way: float
    stress_cost_bps_one_way: float
    champion: ReferencePortfolio
    references: tuple[ReferencePortfolio, ...]
    neighboring_allocations: tuple[ReferencePortfolio, ...]
    required_regime_slices: tuple[str, ...]
    registration_sha256: str

    def __post_init__(self) -> None:
        if self.embargo_sessions != 21:
            raise ValueError("v3 study requires a 21-session embargo")
        if not (
            self.discovery.end < self.validation.start
            and self.validation.end < self.holdout.start
        ):
            raise ValueError("study phases must be ordered and non-overlapping")
        if self.base_cost_bps_one_way != 5.0 or self.stress_cost_bps_one_way != 15.0:
            raise ValueError("v3 study costs are locked at 5/15 bps")
        if self.holdout.end != pd.Timestamp("2026-06-30"):
            raise ValueError("v3 holdout must end on 2026-06-30")
        if len(self.registration_sha256) != 64:
            raise ValueError("study registration requires a SHA-256 identity")

    @classmethod
    def from_config(cls, config: StrategyConfig | None = None) -> "PreRegisteredStudy":
        strategy = config or load_strategy_config()
        champion = ReferencePortfolio(
            "champion_69_30_1",
            strategy.allocation.core_weight,
            strategy.allocation.satellite_weight,
            strategy.allocation.cash_weight,
        )
        references = (
            ReferencePortfolio("spy", 1.0, 0.0, 0.0),
            ReferencePortfolio("core_satellite_80_20", 0.80, 0.20, 0.0),
            ReferencePortfolio("core_satellite_60_40", 0.60, 0.40, 0.0),
            ReferencePortfolio("current_65_percent_invested", 0.455, 0.195, 0.35),
            ReferencePortfolio("active_100", 0.0, 1.0, 0.0),
        )
        neighbors = (
            ReferencePortfolio("neighbor_74_25_1", 0.74, 0.25, 0.01),
            ReferencePortfolio("neighbor_64_35_1", 0.64, 0.35, 0.01),
        )
        regime_slices = ("bull", "bear", "high_volatility", "low_volatility")
        payload = {
            "strategy_id": strategy.strategy_id,
            "strategy_version": strategy.version,
            "config_sha256": strategy.sha256,
            "discovery": list(strategy.research.discovery),
            "validation": list(strategy.research.validation),
            "holdout": list(strategy.research.holdout),
            "embargo_sessions": strategy.research.embargo_sessions,
            "base_cost_bps_one_way": strategy.research.base_cost_bps_one_way,
            "stress_cost_bps_one_way": strategy.research.stress_cost_bps_one_way,
            "champion": [champion.core_weight, champion.satellite_weight, champion.cash_weight],
            "references": [
                [item.name, item.core_weight, item.satellite_weight, item.cash_weight]
                for item in references
            ],
            "neighbors": [
                [item.name, item.core_weight, item.satellite_weight, item.cash_weight]
                for item in neighbors
            ],
            "required_regime_slices": list(regime_slices),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return cls(
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.version,
            config_sha256=strategy.sha256,
            discovery=StudyPeriod(*strategy.research.discovery),
            validation=StudyPeriod(*strategy.research.validation),
            holdout=StudyPeriod(*strategy.research.holdout),
            embargo_sessions=strategy.research.embargo_sessions,
            base_cost_bps_one_way=strategy.research.base_cost_bps_one_way,
            stress_cost_bps_one_way=strategy.research.stress_cost_bps_one_way,
            champion=champion,
            references=references,
            neighboring_allocations=neighbors,
            required_regime_slices=regime_slices,
            registration_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def anchored_folds(self, sessions: Sequence[pd.Timestamp]) -> tuple[AnchoredFold, ...]:
        """Build annual anchored validation/holdout folds with a session embargo."""

        index = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize().drop_duplicates().sort_values()
        index = index[(index >= self.discovery.start) & (index <= self.holdout.end)]
        if len(index) == 0:
            raise ValueError("no sessions overlap the pre-registered study")
        folds: list[AnchoredFold] = []
        first_year = self.validation.start.year
        last_year = self.holdout.end.year
        for year in range(first_year, last_year + 1):
            phase_period = self.validation if year <= self.validation.end.year else self.holdout
            phase = StudyPhase.VALIDATION if phase_period is self.validation else StudyPhase.HOLDOUT
            requested_start = max(pd.Timestamp(year=year, month=1, day=1), phase_period.start)
            requested_end = min(pd.Timestamp(year=year, month=12, day=31), phase_period.end)
            test_sessions = index[(index >= requested_start) & (index <= requested_end)]
            if len(test_sessions) == 0:
                continue
            first_test_position = int(index.get_loc(test_sessions[0]))
            train_end_position = first_test_position - self.embargo_sessions - 1
            if train_end_position < 0:
                raise ValueError(f"insufficient discovery data before {test_sessions[0].date()}")
            train_sessions = index[
                (index >= self.discovery.start) & (index <= index[train_end_position])
            ]
            if len(train_sessions) == 0:
                raise ValueError(f"empty anchored training window before {test_sessions[0].date()}")
            folds.append(
                AnchoredFold(
                    phase=phase,
                    train=StudyPeriod(train_sessions[0], train_sessions[-1]),
                    test=StudyPeriod(test_sessions[0], test_sessions[-1]),
                    embargo_sessions=self.embargo_sessions,
                )
            )
        return tuple(folds)


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    minimum_coverage: float = 0.98
    minimum_net_excess_return: float = 0.015
    minimum_information_ratio: float = 0.40
    minimum_beta: float = 0.90
    maximum_beta: float = 1.10
    maximum_tracking_error: float = 0.06
    maximum_drawdown: float = 0.25
    maximum_drawdown_vs_spy_gap: float = 0.02
    minimum_positive_rolling_12m_fraction: float = 0.55
    minimum_positive_fold_fraction: float = 2.0 / 3.0
    maximum_annual_one_way_turnover: float = 1.50
    maximum_single_fold_alpha_share: float = 0.50
    minimum_psr: float = 0.95
    minimum_dsr: float = 0.95
    maximum_pbo: float = 0.20


@dataclass(frozen=True, slots=True)
class ExperimentEvidence:
    metrics: SpyRelativeMetrics
    point_in_time_validated: bool
    survivorship_safe: bool
    leakage_checks_passed: bool
    pit_warnings: tuple[str, ...]
    required_price_coverage: float
    target_hash_parity: bool
    fold_net_excess_returns: tuple[float, ...]
    fold_alpha_contributions: tuple[float, ...]
    annual_one_way_turnover: float
    stress_net_annualized_excess_return: float
    neighboring_net_excess_returns: tuple[float, ...]
    attempted_trials: int
    recorded_trials: int
    holdout_evaluations: int

    def __post_init__(self) -> None:
        if self.attempted_trials < 1 or self.recorded_trials < 0:
            raise ValueError("trial counts must be positive and non-negative")
        if len(self.fold_net_excess_returns) != len(self.fold_alpha_contributions):
            raise ValueError("fold returns and alpha contributions must align")
        if any(
            not math.isfinite(value)
            for value in (
                *self.fold_net_excess_returns,
                *self.fold_alpha_contributions,
                *self.neighboring_net_excess_returns,
            )
        ):
            raise ValueError("promotion evidence cannot contain non-finite fold results")


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    observed: str
    requirement: str


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promotable: bool
    gates: tuple[GateResult, ...]

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


def evaluate_promotion(
    evidence: ExperimentEvidence,
    thresholds: PromotionThresholds | None = None,
) -> PromotionDecision:
    """Apply every pre-registered v3 research promotion gate."""

    limits = thresholds or PromotionThresholds()
    metrics = evidence.metrics
    folds = evidence.fold_net_excess_returns
    positive_fold_fraction = sum(value > 0 for value in folds) / len(folds) if folds else 0.0
    positive_contributions = [max(0.0, value) for value in evidence.fold_alpha_contributions]
    positive_total = sum(positive_contributions)
    max_fold_share = max(positive_contributions, default=math.inf) / positive_total if positive_total > 0 else math.inf
    pbo_passed = metrics.pbo is not None and metrics.pbo <= limits.maximum_pbo

    checks = (
        ("point_in_time_validated", evidence.point_in_time_validated, str(evidence.point_in_time_validated), "true"),
        ("survivorship_safe", evidence.survivorship_safe, str(evidence.survivorship_safe), "true"),
        ("leakage_checks", evidence.leakage_checks_passed, str(evidence.leakage_checks_passed), "true"),
        ("pit_warnings", not evidence.pit_warnings, str(list(evidence.pit_warnings)), "none"),
        ("price_coverage", evidence.required_price_coverage >= limits.minimum_coverage, f"{evidence.required_price_coverage:.6f}", f">={limits.minimum_coverage}"),
        ("target_hash_parity", evidence.target_hash_parity, str(evidence.target_hash_parity), "true"),
        ("net_excess_return", metrics.net_annualized_excess_return >= limits.minimum_net_excess_return, f"{metrics.net_annualized_excess_return:.6f}", f">={limits.minimum_net_excess_return}"),
        ("information_ratio", metrics.information_ratio >= limits.minimum_information_ratio, f"{metrics.information_ratio:.6f}", f">={limits.minimum_information_ratio}"),
        ("beta", limits.minimum_beta <= metrics.beta <= limits.maximum_beta, f"{metrics.beta:.6f}", f"[{limits.minimum_beta},{limits.maximum_beta}]"),
        ("tracking_error", metrics.tracking_error <= limits.maximum_tracking_error, f"{metrics.tracking_error:.6f}", f"<={limits.maximum_tracking_error}"),
        ("absolute_drawdown", metrics.portfolio_max_drawdown <= limits.maximum_drawdown, f"{metrics.portfolio_max_drawdown:.6f}", f"<={limits.maximum_drawdown}"),
        ("relative_drawdown", metrics.portfolio_max_drawdown <= metrics.spy_max_drawdown + limits.maximum_drawdown_vs_spy_gap, f"{metrics.portfolio_max_drawdown - metrics.spy_max_drawdown:.6f}", f"<={limits.maximum_drawdown_vs_spy_gap} worse than SPY"),
        ("rolling_12m", metrics.positive_rolling_12m_fraction >= limits.minimum_positive_rolling_12m_fraction, f"{metrics.positive_rolling_12m_fraction:.6f}", f">={limits.minimum_positive_rolling_12m_fraction}"),
        ("positive_folds", positive_fold_fraction >= limits.minimum_positive_fold_fraction, f"{positive_fold_fraction:.6f}", f">={limits.minimum_positive_fold_fraction}"),
        ("fold_evidence", bool(folds) and len(folds) == len(evidence.fold_alpha_contributions), f"{len(folds)}", "at least one aligned OOS fold"),
        ("turnover", evidence.annual_one_way_turnover <= limits.maximum_annual_one_way_turnover, f"{evidence.annual_one_way_turnover:.6f}", f"<={limits.maximum_annual_one_way_turnover}"),
        ("stress_cost_alpha", evidence.stress_net_annualized_excess_return > 0, f"{evidence.stress_net_annualized_excess_return:.6f}", ">0"),
        ("fold_concentration", max_fold_share <= limits.maximum_single_fold_alpha_share, f"{max_fold_share:.6f}", f"<={limits.maximum_single_fold_alpha_share}"),
        ("psr", metrics.psr >= limits.minimum_psr, f"{metrics.psr:.6f}", f">={limits.minimum_psr}"),
        ("dsr", metrics.dsr >= limits.minimum_dsr, f"{metrics.dsr:.6f}", f">={limits.minimum_dsr}"),
        ("pbo", pbo_passed, "missing" if metrics.pbo is None else f"{metrics.pbo:.6f}", f"<={limits.maximum_pbo}"),
        ("neighbor_robustness", bool(evidence.neighboring_net_excess_returns) and all(value > 0 for value in evidence.neighboring_net_excess_returns), str(list(evidence.neighboring_net_excess_returns)), "all >0"),
        ("trial_ledger", evidence.attempted_trials > 0 and evidence.recorded_trials == evidence.attempted_trials, f"{evidence.recorded_trials}/{evidence.attempted_trials}", "all attempted trials recorded"),
        ("holdout_once", evidence.holdout_evaluations == 1, str(evidence.holdout_evaluations), "exactly 1"),
    )
    gates = tuple(GateResult(name, bool(passed), observed, requirement) for name, passed, observed, requirement in checks)
    return PromotionDecision(promotable=all(gate.passed for gate in gates), gates=gates)


EXTERNAL_DATA_REQUIREMENTS = (
    "Point-in-time S&P 500 constituent membership with effective dates",
    "Adjusted total-return and unadjusted executable OHLCV histories",
    "Dividends, splits, symbol changes, delisting dates and delisting returns",
    "Historical sector and issuer/share-class mappings",
    "Historical ADV/spread observations and timestamped executable quotes",
    "Independently validated coverage and availability timestamps",
)
