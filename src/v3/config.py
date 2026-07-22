"""Immutable, content-addressed strategy configuration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "strategies"
    / "spy_xsmom_core_satellite.v3.json"
)


class StrategyConfigError(ValueError):
    """Raised when tracked strategy configuration violates its contract."""


@dataclass(frozen=True, slots=True)
class AllocationConfig:
    core_weight: float
    satellite_weight: float
    cash_weight: float


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    lookback_sessions: int
    skip_sessions: int
    target_holdings: int
    entry_percentile: float
    retention_percentile: float
    momentum_weight: float
    smoothness_weight: float


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_direct_issuer_weight: float
    min_beta: float
    max_beta: float
    tracking_error_target_min: float
    tracking_error_target_max: float
    tracking_error_hard_max: float
    active_sector_target_max: float
    active_sector_hard_max: float
    satellite_step_down: float
    minimum_price_coverage: float
    risk_estimation_sessions: int
    warning_drawdown: float
    de_risk_drawdown: float


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    base_cost_bps_one_way: float
    stress_cost_bps_one_way: float
    embargo_sessions: int
    discovery: tuple[str, str]
    validation: tuple[str, str]
    holdout: tuple[str, str]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    rl_enabled: bool
    disabled_sleeves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    strategy_id: str
    version: str
    benchmark: str
    allocation: AllocationConfig
    momentum: MomentumConfig
    risk: RiskConfig
    research: ResearchConfig
    features: FeatureConfig
    sha256: str
    canonical_json: str

    @property
    def identity(self) -> str:
        return f"{self.strategy_id}@{self.version}:{self.sha256}"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _pair(value: Any, field: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise StrategyConfigError(f"{field} must be a two-element JSON array")
    return str(value[0]), str(value[1])


def _validate(config: StrategyConfig) -> None:
    if config.strategy_id != "spy_xsmom_core_satellite" or config.version != "3.0.0":
        raise StrategyConfigError("v3 requires spy_xsmom_core_satellite version 3.0.0")
    if config.benchmark != "SPY":
        raise StrategyConfigError("v3 benchmark must be SPY")
    alloc = config.allocation
    if not math.isclose(
        alloc.core_weight + alloc.satellite_weight + alloc.cash_weight,
        1.0,
        abs_tol=1e-12,
    ):
        raise StrategyConfigError("allocation weights must sum exactly to 1")
    if min(alloc.core_weight, alloc.satellite_weight, alloc.cash_weight) < 0:
        raise StrategyConfigError("allocation weights cannot be negative")

    mom = config.momentum
    if mom.lookback_sessions < 2 or mom.skip_sessions < 0 or mom.target_holdings < 1:
        raise StrategyConfigError("momentum horizon and holding count must be positive")
    if not 0 < mom.entry_percentile <= mom.retention_percentile <= 1:
        raise StrategyConfigError("entry percentile must be inside the retention band")
    if not math.isclose(mom.momentum_weight + mom.smoothness_weight, 1.0, abs_tol=1e-12):
        raise StrategyConfigError("momentum and smoothness weights must sum to 1")

    risk = config.risk
    if not 0 < risk.minimum_price_coverage <= 1:
        raise StrategyConfigError("minimum_price_coverage must be in (0, 1]")
    if not 0 < risk.satellite_step_down <= alloc.satellite_weight:
        raise StrategyConfigError("satellite_step_down is invalid")
    if not risk.min_beta <= risk.max_beta:
        raise StrategyConfigError("beta bounds are inverted")
    if risk.tracking_error_target_max > risk.tracking_error_hard_max:
        raise StrategyConfigError("tracking-error target exceeds its hard cap")
    if risk.warning_drawdown >= risk.de_risk_drawdown:
        raise StrategyConfigError("drawdown warning must precede de-risking")
    if not 0 < risk.max_direct_issuer_weight <= alloc.satellite_weight:
        raise StrategyConfigError("max_direct_issuer_weight is invalid")
    if not 0 <= risk.active_sector_target_max <= risk.active_sector_hard_max:
        raise StrategyConfigError("active-sector bounds are invalid")
    if risk.tracking_error_target_min < 0 or not (
        risk.tracking_error_target_min
        <= risk.tracking_error_target_max
        <= risk.tracking_error_hard_max
    ):
        raise StrategyConfigError("tracking-error bounds are invalid")
    if risk.risk_estimation_sessions < 30:
        raise StrategyConfigError("risk_estimation_sessions is too short")

    research = config.research
    if research.base_cost_bps_one_way != 5.0 or research.stress_cost_bps_one_way != 15.0:
        raise StrategyConfigError("v3 research costs are locked at 5/15 bps one way")
    if research.embargo_sessions != 21:
        raise StrategyConfigError("v3 research embargo is locked at 21 sessions")
    periods = tuple(tuple(map(str, period)) for period in (research.discovery, research.validation, research.holdout))
    try:
        discovery = tuple(date.fromisoformat(value) for value in periods[0])
        validation = tuple(date.fromisoformat(value) for value in periods[1])
        holdout = tuple(date.fromisoformat(value) for value in periods[2])
    except ValueError as exc:
        raise StrategyConfigError("research periods require ISO dates") from exc
    if not (
        discovery[0] <= discovery[1] < validation[0] <= validation[1] < holdout[0] <= holdout[1]
    ):
        raise StrategyConfigError("research periods must be ordered and non-overlapping")
    if periods != (
        ("2005-01-01", "2014-12-31"),
        ("2015-01-01", "2019-12-31"),
        ("2020-01-01", "2026-06-30"),
    ):
        raise StrategyConfigError("v3 research windows are locked")

    if config.features.rl_enabled:
        raise StrategyConfigError("v3.0.0 requires RL to remain quarantined")
    required_disabled = {"value_quality_momentum", "pead", "insider", "macro_timing"}
    if not required_disabled.issubset(config.features.disabled_sleeves):
        raise StrategyConfigError("experimental sleeves must remain disabled in v3.0.0")


def load_strategy_config(path: str | Path | None = None) -> StrategyConfig:
    """Load a tracked JSON config and attach a canonical SHA-256 identity.

    Whitespace and key order do not affect the hash.  The returned object and
    every nested structure are frozen, so runtime code cannot change the
    strategy after a run has been identified.
    """

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyConfigError(f"cannot load strategy config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StrategyConfigError("strategy config root must be an object")

    required = {"strategy_id", "version", "benchmark", "allocation", "momentum", "risk", "research", "features"}
    missing = required.difference(payload)
    if missing:
        raise StrategyConfigError(f"missing strategy fields: {sorted(missing)}")

    canonical = _canonical_json(payload)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    try:
        allocation = AllocationConfig(**payload["allocation"])
        momentum = MomentumConfig(**payload["momentum"])
        risk = RiskConfig(**payload["risk"])
        research_raw = payload["research"]
        research = ResearchConfig(
            base_cost_bps_one_way=float(research_raw["base_cost_bps_one_way"]),
            stress_cost_bps_one_way=float(research_raw["stress_cost_bps_one_way"]),
            embargo_sessions=int(research_raw["embargo_sessions"]),
            discovery=_pair(research_raw["discovery"], "research.discovery"),
            validation=_pair(research_raw["validation"], "research.validation"),
            holdout=_pair(research_raw["holdout"], "research.holdout"),
        )
        rl_enabled = payload["features"]["rl_enabled"]
        if not isinstance(rl_enabled, bool):
            raise StrategyConfigError("features.rl_enabled must be a JSON boolean")
        disabled_sleeves = payload["features"]["disabled_sleeves"]
        if not isinstance(disabled_sleeves, list) or not all(
            isinstance(value, str) for value in disabled_sleeves
        ):
            raise StrategyConfigError("features.disabled_sleeves must be a string array")
        features = FeatureConfig(
            rl_enabled=rl_enabled,
            disabled_sleeves=tuple(disabled_sleeves),
        )
        config = StrategyConfig(
            strategy_id=str(payload["strategy_id"]),
            version=str(payload["version"]),
            benchmark=str(payload["benchmark"]),
            allocation=allocation,
            momentum=momentum,
            risk=risk,
            research=research,
            features=features,
            sha256=digest,
            canonical_json=canonical,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategyConfigError(f"invalid strategy config shape: {exc}") from exc

    _validate(config)
    return config
