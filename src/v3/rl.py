"""Fail-closed RL quarantine and future, bounded eligibility contracts.

No v3 production path imports an RL framework.  A checkpoint can only reach a
caller-supplied deserializer through an explicit production registry entry,
an exact content hash, and matching observation/action schemas.  There is no
directory scan or filename fallback.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, TypeVar


class RLQuarantineError(RuntimeError):
    pass


class RLRegistryState(str, Enum):
    RESEARCH = "research"
    CANDIDATE = "candidate"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class RegisteredRLCheckpoint:
    model_id: str
    state: RLRegistryState
    checkpoint_path: Path
    checkpoint_sha256: str
    observation_schema_sha256: str
    action_schema_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("RL model_id is required")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        for field_name in (
            "checkpoint_sha256",
            "observation_schema_sha256",
            "action_schema_sha256",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise ValueError(f"{field_name} must be a SHA-256 digest")


ModelT = TypeVar("ModelT")


class RegisteredRLLoader(Generic[ModelT]):
    def __init__(
        self,
        *,
        expected_observation_schema_sha256: str,
        expected_action_schema_sha256: str,
    ) -> None:
        self.expected_observation_schema_sha256 = expected_observation_schema_sha256
        self.expected_action_schema_sha256 = expected_action_schema_sha256

    def load(
        self,
        entry: RegisteredRLCheckpoint,
        *,
        rl_enabled: bool,
        deserialize: Callable[[Path], ModelT],
    ) -> ModelT:
        if not rl_enabled:
            raise RLQuarantineError("RL_ENABLED=false; v3 cannot load a checkpoint")
        if entry.state is not RLRegistryState.PRODUCTION:
            raise RLQuarantineError("RL checkpoint is not in production registry state")
        if entry.observation_schema_sha256 != self.expected_observation_schema_sha256:
            raise RLQuarantineError("RL observation schema does not match the runtime")
        if entry.action_schema_sha256 != self.expected_action_schema_sha256:
            raise RLQuarantineError("RL action schema does not match the runtime")
        path = entry.checkpoint_path
        if not path.is_absolute():
            raise RLQuarantineError("registry checkpoint path must be absolute")
        if not path.is_file():
            raise RLQuarantineError("registered RL checkpoint does not exist")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != entry.checkpoint_sha256:
            raise RLQuarantineError("registered RL checkpoint hash mismatch")
        return deserialize(path)


@dataclass(frozen=True, slots=True)
class RLAllocationAction:
    core_weight: float
    satellite_weight: float
    cash_weight: float

    def __post_init__(self) -> None:
        values = (self.core_weight, self.satellite_weight, self.cash_weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("RL allocation weights must be finite")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
            raise ValueError("RL allocation weights must sum to one")
        if not 0.58 <= self.core_weight <= 0.90:
            raise ValueError("RL core weight is outside [58%, 90%]")
        if not 0.10 <= self.satellite_weight <= 0.40:
            raise ValueError("RL satellite weight is outside [10%, 40%]")
        if not 0.0 <= self.cash_weight <= 0.02:
            raise ValueError("RL cash weight is outside [0%, 2%]")


@dataclass(frozen=True, slots=True)
class RLEligibilityEvidence:
    seeds: int
    positive_excess_seed_fraction: float
    median_oos_excess_improvement: float
    information_ratio_improvement: float
    dsr_probability: float
    pbo: float
    drawdown_deterioration: float
    deterministic_v3_promoted: bool
    shadow_sessions: int


def rl_eligibility_failures(evidence: RLEligibilityEvidence) -> tuple[str, ...]:
    checks = (
        (evidence.deterministic_v3_promoted, "deterministic_v3_not_promoted"),
        (evidence.seeds >= 10, "fewer_than_10_seeds"),
        (evidence.positive_excess_seed_fraction >= 0.80, "positive_seed_fraction_below_80pct"),
        (evidence.median_oos_excess_improvement >= 0.005, "median_oos_improvement_below_0_5pct"),
        (evidence.information_ratio_improvement >= 0.10, "information_ratio_improvement_below_0_10"),
        (evidence.dsr_probability >= 0.95, "dsr_below_95pct"),
        (evidence.pbo <= 0.20, "pbo_above_20pct"),
        (evidence.drawdown_deterioration <= 0.02, "drawdown_deterioration_above_2pct"),
        (evidence.shadow_sessions >= 60, "rl_shadow_below_60_sessions"),
    )
    return tuple(message for passed, message in checks if not passed)


__all__ = [
    "RLAllocationAction",
    "RLEligibilityEvidence",
    "RLQuarantineError",
    "RLRegistryState",
    "RegisteredRLCheckpoint",
    "RegisteredRLLoader",
    "rl_eligibility_failures",
]
