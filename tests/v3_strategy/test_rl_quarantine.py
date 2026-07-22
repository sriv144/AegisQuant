from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from src.v3.rl import (
    RLAllocationAction,
    RLEligibilityEvidence,
    RLQuarantineError,
    RLRegistryState,
    RegisteredRLCheckpoint,
    RegisteredRLLoader,
    rl_eligibility_failures,
)


OBSERVATION_SCHEMA = "a" * 64
ACTION_SCHEMA = "b" * 64


def _entry(path, *, state=RLRegistryState.PRODUCTION, observation=OBSERVATION_SCHEMA):
    return RegisteredRLCheckpoint(
        model_id="future-rl-v1",
        state=state,
        checkpoint_path=path,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        observation_schema_sha256=observation,
        action_schema_sha256=ACTION_SCHEMA,
    )


def test_rl_disabled_registry_state_and_schema_all_fail_before_deserialization(tmp_path):
    checkpoint = tmp_path / "registered.chk"
    checkpoint.write_bytes(b"checkpoint")
    loader = RegisteredRLLoader(
        expected_observation_schema_sha256=OBSERVATION_SCHEMA,
        expected_action_schema_sha256=ACTION_SCHEMA,
    )
    calls = []

    with pytest.raises(RLQuarantineError, match="RL_ENABLED=false"):
        loader.load(_entry(checkpoint), rl_enabled=False, deserialize=calls.append)
    with pytest.raises(RLQuarantineError, match="production registry"):
        loader.load(
            _entry(checkpoint, state=RLRegistryState.CANDIDATE),
            rl_enabled=True,
            deserialize=calls.append,
        )
    with pytest.raises(RLQuarantineError, match="observation schema"):
        loader.load(
            _entry(checkpoint, observation="c" * 64),
            rl_enabled=True,
            deserialize=calls.append,
        )
    assert calls == []


def test_no_filename_fallback_and_exact_checkpoint_hash_are_enforced(tmp_path):
    checkpoint = tmp_path / "registered.chk"
    checkpoint.write_bytes(b"approved")
    loader = RegisteredRLLoader(
        expected_observation_schema_sha256=OBSERVATION_SCHEMA,
        expected_action_schema_sha256=ACTION_SCHEMA,
    )
    entry = _entry(checkpoint)
    checkpoint.write_bytes(b"tampered")
    (tmp_path / "production.zip").write_bytes(b"approved")

    with pytest.raises(RLQuarantineError, match="hash mismatch"):
        loader.load(entry, rl_enabled=True, deserialize=lambda path: path)


def test_future_rl_action_bounds_and_all_eligibility_gates_are_executable():
    action = RLAllocationAction(core_weight=0.69, satellite_weight=0.30, cash_weight=0.01)
    assert action.satellite_weight == pytest.approx(0.30)
    with pytest.raises(ValueError, match="core weight"):
        RLAllocationAction(core_weight=0.57, satellite_weight=0.42, cash_weight=0.01)

    passing = RLEligibilityEvidence(
        seeds=10,
        positive_excess_seed_fraction=0.80,
        median_oos_excess_improvement=0.005,
        information_ratio_improvement=0.10,
        dsr_probability=0.95,
        pbo=0.20,
        drawdown_deterioration=0.02,
        deterministic_v3_promoted=True,
        shadow_sessions=60,
    )
    assert rl_eligibility_failures(passing) == ()
    assert "fewer_than_10_seeds" in rl_eligibility_failures(
        replace(passing, seeds=9)
    )
