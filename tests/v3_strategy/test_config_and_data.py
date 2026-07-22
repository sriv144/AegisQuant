from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.v3.config import DEFAULT_CONFIG_PATH, StrategyConfigError, load_strategy_config
from src.v3.data import (
    DataContractError,
    DataManifest,
    DataReadiness,
    DataTier,
    OpenResearchProvider,
    PointInTimeDataProvider,
    REQUIRED_PORTFOLIO_DATASETS,
    snapshot_sha256,
)


def test_tracked_config_is_versioned_frozen_and_content_addressed(tmp_path):
    config = load_strategy_config()
    assert config.strategy_id == "spy_xsmom_core_satellite"
    assert config.version == "3.0.0"
    assert config.allocation.core_weight == 0.69
    assert config.allocation.satellite_weight == 0.30
    assert config.allocation.cash_weight == 0.01
    assert config.features.rl_enabled is False
    assert len(config.sha256) == 64

    with pytest.raises(FrozenInstanceError):
        config.allocation.core_weight = 0.50

    payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    reordered = tmp_path / "same.json"
    reordered.write_text(json.dumps(payload, indent=7, sort_keys=True), encoding="utf-8")
    assert load_strategy_config(reordered).sha256 == config.sha256


def test_config_rejects_non_unit_allocation_and_rl(tmp_path):
    payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["allocation"]["cash_weight"] = 0.02
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="sum exactly"):
        load_strategy_config(bad)

    payload["allocation"]["cash_weight"] = 0.01
    payload["features"]["rl_enabled"] = True
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="quarantined"):
        load_strategy_config(bad)

    payload["features"]["rl_enabled"] = "false"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="JSON boolean"):
        load_strategy_config(bad)


def test_snapshot_hash_is_stable_across_column_order():
    left = pd.DataFrame({"b": [2, 4], "a": [1, 3]}, index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    right = left[["a", "b"]]
    assert snapshot_sha256(left) == snapshot_sha256(right)


def _open_snapshots():
    dates = pd.bdate_range("2026-01-01", periods=5)
    return {
        "constituents": pd.DataFrame(
            {
                "symbol": ["AAA", "OLD"],
                "effective_start": ["2020-01-01", "2020-01-01"],
                "effective_end": [None, "2025-12-31"],
            }
        ),
        "total_return_prices": pd.DataFrame({"AAA": range(5)}, index=dates),
        "execution_prices": pd.DataFrame({"AAA": range(5)}, index=dates),
        "corporate_actions": pd.DataFrame({"AAA": [0, 0, 0, 0, 0]}, index=dates),
        "delistings": pd.DataFrame({"AAA": [None] * 5}, index=dates),
        "symbol_changes": pd.DataFrame(
            columns=["old_symbol", "new_symbol", "effective_date"]
        ),
        "sector_history": pd.DataFrame({"symbol": ["AAA"], "sector": ["Technology"]}),
        "security_master": pd.DataFrame(
            {"symbol": ["AAA"], "issuer_id": ["issuer-a"], "share_class": ["A"]}
        ),
        "benchmark_sector_weights": {"Technology": 1.0},
        "benchmark_total_return": pd.Series(range(5), index=dates, name="SPY"),
        "adv": pd.Series({"AAA": 1_000_000.0}),
        "historical_spreads": pd.DataFrame({"AAA": [5.0] * 5}, index=dates),
        "quotes": pd.DataFrame(
            {
                "symbol": ["AAA"],
                "bid": [100.0],
                "ask": [100.1],
                "quote_time": [dates[-1]],
            }
        ),
    }


def test_open_provider_implements_protocol_but_cannot_self_promote():
    provider = OpenResearchProvider(
        _open_snapshots(),
        availability_at=datetime(2026, 1, 7, tzinfo=timezone.utc),
        frozen_at=datetime(2026, 1, 7, tzinfo=timezone.utc),
    )
    assert isinstance(provider, PointInTimeDataProvider)
    readiness = provider.readiness()
    assert readiness.promotable is False
    assert set(m.dataset for m in readiness.manifests).issuperset(REQUIRED_PORTFOLIO_DATASETS)
    assert all(m.tier is DataTier.RESEARCH_ONLY for m in readiness.manifests)
    assert all("not_promotable:" in blocker for blocker in readiness.blockers)
    current = provider.get_constituents(pd.Timestamp("2026-01-07"))
    assert current["symbol"].tolist() == ["AAA"]


def test_provider_never_returns_data_after_as_of():
    provider = OpenResearchProvider(_open_snapshots())
    prices = provider.get_total_return_prices(
        ["AAA"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-05")
    )
    assert prices.index.max() <= pd.Timestamp("2026-01-05")
    with pytest.raises(DataContractError, match="missing"):
        provider.manifest("filing_vintages")


def test_promotable_manifest_readiness_requires_all_datasets():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifests = [
        DataManifest.from_snapshot(
            name,
            "licensed-pit",
            [1],
            availability_at=now,
            frozen_at=now,
            coverage=1.0,
            tier=DataTier.PROMOTABLE,
        )
        for name in REQUIRED_PORTFOLIO_DATASETS
    ]
    assert DataReadiness.evaluate(manifests).promotable
    missing = DataReadiness.evaluate(manifests[:-1])
    assert not missing.promotable
    assert missing.blockers == (f"missing:{REQUIRED_PORTFOLIO_DATASETS[-1]}",)
