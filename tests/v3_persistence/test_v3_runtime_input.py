from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.v3.runtime_input import load_runtime_input


def test_frozen_runtime_input_is_content_addressed_and_typed(tmp_path: Path) -> None:
    now = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    payload = {
        "signal_date": now.isoformat(),
        "total_return_prices": {
            "dates": ["2026-06-27", "2026-06-30"],
            "values": {"SPY": [600, 606], "AAA": [100, 102]},
        },
        "securities": [
            {"symbol": "AAA", "issuer_id": "issuer-a", "sector": "Tech", "adv_30d": 1_000_000}
        ],
        "benchmark_sector_weights": {"Tech": 1.0},
        "manifests": [
            {
                "dataset": "constituents",
                "source": "fixture",
                "availability_at": now.isoformat(),
                "frozen_at": now.isoformat(),
                "row_count": 1,
                "coverage": 1.0,
                "sha256": "a" * 64,
                "tier": "RESEARCH_ONLY",
                "warnings": ["fixture"],
            }
        ],
        "quotes": {
            "SPY": {
                "bid_price": 605,
                "ask_price": 607,
                "observed_at": now.isoformat(),
                "adv_dollars_30d": 10_000_000,
            }
        },
        "scheduled_eligible_sessions": ["2026-07-01", "2026-07-02", "2026-07-06"],
        "starting_nav": "100000",
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = load_runtime_input(path)

    assert len(bundle.bundle_sha256) == 64
    assert list(bundle.total_return_prices.columns) == ["SPY", "AAA"]
    assert bundle.securities[0].symbol == "AAA"
    assert bundle.quotes["SPY"].midpoint == 606
    assert bundle.scheduled_eligible_sessions[0].isoformat() == "2026-07-01"
