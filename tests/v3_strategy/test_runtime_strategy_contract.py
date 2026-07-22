from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from src.v3.portfolio import PortfolioConstructor
from src.v3.runtime_input import load_runtime_input

from .conftest import make_portfolio_inputs


def test_aware_runtime_signal_and_naive_price_sessions_construct_end_to_end(tmp_path):
    source = make_portfolio_inputs(n_symbols=300, manifests=())
    signal = source.signal_date.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    payload = {
        "signal_date": signal.tz_convert("UTC").isoformat(),
        "total_return_prices": {
            "dates": [value.isoformat() for value in source.total_return_prices.index],
            "values": {
                symbol: values.tolist()
                for symbol, values in source.total_return_prices.items()
            },
        },
        "securities": [
            {
                "symbol": security.symbol,
                "issuer_id": security.issuer_id,
                "sector": security.sector,
                "adv_30d": security.adv_30d,
            }
            for security in source.securities
        ],
        "benchmark_sector_weights": dict(source.benchmark_sector_weights),
        "prior_de_risked": False,
        "satellite_reentry_approved": True,
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = load_runtime_input(path)
    normal = PortfolioConstructor().construct(bundle.portfolio_inputs())
    durable_override = PortfolioConstructor().construct(
        bundle.portfolio_inputs(
            prior_de_risked=True,
            satellite_reentry_approved=False,
        )
    )

    assert normal.cash_weight == 0.01
    assert dict(durable_override.diagnostics)["construction"] == "drawdown_de_risk"
    assert durable_override.cash_weight == 0.31


def _small_payload(quote: float) -> dict:
    observed = datetime(2026, 6, 30, 15, 0, tzinfo=UTC)
    return {
        "signal_date": datetime(2026, 6, 30, 20, 0, tzinfo=UTC).isoformat(),
        "total_return_prices": {
            "dates": ["2026-06-27", "2026-06-30"],
            "values": {"SPY": [600.0, 606.0], "AAA": [100.0, 102.0]},
        },
        "securities": [
            {"symbol": "AAA", "issuer_id": "issuer-a", "sector": "Tech", "adv_30d": 1_000_000}
        ],
        "benchmark_sector_weights": {"Tech": 1.0},
        "manifests": [
            {
                "dataset": "constituents",
                "source": "stable-pit-fixture",
                "availability_at": "2026-06-30T19:00:00+00:00",
                "frozen_at": "2026-06-30T20:00:00+00:00",
                "row_count": 1,
                "coverage": 1.0,
                "sha256": "a" * 64,
                "tier": "RESEARCH_ONLY",
                "warnings": ["fixture"],
            }
        ],
        "quotes": {
            "AAA": {
                "bid_price": quote,
                "ask_price": quote + 0.1,
                "observed_at": observed.isoformat(),
                "adv_dollars_30d": 1_000_000,
            }
        },
    }


def test_research_identity_excludes_ephemeral_quote_and_benchmark_is_purpose_optional(tmp_path):
    paths = []
    for index, quote in enumerate((100.0, 101.0)):
        path = tmp_path / f"bundle-{index}.json"
        path.write_text(json.dumps(_small_payload(quote)), encoding="utf-8")
        paths.append(path)
    first, second = (load_runtime_input(path) for path in paths)

    assert first.research_data_sha256 == second.research_data_sha256
    assert first.bundle_sha256 != second.bundle_sha256
    assert "missing_runtime_benchmark_mark" not in first.content_validation_blockers
    assert "unvalidated_runtime_content:quotes" in first.content_validation_blockers
