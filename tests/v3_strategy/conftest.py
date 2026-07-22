from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from src.v3.data import DataManifest, DataTier, REQUIRED_PORTFOLIO_DATASETS
from src.v3.portfolio import PortfolioInputs, SecurityMetadata


def promotable_manifests(signal_date: pd.Timestamp) -> tuple[DataManifest, ...]:
    available = pd.Timestamp(signal_date).to_pydatetime().replace(tzinfo=timezone.utc)
    frozen = available
    return tuple(
        DataManifest.from_snapshot(
            dataset,
            "validated-fixture",
            [dataset],
            availability_at=available,
            frozen_at=frozen,
            coverage=1.0,
            tier=DataTier.PROMOTABLE,
        )
        for dataset in REQUIRED_PORTFOLIO_DATASETS
    )


def make_portfolio_inputs(
    *,
    n_symbols: int = 400,
    manifests: tuple[DataManifest, ...] | None = None,
    common_active_return: np.ndarray | None = None,
) -> PortfolioInputs:
    dates = pd.bdate_range("2024-01-02", periods=300)
    t = np.arange(len(dates), dtype=float)
    spy_daily = 0.00035 + 0.0007 * np.sin(t / 9.0)
    spy = 100.0 * np.exp(np.cumsum(spy_daily))
    data: dict[str, np.ndarray] = {"SPY": spy}
    securities: list[SecurityMetadata] = []
    active = np.zeros(len(dates)) if common_active_return is None else np.asarray(common_active_return, dtype=float)
    if len(active) != len(dates):
        raise ValueError("common_active_return must match the fixture dates")
    for index in range(n_symbols):
        symbol = f"S{index:03d}"
        alpha = -0.0001 + index * 0.0000015
        wobble = 0.00002 * np.sin(t / (5.0 + index % 7))
        daily = spy_daily + alpha + wobble + active
        data[symbol] = (40.0 + index / 10.0) * np.exp(np.cumsum(daily))
        securities.append(
            SecurityMetadata(
                symbol=symbol,
                issuer_id=f"ISSUER-{index:03d}",
                sector="Technology" if index % 2 == 0 else "Health Care",
                adv_30d=1_000_000.0 + index,
            )
        )
    prices = pd.DataFrame(data, index=dates)
    effective_manifests = promotable_manifests(dates[-1]) if manifests is None else manifests
    return PortfolioInputs(
        signal_date=dates[-1],
        total_return_prices=prices,
        securities=tuple(securities),
        benchmark_sector_weights={"Technology": 0.5, "Health Care": 0.5},
        manifests=effective_manifests,
    )


@pytest.fixture
def portfolio_inputs() -> PortfolioInputs:
    return make_portfolio_inputs()
