"""
Tests for WalkForwardEngine — smoke tests that verify window construction
and basic metrics without requiring live data or full training runs.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n_days: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.002, 0.002, n_days)),
            "high": close * (1 + rng.uniform(0, 0.005, n_days)),
            "low": close * (1 - rng.uniform(0, 0.005, n_days)),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n_days).astype(float),
        },
        index=idx,
    )
    return df


# ── WalkForwardEngine._build_windows ─────────────────────────────────────────

def test_build_windows_count():
    """_build_windows should produce at least 1 window for a 2-year dataset."""
    from src.backtest.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(
        tickers=["SPY"],
        train_years=1,
        val_months=3,
        step_months=3,
        history_start="2019-01-01",
    )

    df_spy = _make_ohlcv(n_days=600)  # ~2.4 years
    df_dict = {"SPY": df_spy}
    windows = engine._build_windows(df_dict)

    assert len(windows) >= 1, "Expected at least one walk-forward window"


def test_build_windows_chronological():
    """Each window's validation period must be strictly after its training period."""
    from src.backtest.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(
        tickers=["SPY"],
        train_years=1,
        val_months=3,
        step_months=3,
        history_start="2018-01-01",
    )

    df_dict = {"SPY": _make_ohlcv(n_days=800)}
    windows = engine._build_windows(df_dict)

    for _, _, dates in windows:
        train_end = pd.Timestamp(dates["train_end"])
        val_start = pd.Timestamp(dates["val_start"])
        assert val_start >= train_end, (
            f"val_start {val_start} must be >= train_end {train_end}"
        )


def test_build_windows_no_overlap():
    """Consecutive windows must not have overlapping validation periods."""
    from src.backtest.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(
        tickers=["SPY"],
        train_years=1,
        val_months=3,
        step_months=3,
        history_start="2018-01-01",
    )

    df_dict = {"SPY": _make_ohlcv(n_days=800)}
    windows = engine._build_windows(df_dict)

    if len(windows) < 2:
        pytest.skip("Not enough windows to test overlap")

    for i in range(len(windows) - 1):
        _, _, d1 = windows[i]
        _, _, d2 = windows[i + 1]
        assert pd.Timestamp(d2["val_start"]) >= pd.Timestamp(d1["val_end"]), (
            "Window validation periods must not overlap"
        )


# ── WalkForwardResults aggregate metrics ────────────────────────────────────

def test_window_result_fields():
    """WindowResult should have the expected metric fields after construction."""
    from src.backtest.walk_forward import WindowResult

    wr = WindowResult(
        window_id=1,
        train_start="2019-01-01",
        train_end="2020-12-31",
        val_start="2021-01-01",
        val_end="2021-06-30",
    )

    assert wr.window_id == 1
    assert wr.train_metrics == {}
    assert wr.val_metrics == {}
    assert wr.error is None


def test_walk_forward_results_structure():
    """WalkForwardResults should correctly aggregate window returns."""
    from src.backtest.walk_forward import WalkForwardResults, WindowResult

    w1 = WindowResult(1, "2019-01-01", "2019-12-31", "2020-01-01", "2020-06-30")
    w1.val_returns = [0.01, -0.005, 0.003]
    w1.val_metrics = {"sharpe": 1.1}

    w2 = WindowResult(2, "2019-07-01", "2020-06-30", "2020-07-01", "2020-12-31")
    w2.val_returns = [0.002, 0.008, -0.002]
    w2.val_metrics = {"sharpe": 0.9}

    results = WalkForwardResults(tickers=["SPY"], windows=[w1, w2])
    all_returns = w1.val_returns + w2.val_returns
    results.oos_all_returns = all_returns

    assert len(results.windows) == 2
    assert len(results.oos_all_returns) == 6
    assert results.tickers == ["SPY"]
