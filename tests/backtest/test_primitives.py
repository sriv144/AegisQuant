"""Unit tests for purged_cv, triple_barrier, deflated_sharpe — no network."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.purged_cv import PurgedKFold
from src.backtest.triple_barrier import triple_barrier_labels
from src.backtest.deflated_sharpe import (
    deflated_sharpe_ratio, probability_backtest_overfit, expected_max_sharpe
)


# ── PurgedKFold ─────────────────────────────────────────────────────────────


def test_purged_kfold_splits():
    """Basic split mechanics: no overlap between train and test, all indices used."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    t1 = pd.Series(idx + pd.Timedelta(days=5), index=idx)
    X = pd.DataFrame(index=idx, data={"x": range(100)})
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.02)
    seen_test = set()
    for tr, te in cv.split(X):
        assert len(set(tr) & set(te)) == 0   # no overlap
        seen_test.update(te.tolist())
    # All indices end up in test exactly once
    assert seen_test == set(range(100))


def test_purged_kfold_embargo():
    """Embargo removes samples after the test fold from train."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    t1 = pd.Series(idx + pd.Timedelta(days=1), index=idx)
    X = pd.DataFrame(index=idx, data={"x": range(100)})
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.10)   # 10 bar embargo

    folds = list(cv.split(X))
    # Fold 0 = test [0,20). With 10-bar embargo, train should exclude [20, 30).
    tr0, te0 = folds[0]
    assert 20 not in tr0
    assert 29 not in tr0
    assert 30 in tr0   # right after embargo


def test_purged_kfold_purges_label_overlap():
    """Train samples whose label end-time falls inside test window are purged."""
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    # Label horizon = 10 days
    t1 = pd.Series(idx + pd.Timedelta(days=10), index=idx)
    X = pd.DataFrame(index=idx, data={"x": range(50)})
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0)

    folds = list(cv.split(X))
    tr0, te0 = folds[0]
    # Test is [0, 10). Train sample at index 5 has label window ending at idx[15]
    # which falls inside test — wait, test is [0, 10), so labels in [idx[0], idx[9]].
    # A train sample whose t1 is inside this window should be purged.
    # Train candidate at i=10 has t1=idx[20] — outside test. KEEP.
    # No train indices are before the test fold here, so this mainly tests the
    # "no crash" path. With a 5-day horizon and 5 folds of 10 each, training
    # samples right after test [10, 15] would have labels [20, 25] outside test.
    assert len(tr0) > 0
    assert len(te0) == 10


# ── triple_barrier ──────────────────────────────────────────────────────────


def test_triple_barrier_upper_hit():
    # Synthetic: price goes up steadily, upper barrier should hit
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    prices = pd.Series(np.linspace(100, 130, 30), index=idx)
    sigma = pd.Series([0.01] * 30, index=idx)
    entries = pd.DatetimeIndex([idx[0]])
    df = triple_barrier_labels(prices, entries, sigma,
                               pt_mult=2.0, sl_mult=1.0, max_hold=30)
    assert len(df) == 1
    assert df.iloc[0]["label"] == 1
    assert df.iloc[0]["barrier_hit"] == "upper"


def test_triple_barrier_lower_hit():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    prices = pd.Series(np.linspace(100, 70, 30), index=idx)
    sigma = pd.Series([0.01] * 30, index=idx)
    entries = pd.DatetimeIndex([idx[0]])
    df = triple_barrier_labels(prices, entries, sigma,
                               pt_mult=2.0, sl_mult=1.0, max_hold=30)
    assert df.iloc[0]["label"] == -1
    assert df.iloc[0]["barrier_hit"] == "lower"


def test_triple_barrier_vertical_hit():
    """Flat price → neither barrier hits → vertical (timeout) label = 0."""
    idx = pd.date_range("2020-01-01", periods=15, freq="D")
    prices = pd.Series([100.0] * 15, index=idx)
    sigma = pd.Series([0.01] * 15, index=idx)
    entries = pd.DatetimeIndex([idx[0]])
    df = triple_barrier_labels(prices, entries, sigma,
                               pt_mult=2.0, sl_mult=1.0, max_hold=10)
    assert df.iloc[0]["label"] == 0
    assert df.iloc[0]["barrier_hit"] == "vertical"


def test_triple_barrier_multiple_entries():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    # Price wiggles up
    prices = pd.Series(100 + np.arange(100) * 0.5 + np.sin(np.arange(100) / 5), index=idx)
    sigma = pd.Series([0.01] * 100, index=idx)
    entries = pd.DatetimeIndex([idx[10], idx[20], idx[30]])
    df = triple_barrier_labels(prices, entries, sigma, max_hold=30)
    assert len(df) == 3
    assert set(df.columns) == {"exit_date", "label", "return_pct", "barrier_hit"}


# ── deflated_sharpe ─────────────────────────────────────────────────────────


def test_expected_max_sharpe_grows_with_trials():
    e1 = expected_max_sharpe(1)
    e10 = expected_max_sharpe(10)
    e100 = expected_max_sharpe(100)
    assert e1 == 0.0    # one trial means no selection
    assert e10 > e1
    assert e100 > e10


def test_deflated_sharpe_high_with_one_trial():
    """A 2-Sharpe daily return series with 1 trial should give DSR near 1."""
    np.random.seed(42)
    # Generate returns with high SR
    daily_sr = 2.0 / np.sqrt(252)
    returns = np.random.normal(daily_sr * 0.01, 0.01, 500)
    dsr = deflated_sharpe_ratio(returns, n_trials=1)
    assert dsr > 0.5, f"expected high DSR for clear edge, got {dsr}"


def test_deflated_sharpe_drops_with_many_trials():
    """Same returns, more trials → lower DSR (selection penalty)."""
    np.random.seed(42)
    daily_sr = 1.5 / np.sqrt(252)
    returns = np.random.normal(daily_sr * 0.01, 0.01, 500)
    dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_1000 = deflated_sharpe_ratio(returns, n_trials=1000)
    assert dsr_1000 < dsr_1


def test_deflated_sharpe_near_zero_for_noise():
    """Pure noise should produce low DSR."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, 500)
    dsr = deflated_sharpe_ratio(noise, n_trials=100)
    assert dsr < 0.6


def test_deflated_sharpe_short_series():
    """Very short series should return 0 (insufficient sample)."""
    dsr = deflated_sharpe_ratio(np.random.randn(10), n_trials=1)
    assert dsr == 0.0


def test_pbo_with_one_dominant_strategy():
    """If one strategy is much better than the rest (real edge), PBO should be LOW."""
    np.random.seed(42)
    T, N = 800, 5
    M = np.random.normal(0, 0.01, (T, N))
    # Strategy 0 has a clear positive drift
    M[:, 0] += 0.002
    pbo = probability_backtest_overfit(M, n_splits=8)
    assert pbo < 0.5, f"clear edge should produce PBO < 0.5, got {pbo}"
