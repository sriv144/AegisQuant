"""Unit tests for Factor base class — no network required."""
import math
import pytest

from src.factors.base import Factor, FactorResult


def test_zscore_basic():
    vals = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
    z = Factor.zscore(vals)
    # mean should be ~0, sum near 0
    assert abs(sum(z.values())) < 1e-9
    # monotonic preserving
    ordered = [z[k] for k in ["A", "B", "C", "D", "E"]]
    assert all(ordered[i] < ordered[i + 1] for i in range(4))


def test_zscore_winsorize():
    vals = {f"T{i}": float(i) for i in range(10)}
    vals["OUTLIER"] = 1e6
    z = Factor.zscore(vals, winsorize=3.0)
    # Outlier is clipped at +3
    assert z["OUTLIER"] == 3.0


def test_zscore_drops_nan_and_none():
    vals = {"A": 1.0, "B": None, "C": float("nan"), "D": 2.0, "E": 3.0, "F": 4.0, "G": 5.0}
    z = Factor.zscore(vals)
    assert "B" not in z and "C" not in z
    assert all(math.isfinite(v) for v in z.values())


def test_zscore_too_small_returns_zeros():
    z = Factor.zscore({"A": 1.0, "B": 2.0})  # only 2 → < 5 minimum
    assert z == {"A": 0.0, "B": 0.0}


def test_zscore_zero_variance():
    z = Factor.zscore({f"T{i}": 5.0 for i in range(10)})
    assert all(v == 0.0 for v in z.values())


def test_rank_pct():
    vals = {"A": 10, "B": 20, "C": 30, "D": 40, "E": 50}
    r = Factor.rank_pct(vals, higher_is_better=True)
    assert r["A"] < r["E"]
    assert abs(r["E"] - 1.0) < 1e-9


def test_factor_result_topn():
    res = FactorResult(
        factor_name="t", as_of=None,
        scores={"A": 1.0, "B": 3.0, "C": 2.0, "D": -1.0},
    )
    assert res.top_n(2) == ["B", "C"]
    assert res.bottom_n(2) == ["D", "A"]


def test_factor_result_to_frame():
    res = FactorResult(
        factor_name="t", as_of=None,
        scores={"A": 1.0, "B": 2.0},
        confidence={"A": 0.5, "B": 0.9},
        raw={"A": {"x": 1.0}, "B": {"x": 2.0}},
    )
    df = res.to_frame()
    assert list(df.index) == ["B", "A"]   # sorted by score descending
    assert df.loc["A", "confidence"] == 0.5
    assert df.loc["B", "x"] == 2.0
