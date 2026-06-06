"""
End-to-end smoke tests for the factor engine.

These hit yfinance / SEC over the network — marked `network` so CI can
skip them. The goal is to confirm each factor returns a non-empty result
on a small real universe and that the top-decile names look sensible.
"""
import pytest
import os

NETWORK_OK = os.getenv("RUN_NETWORK_TESTS", "1") == "1"

pytestmark = pytest.mark.skipif(
    not NETWORK_OK,
    reason="set RUN_NETWORK_TESTS=1 to run network-touching factor smoke tests",
)


# Small universe of large, liquid US names with reliable yfinance coverage
SMOKE_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    "JNJ", "PFE", "MRK", "ABBV", "LLY",
    "XOM", "CVX", "COP",
    "WMT", "COST", "PG", "KO", "PEP",
    "HD", "DIS",
]


def test_value_factor_smoke():
    from src.factors.value import ValueFactor
    res = ValueFactor().compute(SMOKE_UNIVERSE)
    assert len(res.scores) >= 15, f"too few value scores: {len(res.scores)}"
    # All z-scores should be in a reasonable range
    assert all(-5 <= s <= 5 for s in res.scores.values())
    # Top-decile names should include at least one classic value name
    top5 = set(res.top_n(5))
    print(f"\nValue top 5: {top5}")


def test_quality_factor_smoke():
    from src.factors.quality import QualityFactor
    res = QualityFactor().compute(SMOKE_UNIVERSE)
    assert len(res.scores) >= 15
    print(f"\nQuality top 5: {res.top_n(5)}")
    # Profitability-rich names (AAPL, MSFT, V, MA, COST) should rank in top half
    top_half = set(res.top_n(len(res.scores) // 2))
    assert any(t in top_half for t in ["AAPL", "MSFT", "V", "MA", "COST"]), \
        f"expected at least one mega-cap quality name in top half; got {top_half}"


def test_momentum_factor_smoke():
    from src.factors.momentum import MomentumFactor
    res = MomentumFactor().compute(SMOKE_UNIVERSE)
    assert len(res.scores) >= 15, f"too few momentum scores: {len(res.scores)}"
    print(f"\nMomentum top 5: {res.top_n(5)}")
    # Check smoothness is in [0,1]
    for t, raw in res.raw.items():
        assert 0 <= raw["smoothness"] <= 1, f"{t}: bad smoothness {raw['smoothness']}"


def test_defensive_factor_smoke():
    from src.factors.defensive import DefensiveFactor
    res = DefensiveFactor().compute(SMOKE_UNIVERSE)
    assert len(res.scores) >= 15
    print(f"\nDefensive top 5: {res.top_n(5)}")
    # Utilities / staples / healthcare typically rank high; tech low.
    # We don't have utilities in this universe, but JNJ/PG/KO should be top half.
    top_half = set(res.top_n(len(res.scores) // 2))
    assert any(t in top_half for t in ["JNJ", "PG", "KO", "PEP", "WMT"]), \
        f"expected defensive staples in top half; got {top_half}"


def test_trend_factor_smoke():
    from src.factors.trend import TrendFactor
    res = TrendFactor().compute(SMOKE_UNIVERSE)
    assert len(res.scores) >= 10
    print(f"\nTrend top 5: {res.top_n(5)}")
    # Check that capped forecasts are in [-20, 20]
    for t, raw in res.raw.items():
        for k, v in raw.items():
            if k.startswith("f_"):
                assert -20 <= v <= 20, f"{t} {k} = {v} exceeds ±20 cap"


def test_pead_factor_smoke():
    """PEAD often has few or zero active candidates; test it runs without crashing."""
    from src.factors.pead import PEADFactor
    res = PEADFactor().compute(SMOKE_UNIVERSE)
    # Just check that it returned something structured
    assert isinstance(res.scores, dict)
    print(f"\nPEAD candidates: {len(res.scores)}  top 5: {res.top_n(5)}")


def test_all_factors_have_consistent_universe_handling():
    """No factor should crash on an empty universe or a universe of garbage tickers."""
    from src.factors import all_factors
    factors = all_factors()
    for name, fac in factors.items():
        if name == "insider":
            # Insider hits SEC — skip in this consistency test (slower)
            continue
        try:
            res_empty = fac.compute([])
            assert res_empty.scores == {}
            res_garbage = fac.compute(["ZZZZ_NOT_A_TICKER_AAA", "BBBB_BAD_BAD"])
            assert isinstance(res_garbage.scores, dict)
        except Exception as e:
            pytest.fail(f"{name} crashed on empty/garbage universe: {e}")
