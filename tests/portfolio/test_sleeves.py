"""
Tests for sleeves.

Unit tests (no network) verify the base Sleeve weighting logic.
Smoke tests (network) confirm each concrete sleeve produces sensible output.
"""
import os
from datetime import datetime
from typing import Dict, Optional

import pytest

from src.portfolio.sleeves import Sleeve, SleeveResult


NETWORK_OK = os.getenv("RUN_NETWORK_TESTS", "1") == "1"


# ── Stub sleeve for unit tests (no network) ──────────────────────────────────


class StubSleeve(Sleeve):
    name = "stub"
    rebalance_freq = "monthly"
    target_positions = 5
    max_position_weight = 0.5
    min_score_for_inclusion = 0.0

    def __init__(self, scores: Dict[str, float]):
        self._scores = scores

    def universe(self):
        return list(self._scores.keys())

    def score(self, as_of: Optional[datetime] = None):
        return self._scores


def test_sleeve_empty_scores():
    s = StubSleeve({})
    res = s.weights()
    assert res.weights == {}
    assert res.n_candidates == 0
    assert not res.is_active()


def test_sleeve_weights_sum_to_one():
    s = StubSleeve({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0})
    res = s.weights()
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6
    # Higher score → higher weight
    assert res.weights["E"] >= res.weights["A"]


def test_sleeve_respects_target_positions():
    scores = {f"T{i}": float(i) for i in range(20)}
    s = StubSleeve(scores)
    res = s.weights()
    assert len(res.weights) == 5   # target_positions
    # Should be the top 5 scores
    assert set(res.weights.keys()) == {"T15", "T16", "T17", "T18", "T19"}


def test_sleeve_caps_max_position():
    # Big spread → without cap the top would get ~all the weight
    scores = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 100.0}
    s = StubSleeve(scores)
    res = s.weights()
    # No single position above the cap (0.5)
    assert max(res.weights.values()) <= 0.5 + 1e-9
    # Weights still sum to ~1
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6


def test_sleeve_filters_below_min_score():
    class HighMinSleeve(StubSleeve):
        min_score_for_inclusion = 3.0
        target_positions = 10

    s = HighMinSleeve({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0,
                        "F": 6.0, "G": 7.0, "H": 8.0})
    res = s.weights()
    # All picks should have score >= 3.0
    for t in res.weights:
        assert s._scores[t] >= 3.0


def test_sleeve_drops_nan_scores():
    s = StubSleeve({"A": 1.0, "B": float("nan"), "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0})
    res = s.weights()
    assert "B" not in res.weights


# ── Smoke tests (network) ────────────────────────────────────────────────────


@pytest.mark.skipif(not NETWORK_OK, reason="RUN_NETWORK_TESTS=0")
def test_vqm_sleeve_smoke():
    """ValueQualityMomentum sleeve should produce a non-trivial portfolio."""
    from src.portfolio.sleeves import ValueQualityMomentumSleeve
    # Use a smaller universe to keep test fast
    sleeve = ValueQualityMomentumSleeve()
    sleeve.target_positions = 10
    # Override universe to keep network calls bounded
    sleeve.universe = lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "JPM", "BAC", "WFC", "V", "MA",
        "JNJ", "PFE", "LLY", "XOM", "CVX",
        "WMT", "COST", "PG", "KO",
    ]
    res = sleeve.weights()
    assert isinstance(res, SleeveResult)
    assert res.is_active(), "VQM sleeve produced no positions"
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6
    print(f"\nVQM sleeve picks ({len(res.weights)}): {sorted(res.weights.items(), key=lambda kv: -kv[1])[:5]}")


@pytest.mark.skipif(not NETWORK_OK, reason="RUN_NETWORK_TESTS=0")
def test_xs_momentum_sleeve_smoke():
    from src.portfolio.sleeves import CrossSectionalMomentumSleeve
    sleeve = CrossSectionalMomentumSleeve()
    sleeve.target_positions = 10
    sleeve.universe = lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "TSLA",
        "JPM", "BAC", "WFC", "GS", "MS",
        "XOM", "CVX", "COP", "JNJ", "PFE", "LLY", "MRK",
    ]
    res = sleeve.weights()
    assert res.is_active()
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6
    print(f"\nXSMomentum picks ({len(res.weights)}): {sorted(res.weights.items(), key=lambda kv: -kv[1])[:5]}")


@pytest.mark.skipif(not NETWORK_OK, reason="RUN_NETWORK_TESTS=0")
def test_pead_sleeve_smoke():
    from src.portfolio.sleeves import PEADSleeve
    sleeve = PEADSleeve()
    sleeve.target_positions = 10
    sleeve.universe = lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "JPM", "BAC", "V", "MA", "JNJ", "PFE", "LLY",
        "XOM", "CVX", "WMT", "COST", "HD", "DIS", "NFLX",
    ]
    res = sleeve.weights()
    # PEAD may be empty if no recent earnings — that's OK
    assert isinstance(res, SleeveResult)
    if res.is_active():
        assert abs(sum(res.weights.values()) - 1.0) < 1e-6
    print(f"\nPEAD picks ({len(res.weights)}): {res.weights}")
