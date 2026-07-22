"""
Tests for Combiner and RiskOfficer.

Unit tests use synthetic SleeveResults and a stubbed DataProvider so they
run offline. Smoke tests at the end hit real data for end-to-end verification.
"""
import os
from datetime import datetime
from typing import Dict
from unittest.mock import MagicMock

import pytest
import pandas as pd
import numpy as np

from src.portfolio.sleeves import SleeveResult
from src.portfolio.combiner import Combiner, PortfolioTarget
from src.portfolio.risk_officer import RiskOfficer, RiskReview


NETWORK_OK = os.getenv("RUN_NETWORK_TESTS", "1") == "1"


def _stub_dp(vols: Dict[str, float] = None, sectors: Dict[str, str] = None,
             betas: Dict[str, float] = None):
    """Build a stub DataProvider with controllable vol/sector/beta."""
    vols = vols or {}
    sectors = sectors or {}
    betas = betas or {}
    dp = MagicMock()

    def get_prices(tickers, *a, **kw):
        # Build a fake price frame where the per-ticker vol matches `vols`
        n = 180
        idx = pd.bdate_range(end="2026-05-23", periods=n)
        data = {}
        rng = np.random.default_rng(42)
        for t in tickers:
            v = vols.get(t, 0.20)
            daily_vol = v / np.sqrt(252)
            rets = rng.normal(0.0005, daily_vol, n)
            prices = 100.0 * np.exp(np.cumsum(rets))
            data[t] = prices
        return pd.DataFrame(data, index=idx)

    def get_sector(t):
        return sectors.get(t, "Unknown")

    def get_fundamentals(t):
        return {"beta": betas.get(t, 1.0)} if t in betas else {}

    dp.get_prices = get_prices
    dp.get_sector = get_sector
    dp.get_fundamentals = get_fundamentals
    return dp


# ── Combiner unit tests ─────────────────────────────────────────────────────


def test_combiner_empty_input():
    c = Combiner(data_provider=_stub_dp())
    out = c.combine({})
    assert isinstance(out, PortfolioTarget)
    assert out.ticker_weights == {}
    assert out.cash_weight == 1.0


def test_combiner_single_sleeve():
    c = Combiner(data_provider=_stub_dp(vols={"A": 0.20, "B": 0.20}))
    sr = SleeveResult(sleeve_name="s1", as_of=datetime.utcnow(),
                      weights={"A": 0.6, "B": 0.4})
    out = c.combine({"s1": sr})
    # Single sleeve with cap of 0.4 → sleeve_w = 0.4, ticker weights scale
    assert out.sleeve_weights["s1"] <= 0.40 + 1e-6
    # ticker_weight = sleeve_w * in_sleeve_w
    assert out.ticker_weights["A"] == pytest.approx(out.sleeve_weights["s1"] * 0.6, rel=1e-3)


def test_combiner_two_sleeves_equal_vol():
    c = Combiner(data_provider=_stub_dp(vols={"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20}))
    s1 = SleeveResult("s1", datetime.utcnow(), weights={"A": 0.5, "B": 0.5})
    s2 = SleeveResult("s2", datetime.utcnow(), weights={"C": 0.5, "D": 0.5})
    out = c.combine({"s1": s1, "s2": s2})
    # Equal vol → equal weight (each ~0.4 after cap)
    assert out.sleeve_weights["s1"] == pytest.approx(out.sleeve_weights["s2"], rel=1e-3)
    # Each ticker = 0.5 in-sleeve * 0.4 sleeve_w = 0.20
    assert all(abs(w - 0.20) < 1e-3 for w in out.ticker_weights.values())


def test_combiner_inverse_vol_weighting():
    """
    High-vol sleeve should get less weight than low-vol sleeve.

    With only 2 sleeves both would cap at 40% (MAX_SLEEVE_NAV), masking the
    inverse-vol signal. Use 3 sleeves so the cap doesn't bind on all of them.
    """
    c = Combiner(data_provider=_stub_dp(vols={"A": 0.10, "B": 0.40, "C": 0.40}))
    s1 = SleeveResult("low_vol", datetime.utcnow(), weights={"A": 1.0})
    s2 = SleeveResult("high_vol_1", datetime.utcnow(), weights={"B": 1.0})
    s3 = SleeveResult("high_vol_2", datetime.utcnow(), weights={"C": 1.0})
    out = c.combine({"low_vol": s1, "high_vol_1": s2, "high_vol_2": s3})
    # With 1/0.10 + 1/0.40 + 1/0.40 = 10+2.5+2.5 = 15, raw weights = [10/15, 2.5/15, 2.5/15]
    # = [0.667, 0.167, 0.167]. After cap at 0.40, low_vol = 0.40, excess 0.267 spreads
    # to the others: each gets 0.167 + 0.267/2 = 0.30.
    assert out.sleeve_weights["low_vol"] > out.sleeve_weights["high_vol_1"]
    assert out.sleeve_weights["low_vol"] > out.sleeve_weights["high_vol_2"]


def test_combiner_sleeve_cap_enforced():
    """No sleeve > 40% even if its inverse-vol weight would push higher."""
    c = Combiner(data_provider=_stub_dp(vols={"A": 0.05, "B": 0.50, "C": 0.50}))  # A is dirt-cheap vol
    s1 = SleeveResult("big", datetime.utcnow(), weights={"A": 1.0})
    s2 = SleeveResult("small", datetime.utcnow(), weights={"B": 1.0})
    s3 = SleeveResult("tiny", datetime.utcnow(), weights={"C": 1.0})
    out = c.combine({"big": s1, "small": s2, "tiny": s3})
    for k, w in out.sleeve_weights.items():
        assert w <= Combiner.MAX_SLEEVE_NAV + 1e-6, f"{k} = {w}"


def test_combiner_runtime_rollout_caps():
    """Runtime caps allow a staged rollout without changing global defaults."""
    c = Combiner(
        data_provider=_stub_dp(vols={"A": 0.20, "B": 0.20}),
        max_sleeve_nav=0.325,
        max_total_invested=0.65,
    )
    s1 = SleeveResult("xs_momentum", datetime.utcnow(), weights={"A": 1.0})
    s2 = SleeveResult("value_quality_momentum", datetime.utcnow(), weights={"B": 1.0})
    out = c.combine({"xs_momentum": s1, "value_quality_momentum": s2})
    assert out.sleeve_weights["xs_momentum"] == pytest.approx(0.325)
    assert out.sleeve_weights["value_quality_momentum"] == pytest.approx(0.325)
    assert out.total_invested == pytest.approx(0.65)


def test_combiner_macro_overlay_riskoff():
    """
    In risk-off, equity sleeves should shrink and defensive should grow.

    Use 3 sleeves with one big enough that the cap doesn't bind, so we can
    observe the macro overlay reshape weights.
    """
    c = Combiner(data_provider=_stub_dp(vols={"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20}))
    s1 = SleeveResult("xs_momentum", datetime.utcnow(), weights={"A": 0.5, "B": 0.5})    # equity beta
    s2 = SleeveResult("value_quality_momentum", datetime.utcnow(), weights={"C": 1.0})    # equity beta
    s3 = SleeveResult("insider_buying", datetime.utcnow(), weights={"D": 1.0})            # defensive
    risk_off = c.combine({"xs_momentum": s1, "value_quality_momentum": s2, "insider_buying": s3},
                         macro_regime_score=-3.0, macro_regime_confidence=1.0)
    risk_on = c.combine({"xs_momentum": s1, "value_quality_momentum": s2, "insider_buying": s3},
                        macro_regime_score=0.0)
    # Sum of equity sleeves should shrink in risk-off
    equity_off = risk_off.sleeve_weights["xs_momentum"] + risk_off.sleeve_weights["value_quality_momentum"]
    equity_on = risk_on.sleeve_weights["xs_momentum"] + risk_on.sleeve_weights["value_quality_momentum"]
    assert equity_off < equity_on, f"equity_off={equity_off:.3f} should be < equity_on={equity_on:.3f}"
    # Defensive sleeve should grow
    assert risk_off.sleeve_weights["insider_buying"] > risk_on.sleeve_weights["insider_buying"]


# ── RiskOfficer unit tests ──────────────────────────────────────────────────


def test_risk_officer_position_cap():
    ro = RiskOfficer(data_provider=_stub_dp())
    tgt = PortfolioTarget(
        as_of=datetime.utcnow(),
        ticker_weights={"A": 0.10, "B": 0.04, "C": 0.03},   # A exceeds 5% cap
        sleeve_weights={"s": 0.30},
    )
    review = ro.review(tgt)
    assert review.approved_weights["A"] == pytest.approx(0.05)
    assert any("Position cap" in v for v in review.violations)


def test_risk_officer_sector_cap():
    """
    Sector concentration > 20% -> scale all sector members down.

    Need 5 financials × 5% each = 25% sector total to trigger the cap.
    (Each individual is at the position cap, so position-cap-alone wouldn't catch it.)
    """
    sectors = {"BAC": "Financial Services", "JPM": "Financial Services",
               "MA": "Financial Services", "WFC": "Financial Services",
               "C": "Financial Services", "AAPL": "Technology"}
    ro = RiskOfficer(data_provider=_stub_dp(sectors=sectors))
    tgt = PortfolioTarget(
        as_of=datetime.utcnow(),
        ticker_weights={"BAC": 0.05, "JPM": 0.05, "MA": 0.05, "WFC": 0.05, "C": 0.05,
                        "AAPL": 0.05},
        sleeve_weights={"s": 0.30},
    )
    review = ro.review(tgt)
    fin_total = sum(review.approved_weights[t] for t in ["BAC", "JPM", "MA", "WFC", "C"])
    assert fin_total <= 0.20 + 1e-6, f"financials still at {fin_total*100:.1f}%"
    assert review.approved_weights["AAPL"] == 0.05   # untouched
    assert any("Sector cap" in v for v in review.violations)


def test_risk_officer_drawdown_gate():
    ro = RiskOfficer(data_provider=_stub_dp())
    tgt = PortfolioTarget(
        as_of=datetime.utcnow(),
        ticker_weights={"A": 0.04, "B": 0.04},
        sleeve_weights={"s": 0.10},
    )
    review = ro.review(tgt, current_drawdown=0.20)
    assert review.drawdown_scaling == 0.5
    assert review.approved_weights["A"] == pytest.approx(0.02)
    assert any("Drawdown gate" in v for v in review.violations)


def test_risk_officer_drawdown_no_gate():
    ro = RiskOfficer(data_provider=_stub_dp())
    tgt = PortfolioTarget(
        as_of=datetime.utcnow(),
        ticker_weights={"A": 0.04, "B": 0.04},
        sleeve_weights={"s": 0.10},
    )
    review = ro.review(tgt, current_drawdown=0.05)  # only 5%, no gate
    assert review.drawdown_scaling == 1.0
    assert review.approved_weights["A"] == 0.04


def test_risk_officer_clean_pass():
    """Well-diversified portfolio under all caps → no violations."""
    sectors = {"A": "Tech", "B": "Healthcare", "C": "Energy", "D": "Industrials"}
    ro = RiskOfficer(data_provider=_stub_dp(sectors=sectors))
    tgt = PortfolioTarget(
        as_of=datetime.utcnow(),
        ticker_weights={"A": 0.04, "B": 0.04, "C": 0.04, "D": 0.04},
        sleeve_weights={"s": 0.16},
    )
    review = ro.review(tgt)
    assert review.violations == []
    assert sum(review.approved_weights.values()) == pytest.approx(0.16)


def test_risk_officer_beta_cap_scales_down_high_beta_only():
    tickers = {f"T{i}": 0.05 for i in range(6)}
    ro = RiskOfficer(
        data_provider=_stub_dp(
            betas={t: 2.0 for t in tickers},
            sectors={t: f"Sector{i}" for i, t in enumerate(tickers)},
        ),
        enforce_beta=True,
        beta_max=0.5,
    )
    tgt = PortfolioTarget(
        as_of=datetime.utcnow(),
        ticker_weights=tickers,
        sleeve_weights={"s": 0.30},
    )
    review = ro.review(tgt)
    assert any("Beta cap" in v for v in review.violations)
    assert sum(review.approved_weights.values()) < 0.30


def test_risk_officer_beta_floor_does_not_increase_exposure():
    ro = RiskOfficer(
        data_provider=_stub_dp(betas={"A": 0.2}),
        enforce_beta=True,
        beta_min=0.4,
        max_position_nav=0.20,
    )
    tgt = PortfolioTarget(
        as_of=datetime.utcnow(),
        ticker_weights={"A": 0.10},
        sleeve_weights={"s": 0.10},
    )
    review = ro.review(tgt)
    assert any("Portfolio beta" in v for v in review.violations)
    assert review.approved_weights["A"] == pytest.approx(0.10)


# ── End-to-end smoke test ───────────────────────────────────────────────────


@pytest.mark.skipif(not NETWORK_OK, reason="RUN_NETWORK_TESTS=0")
def test_e2e_sleeves_to_combiner_to_riskofficer():
    """
    Full pipeline:  Sleeves -> Combiner -> RiskOfficer
    on a small real universe. Verify the final portfolio is diversified and
    no constraints are violated.
    """
    from src.portfolio.sleeves import (
        ValueQualityMomentumSleeve, CrossSectionalMomentumSleeve, PEADSleeve
    )
    from src.agents.text_features.macro_regime_agent import MacroRegimeAgent

    small_universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "TSLA",
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
        "JNJ", "PFE", "LLY", "MRK",
        "XOM", "CVX", "COP",
        "WMT", "COST", "PG", "KO", "PEP",
        "HD", "DIS",
    ]

    vqm = ValueQualityMomentumSleeve()
    vqm.universe = lambda: small_universe
    vqm.target_positions = 8

    xs = CrossSectionalMomentumSleeve()
    xs.universe = lambda: small_universe
    xs.target_positions = 8

    pead = PEADSleeve()
    pead.universe = lambda: small_universe
    pead.target_positions = 6

    sleeve_results = {
        "value_quality_momentum": vqm.weights(),
        "xs_momentum": xs.weights(),
        "pead": pead.weights(),
    }

    macro = MacroRegimeAgent().compute()
    combiner = Combiner()
    target = combiner.combine(
        sleeve_results,
        macro_regime_score=macro.score,
        macro_regime_confidence=macro.confidence,
    )

    officer = RiskOfficer()
    review = officer.review(target, current_drawdown=0.046)  # current Alpaca DD magnitude

    print(f"\n=== E2E ===")
    print(f"Macro regime: {macro.score:+.2f} (conf {macro.confidence:.2f})")
    print(f"Sleeve weights: {target.sleeve_weights}")
    print(f"Cash: {target.cash_weight:.1%}  Total invested: {target.total_invested:.1%}")
    print(f"Top 10 positions:")
    for t, w in sorted(review.approved_weights.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {t}: {w*100:.2f}%")
    print(f"Violations: {len(review.violations)}")
    for v in review.violations:
        print(f"  - {v}")

    # Hard assertions: no constraint violated
    assert max(review.approved_weights.values()) <= RiskOfficer.MAX_POSITION_NAV + 1e-6
    assert all(w <= RiskOfficer.MAX_SLEEVE_NAV + 1e-6 for w in target.sleeve_weights.values())
