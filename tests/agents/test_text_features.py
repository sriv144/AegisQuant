"""
Tests for the text-feature agents.

Unit tests verify the TextFeature contract and lexicon/heuristic fallbacks
(no network, no LLM). Smoke tests hit yfinance/SEC for end-to-end coverage.
"""
import os
import pytest
from datetime import datetime

from src.agents.text_features.base import TextFeature, TextFeatureAgent
from src.agents.text_features.news_agent import NewsAgent
from src.agents.text_features.macro_regime_agent import MacroRegimeAgent


NETWORK_OK = os.getenv("RUN_NETWORK_TESTS", "1") == "1"


# ── TextFeature contract ─────────────────────────────────────────────────────


def test_textfeature_empty():
    f = TextFeature.empty("test", "because")
    assert f.feature_name == "test"
    assert f.score == 0.0
    assert f.confidence == 0.0
    assert not f.is_actionable()
    assert "because" in f.rationale


def test_textfeature_is_actionable():
    f = TextFeature("t", score=2.0, confidence=0.5)
    assert f.is_actionable(min_confidence=0.3)
    assert not f.is_actionable(min_confidence=0.7)


# ── News lexicon fallback (no network, no LLM) ───────────────────────────────


def test_news_lexicon_bullish():
    agent = NewsAgent()
    # Force lexicon path by stubbing _get_openai
    agent._get_openai = lambda: None
    headlines = [
        {"title": "AAPL beats earnings, raises guidance", "ts": None},
        {"title": "Apple stock soars on strong iPhone sales", "ts": None},
    ]
    feat = agent._lexicon_aggregate("AAPL", headlines)
    assert feat.score > 0, f"expected bullish, got {feat.score}"
    assert feat.confidence > 0
    assert feat.metadata.get("source") == "lexicon"


def test_news_lexicon_bearish():
    agent = NewsAgent()
    agent._get_openai = lambda: None
    headlines = [
        {"title": "AAPL misses earnings, downgrade follows", "ts": None},
        {"title": "Apple stock plunge on weak demand", "ts": None},
    ]
    feat = agent._lexicon_aggregate("AAPL", headlines)
    assert feat.score < 0
    assert feat.confidence > 0


def test_news_lexicon_neutral():
    agent = NewsAgent()
    agent._get_openai = lambda: None
    headlines = [{"title": "AAPL announces new product line", "ts": None}]
    feat = agent._lexicon_aggregate("AAPL", headlines)
    # No lexicon matches → near-zero conf
    assert feat.score == 0.0
    assert feat.confidence < 0.5


# ── Macro regime numeric path ────────────────────────────────────────────────


def test_macro_regime_components_clamped():
    """Synthetic — bypass network and inject values directly."""
    agent = MacroRegimeAgent()
    # Monkey-patch the fetchers
    agent._fetch_close = lambda s: {"^VIX": 10.0, "^TNX": 4.5, "^IRX": 3.0}.get(s)
    agent._fetch_n_day_return = lambda s, n: 0.10
    feat = agent.compute()
    assert feat.score > 0, f"expected risk-on with VIX=10/curve+/SPY+, got {feat.score}"
    assert feat.confidence == 1.0
    assert -3.0 <= feat.score <= 3.0


def test_macro_regime_riskoff():
    agent = MacroRegimeAgent()
    agent._fetch_close = lambda s: {"^VIX": 40.0, "^TNX": 3.0, "^IRX": 4.0}.get(s)
    agent._fetch_n_day_return = lambda s, n: -0.10
    feat = agent.compute()
    assert feat.score < 0, f"expected risk-off, got {feat.score}"


def test_macro_regime_no_data():
    agent = MacroRegimeAgent()
    agent._fetch_close = lambda s: None
    agent._fetch_n_day_return = lambda s, n: None
    feat = agent.compute()
    assert feat.confidence == 0.0
    assert not feat.is_actionable()


# ── Smoke tests (network) ────────────────────────────────────────────────────


@pytest.mark.skipif(not NETWORK_OK, reason="RUN_NETWORK_TESTS=0")
def test_news_agent_smoke():
    """Hit yfinance news for a real ticker; ensure we return *some* feature."""
    agent = NewsAgent()
    feat = agent.compute("AAPL")
    assert isinstance(feat, TextFeature)
    # AAPL almost always has news; if we got 0 confidence it means yfinance
    # returned empty news — log and skip rather than fail (yfinance flakiness)
    if feat.confidence == 0.0:
        pytest.skip(f"yfinance returned no news for AAPL: {feat.rationale}")
    assert -3.0 <= feat.score <= 3.0
    assert 0.0 <= feat.confidence <= 1.0
    print(f"\nNews AAPL: score={feat.score:+.2f} conf={feat.confidence:.2f} "
          f"src={feat.metadata.get('source')}  {feat.rationale[:80]}")


@pytest.mark.skipif(not NETWORK_OK, reason="RUN_NETWORK_TESTS=0")
def test_macro_regime_smoke():
    agent = MacroRegimeAgent()
    feat = agent.compute()
    assert isinstance(feat, TextFeature)
    print(f"\nMacro regime: score={feat.score:+.2f} conf={feat.confidence:.2f}")
    print(f"  {feat.rationale}")
    print(f"  metadata: {feat.metadata}")
