"""
Text Feature Agents
===================

LLM-powered agents that consume *unstructured text* (filings, news, FOMC
statements) and emit *numeric features* — never votes, never trade actions.

These features feed the Sleeve layer (which may use them to modulate weights)
or the Risk Officer (which may use macro-regime features to scale exposure).

Architecture rationale: per the research review of the LLM-trading literature
(FinMem, TradingAgents, etc.), LLMs are unreliable stock pickers but excellent
document analyzers. We use them where they win.

All agents follow the contract:
    agent.compute(ticker_or_context) -> TextFeature

Failures (no LLM key, API down, parsing error) return a `TextFeature` with
score=0.0 and confidence=0.0 — never raise. The downstream consumer can then
ignore zero-confidence features cleanly.
"""

from src.agents.text_features.base import TextFeature, TextFeatureAgent
from src.agents.text_features.news_agent import NewsAgent
from src.agents.text_features.filing_10k_agent import Filing10KAgent
from src.agents.text_features.macro_regime_agent import MacroRegimeAgent

__all__ = [
    "TextFeature",
    "TextFeatureAgent",
    "NewsAgent",
    "Filing10KAgent",
    "MacroRegimeAgent",
]
