"""
AegisQuant Factor Engine
========================

Modular factor library inspired by:
  - AQR / Asness for value + momentum + quality + defensive
  - Wes Gray (Alpha Architect) for momentum smoothness
  - Robert Carver (Systematic Trading) for trend EWMAC
  - PEAD literature for post-earnings drift
  - Cohen-Malloy-Pomorski for insider buying

Each factor returns a FactorResult with:
  - scores: dict[ticker, float]      cross-sectional z-score (lower=worse, higher=better)
  - confidence: dict[ticker, float]  0..1 — how reliable is this score for this ticker

Factors are deliberately independent — they share a DataProvider for caching
but do not depend on each other. Sleeve constructors combine them.
"""

from src.factors.base import Factor, FactorResult
from src.factors.data_provider import DataProvider, get_data_provider
from src.factors.universe import sp500_tickers, sp100_tickers

from src.factors.value import ValueFactor
from src.factors.quality import QualityFactor
from src.factors.momentum import MomentumFactor
from src.factors.defensive import DefensiveFactor
from src.factors.trend import TrendFactor
from src.factors.pead import PEADFactor
from src.factors.insider import InsiderFactor


def all_factors():
    """Convenience: instantiate one of each factor (shared DataProvider)."""
    return {
        "value": ValueFactor(),
        "quality": QualityFactor(),
        "momentum": MomentumFactor(),
        "defensive": DefensiveFactor(),
        "trend": TrendFactor(),
        "pead": PEADFactor(),
        "insider": InsiderFactor(),
    }


__all__ = [
    "Factor",
    "FactorResult",
    "DataProvider",
    "get_data_provider",
    "sp500_tickers",
    "sp100_tickers",
    "ValueFactor",
    "QualityFactor",
    "MomentumFactor",
    "DefensiveFactor",
    "TrendFactor",
    "PEADFactor",
    "InsiderFactor",
    "all_factors",
]
