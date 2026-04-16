"""
Strategy Registry
=================
Maps strategy names to implementations. All strategies inherit from BaseStrategy.
"""

from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.factor_investing import FactorInvestingStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.gap_fill import GapFillStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.earnings_momentum import EarningsMomentumStrategy
from src.strategies.sector_rotation import SectorRotationStrategy

STRATEGY_REGISTRY = {
    "momentum": MomentumStrategy(),
    "mean_reversion": MeanReversionStrategy(),
    "trend_following": TrendFollowingStrategy(),
    "factor_investing": FactorInvestingStrategy(),
    "pairs_trading": PairsTradingStrategy(),
    "gap_fill": GapFillStrategy(),
    "volatility_breakout": VolatilityBreakoutStrategy(),
    "earnings_momentum": EarningsMomentumStrategy(),
    "sector_rotation": SectorRotationStrategy(),
}

def get_strategy(name: str):
    """Retrieve a strategy by name from registry."""
    return STRATEGY_REGISTRY.get(name)

def list_strategies():
    """List all available strategies."""
    return list(STRATEGY_REGISTRY.keys())
