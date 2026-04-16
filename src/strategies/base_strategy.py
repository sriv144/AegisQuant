"""
Base Strategy Class
===================
Abstract base for all 9 trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        alt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate trading signal for a ticker.

        Args:
            ticker: Stock/ETF symbol
            indicators: Technical indicators dict from feature_engineering
                       (RSI_14_Z, MACD_Z, Volatility_20_Z, etc.)
            portfolio_state: Current portfolio state
                            (current_drawdown, vix_raw, current_weights, portfolio_value)
            alt_data: Alternative data (sentiment, news_volume, etc.)

        Returns:
            {
                "action": "LONG" | "SHORT" | "HOLD",
                "confidence": float (0.0 to 1.0),
                "rationale": str,
                "strategy": self.name
            }
        """
        raise NotImplementedError
