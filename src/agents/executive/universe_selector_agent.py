"""
Universe Selector Agent
=======================
Wraps UniverseScreener for LangGraph.
Runs weekly to refresh the trading universe from ~2000+ NSE stocks.
"""

import logging
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent
from src.agents.state import AgentState
from src.data.universe_screener import universe_screener
from src.engine.position_manager import position_manager

logger = logging.getLogger(__name__)


class UniverseSelectorAgent(BaseAgent):
    """
    LangGraph node that selects the weekly trading universe.
    Runs once per week (or manually forced).
    """

    def __init__(self):
        super().__init__(
            name="Universe_Selector",
            role=(
                "Universe screener. You analyze all ~2000+ NSE-listed stocks weekly and select "
                "the top 30-50 tradeable candidates based on liquidity, quality, and opportunity. "
                "You prefer stocks with good momentum, stable volatility (1-4% daily), and balanced "
                "sector diversification. You exclude penny stocks, low-volume names, and those with "
                "recent circuit breaker activity."
            ),
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        """
        Select the weekly universe.

        Args:
            state: AgentState (not heavily used, but part of orchestrator interface)

        Returns:
            {"universe": List[str], "universe_info": dict}
        """
        print(f"[{self.name}] Screening NSE universe for {state.get('current_asset', 'portfolio')}...")

        # Get open positions to exclude from screening
        open_positions = position_manager.get_open_positions()

        # Screen universe
        universe = universe_screener.screen_universe(
            force_refresh=False,
            open_positions=open_positions,
        )

        info = {
            "universe_size": len(universe),
            "timestamp": state.get("timestamp"),
            "open_positions_excluded": len(open_positions),
            "top_5": universe[:5] if universe else [],
        }

        logger.info(
            f"[{self.name}] Selected {len(universe)} tickers. "
            f"Top 5: {info['top_5']}"
        )

        return {
            "universe": universe,
            "universe_info": info,
        }


# Module-level singleton
universe_selector_agent = UniverseSelectorAgent()
