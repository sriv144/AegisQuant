"""
Invokes the appropriate trading strategy's generate_signal() based on
the active_strategies selected by strategy_selector_agent.
Returns strategy signals in the same format as other research agents.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agents.state import AgentState
from src.strategies.earnings_momentum import EarningsMomentumStrategy
from src.strategies.factor_investing import FactorInvestingStrategy
from src.strategies.gap_fill import GapFillStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.sector_rotation import SectorRotationStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy


class StrategyRunnerAgent:
    """Thin adapter from concrete strategy classes to research signal rows."""

    STRATEGY_MAP = {
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

    def invoke(self, state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
        ticker = state.get("current_asset", "")
        indicators = state.get("technical_indicators", {}) or {}
        portfolio_state = state.get("portfolio_state", {}) or {}
        alt_data = state.get("alternative_data", {}) or {}

        active = list(state.get("active_strategies") or [])
        current = state.get("current_strategy")
        if not active and current:
            active = [current]

        signals = []
        seen = set()
        for name in active:
            key = str(name or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            strategy = self.STRATEGY_MAP.get(key)
            if strategy is None:
                continue
            try:
                raw = strategy.generate_signal(ticker, indicators, portfolio_state, alt_data)
            except Exception as exc:
                raw = {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "rationale": f"Strategy runner error for {key}: {exc}",
                }
            signals.append(self._convert_signal(strategy.name, raw))

        return {"research_signals": signals}

    @staticmethod
    def _convert_signal(strategy_name: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        action = str(signal.get("action", "HOLD"))
        if action == "LONG":
            action = "PROPOSE_LONG"
        elif action == "SHORT":
            action = "PROPOSE_SHORT"
        return {
            "agent_name": f"Strategy_{strategy_name}",
            "action": action,
            "confidence": float(signal.get("confidence", 0.0) or 0.0),
            "rationale": str(signal.get("rationale", "")),
        }


strategy_runner_agent = StrategyRunnerAgent()
