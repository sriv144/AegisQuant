"""
Circuit Breakers
================
Hard-coded safety overrides protecting the execution engine from
unbounded RL policy decisions or extreme macro events.
Supports both US (ET) and India (IST) market timezones.
"""
import os
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
ET = timezone(timedelta(hours=-5))  # EST (no DST adjustment — use pytz for production)

# Market configuration
MARKET = os.getenv("MARKET", "US").upper()
MARKET_TZ = ET if MARKET == "US" else IST

class MaxPositionRule:
    def __init__(self, max_weight: float = 0.95):
        self.max_weight = max_weight
        
    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """Caps the absolute weight of any single position."""
        modified = False
        safe_weights = target_weights.copy()
        
        for i, w in enumerate(safe_weights):
            if abs(w) > self.max_weight:
                safe_weights[i] = np.sign(w) * self.max_weight
                modified = True
                
        return safe_weights, modified


class DrawdownCircuitBreaker:
    def __init__(self, max_drawdown: float = 0.20):
        self.max_drawdown = max_drawdown
        self.halted = False
        
    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """Flattens portfolio and halts if drawdown breach occurs."""
        dd = state.get("drawdown", 0.0)
        
        if self.halted or dd >= self.max_drawdown:
            self.halted = True
            return np.zeros_like(target_weights), True
            
        return target_weights, False


class VolatilityCircuitBreaker:
    def __init__(self, vix_threshold: float = 60.0, reduction_factor: float = 0.50):
        self.vix_threshold = vix_threshold
        self.reduction_factor = reduction_factor
        
    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """Auto-deleverages the entire portfolio if VIX is dangerously high."""
        current_vix = state.get("vix_raw", 20.0)  # Needs to be passed in from state
        
        if current_vix >= self.vix_threshold:
            # Scale down all positions
            return target_weights * self.reduction_factor, True
            
        return target_weights, False


class TimeWindowRule:
    def __init__(self, no_trade_before: str = None, no_trade_after: str = None):
        # US market: 9:30 AM - 3:55 PM ET; India: 9:15 AM - 3:25 PM IST
        if no_trade_before is None:
            no_trade_before = "09:30" if MARKET == "US" else "09:15"
        if no_trade_after is None:
            no_trade_after = "15:55" if MARKET == "US" else "15:25"
        self.start_fmt = datetime.strptime(no_trade_before, "%H:%M").time()
        self.end_fmt = datetime.strptime(no_trade_after, "%H:%M").time()

    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """Prevents trading during highly illiquid open/close auctions."""
        curr_time = datetime.now(MARKET_TZ).time()

        if curr_time < self.start_fmt or curr_time > self.end_fmt:
            current_weights = state.get("current_weights", np.zeros_like(target_weights))
            return current_weights, True

        return target_weights, False


class PositionStopLossRule:
    """Enforces per-position stop-loss limits from PositionManager."""
    def __init__(self):
        self.triggered_tickers = []

    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """
        Check open positions against current prices.
        If any SL hit, zero out that position.

        Requires state to include:
          - position_manager: PositionManager instance
          - current_prices: Dict[ticker, price]
        """
        try:
            from src.engine.position_manager import position_manager

            current_prices = state.get("current_prices", {})
            tickers = state.get("tickers", [])

            if not current_prices or not tickers:
                return target_weights, False

            modified = False
            safe_weights = target_weights.copy()

            # Check each position
            to_exit = position_manager.daily_check(current_prices)
            for ticker in to_exit:
                if ticker in tickers:
                    idx = tickers.index(ticker)
                    safe_weights[idx] = 0.0
                    modified = True
                    self.triggered_tickers.append(ticker)

            return safe_weights, modified
        except Exception as e:
            # Fail gracefully if position manager unavailable
            return target_weights, False


class MISAutoCloseRule:
    """Auto-closes MIS/day-trade positions before market close to avoid auto-square-off."""
    def __init__(self, close_time: str = None):
        # US: close at 3:50 PM ET; India: close at 3:10 PM IST
        if close_time is None:
            close_time = "15:50" if MARKET == "US" else "15:10"
        self.close_time_fmt = datetime.strptime(close_time, "%H:%M").time()

    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """
        If current time >= close_time and any MIS/day-trade position exists, close them.
        """
        curr_time = datetime.now(MARKET_TZ).time()

        if curr_time >= self.close_time_fmt:
            # Check if any trade_types are MIS
            trade_types = state.get("trade_types", {})
            tickers = state.get("tickers", [])

            modified = False
            safe_weights = target_weights.copy()

            for i, ticker in enumerate(tickers):
                if trade_types.get(ticker) == "MIS":
                    safe_weights[i] = 0.0
                    modified = True

            return safe_weights, modified

        return target_weights, False


class ExecutionFailsafe:
    def __init__(self):
        self.rules = [
            DrawdownCircuitBreaker(max_drawdown=0.20),
            VolatilityCircuitBreaker(vix_threshold=60.0),
            MaxPositionRule(max_weight=0.95),
            TimeWindowRule(),
            PositionStopLossRule(),
            MISAutoCloseRule(),
        ]

    def process_action(self, proposed_weights: np.ndarray, current_state: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Passes the RL output through all circuit breakers."""
        safe_weights = proposed_weights.copy()
        modifications = []

        for rule in self.rules:
            safe_weights, triggered = rule.enforce(safe_weights, current_state)
            if triggered:
                modifications.append(rule.__class__.__name__)

        reason = " | ".join(modifications) if modifications else "OK"
        return safe_weights, reason
