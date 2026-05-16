"""
Circuit Breakers
================
Hard-coded safety overrides protecting the execution engine from
unbounded RL policy decisions or extreme macro events.
Supports both US (ET/EST/EDT) and India (IST) market timezones.
"""
import os
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timezone, timedelta, date

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _ET = ZoneInfo("America/New_York")   # handles EDT/EST automatically
    _IST = ZoneInfo("Asia/Kolkata")
    def _now_market():
        return datetime.now(_ET if MARKET == "US" else _IST)
except ImportError:
    # Fallback: fixed offsets (no DST — acceptable for paper trading)
    _ET = timezone(timedelta(hours=-5))
    _IST = timezone(timedelta(hours=5, minutes=30))
    def _now_market():
        tz = _ET if MARKET == "US" else _IST
        return datetime.now(tz)

# US market holidays (NYSE observed, 2025-2027)
_US_HOLIDAYS = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
}

# Market configuration
MARKET = os.getenv("MARKET", "US").upper()
IST = timezone(timedelta(hours=5, minutes=30))  # keep for backward compat
ET = timezone(timedelta(hours=-5))              # keep for backward compat
MARKET_TZ = _ET if MARKET == "US" else _IST

class LongOnlyRule:
    """Zeroes any negative weights — enforces long-only + flat (no shorts)."""

    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        modified = False
        safe_weights = target_weights.copy()
        for i, w in enumerate(safe_weights):
            if w < 0:
                safe_weights[i] = 0.0
                modified = True
        return safe_weights, modified


class MaxPositionRule:
    def __init__(self, max_weight: float = 0.10):
        self.max_weight = max_weight

    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """Caps the absolute weight of any single position (default 10%)."""
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
        """Prevents trading outside market hours or on holidays."""
        now = _now_market()
        curr_time = now.time()
        curr_date = now.date()

        # Block on weekends
        if curr_date.weekday() >= 5:  # 5=Sat, 6=Sun
            current_weights = state.get("current_weights", np.zeros_like(target_weights))
            return current_weights, True

        # Block on US market holidays
        if MARKET == "US" and curr_date in _US_HOLIDAYS:
            current_weights = state.get("current_weights", np.zeros_like(target_weights))
            return current_weights, True

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
        curr_time = _now_market().time()

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
            LongOnlyRule(),                           # FIRST: enforce no shorts
            DrawdownCircuitBreaker(max_drawdown=0.20),
            VolatilityCircuitBreaker(vix_threshold=60.0),
            MaxPositionRule(max_weight=0.10),          # Max 10% per ticker
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
