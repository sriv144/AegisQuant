"""
Circuit Breakers
================
Hard-coded safety overrides protecting the execution engine from 
unbounded RL policy decisions or extreme macro events.
"""
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime

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
    def __init__(self, no_trade_before: str = "09:35", no_trade_after: str = "15:55"):
        self.start_fmt = datetime.strptime(no_trade_before, "%H:%M").time()
        self.end_fmt = datetime.strptime(no_trade_after, "%H:%M").time()
        
    def enforce(self, target_weights: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """Prevents trading during highly illiquid open/close auctions."""
        # Note: 'current_weights' must be passed in state to preserve them
        curr_time = datetime.now().time()
        
        if curr_time < self.start_fmt or curr_time > self.end_fmt:
            current_weights = state.get("current_weights", target_weights)
            return current_weights, True
            
        return target_weights, False


class ExecutionFailsafe:
    def __init__(self):
        self.rules = [
            DrawdownCircuitBreaker(max_drawdown=0.20),
            VolatilityCircuitBreaker(vix_threshold=60.0),
            MaxPositionRule(max_weight=0.95),
            TimeWindowRule()
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
