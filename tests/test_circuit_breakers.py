import pytest
import numpy as np
from src.engine.circuit_breakers import DrawdownCircuitBreaker, VolatilityCircuitBreaker, MaxPositionRule

def test_max_position_rule():
    rule = MaxPositionRule(max_weight=0.95)
    
    # Under limit
    weights = np.array([0.5, -0.2, 0.1])
    safe_weights, triggered = rule.enforce(weights, {})
    assert not triggered
    np.testing.assert_array_equal(weights, safe_weights)
    
    # Over limit
    weights_breach = np.array([0.99, -1.0, 0.1])
    safe_weights_breach, triggered_breach = rule.enforce(weights_breach, {})
    assert triggered_breach
    assert safe_weights_breach[0] == 0.95
    assert safe_weights_breach[1] == -0.95
    
def test_drawdown_circuit_breaker():
    rule = DrawdownCircuitBreaker(max_drawdown=0.20)
    target = np.array([0.5, 0.5])
    
    # Safe drawdown
    safe, trig = rule.enforce(target, {"drawdown": 0.10})
    assert not trig
    
    # Breach
    safe_breach, trig_breach = rule.enforce(target, {"drawdown": 0.25})
    assert trig_breach
    assert np.all(safe_breach == 0.0) # Flattens portfolio
    
    # Subsequent calls should remain halted
    safe_halt, trig_halt = rule.enforce(target, {"drawdown": 0.05})
    assert trig_halt
    assert np.all(safe_halt == 0.0)

def test_volatility_circuit_breaker():
    rule = VolatilityCircuitBreaker(vix_threshold=60.0, reduction_factor=0.5)
    target = np.array([1.0, -1.0])
    
    # Safe VIX
    safe, trig = rule.enforce(target, {"vix_raw": 20.0})
    assert not trig
    np.testing.assert_array_equal(safe, target)
    
    # Spiked VIX > 60
    safe_vix, trig_vix = rule.enforce(target, {"vix_raw": 65.0})
    assert trig_vix
    np.testing.assert_array_equal(safe_vix, np.array([0.5, -0.5]))
