import pytest
from src.engine.cost_model import TransactionCostModel

def test_cost_model():
    model = TransactionCostModel()
    
    # Simple market impact check
    # TWAP avoids some market impact compared to MARKET
    cost_market, slip = model.compute_cost(price=100.0, quantity=50000, adv=1e6, algo="MARKET", ticker="SPY")
    cost_twap, slip_twap = model.compute_cost(price=100.0, quantity=50000, adv=1e6, algo="TWAP", ticker="SPY")
    
    assert cost_market > 0.0
    assert cost_twap > 0.0
    assert cost_market > cost_twap # TWAP is cheaper
    
    # 0 quantity should return flat commission
    c_zero, s_zero = model.compute_cost(price=100.0, quantity=0, adv=1e6)
    assert c_zero == 1.0
    assert isinstance(s_zero, dict)
    assert s_zero["notional"] == 0.0
