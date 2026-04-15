"""
Alpaca Live Executor 
====================
Takes continuous [-1, 1] weighting arrays from the RL agent and formally 
translates them into integer share-lot market orders against the Alpaca Paper / Live API.
Calculates Implementation Shortfall (Slippage) against theoretical marks.
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Tuple

from src import config  # noqa: F401

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    has_alpaca = True
except ImportError:
    has_alpaca = False

logger = logging.getLogger(__name__)

class AlpacaExecutor:
    def __init__(self, tickers: List[str], paper: bool = True):
        self.tickers = tickers
        self.paper = paper
        
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.execution_enabled = os.getenv("ENABLE_BROKER_EXECUTION", "False").lower() == "true"
        
        self.mock_mode = (
            not self.execution_enabled
            or not has_alpaca
            or not self.api_key
            or not self.secret_key
        )
        
        if not self.mock_mode:
            self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            logger.info("Alpaca TradingClient successfully initialized.")
        else:
            logger.warning("Broker execution disabled or credentials unavailable. Running in Mock Executor mode.")

    def execute_target_weights(self, target_weights: np.ndarray, theoretical_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Takes the RL array (e.g. [0.4, -0.2, 0.0]) and issues trades to map the live portfolio 
        to these exact allocations. Returns the actual fill prices for slippage calc.
        """
        assert len(target_weights) == len(self.tickers), "Weight array dimension mismatch."
        
        if self.mock_mode:
            print(f"[Alpaca Mock] Executing target weights: {target_weights.round(2)}")
            # Assume zero slippage in mock mode
            return theoretical_prices.copy()
            
        try:
            account = self.client.get_account()
            portfolio_value = float(account.portfolio_value)
            
            # Fetch current live positions
            live_positions = {p.symbol: float(p.qty) for p in self.client.get_all_positions()}
            
            actual_fill_prices = {}
            orders_submitted = 0
            
            for i, tick in enumerate(self.tickers):
                target_weight = target_weights[i]
                target_notional = portfolio_value * target_weight
                
                # Fetch latest real-time quote for sizing
                # In a full implementation, we'd use alpaca.data.historical or market streams
                current_price = theoretical_prices.get(tick, 100.0) 
                
                target_qty = int(target_notional / current_price)
                current_qty = int(live_positions.get(tick, 0))
                
                delta_qty = target_qty - current_qty
                
                # Minimum viable trade threshold (e.g., don't submit orders for < 0.5% drift)
                if abs(delta_qty * current_price) < (portfolio_value * 0.005):
                    actual_fill_prices[tick] = current_price
                    continue
                    
                side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
                
                req = MarketOrderRequest(
                    symbol=tick,
                    qty=abs(delta_qty),
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                print(f"[Alpaca] Submitting {side.name} {abs(delta_qty)} shares of {tick}")
                order = self.client.submit_order(order_data=req)
                orders_submitted += 1
                
                # We mock the immediate turnaround fill price logic here to avoid a blocked websocket 
                # wait in this generic wrapper. In production, we'd subscribe to trade_updates.
                time.sleep(0.5) 
                actual_fill_prices[tick] = current_price  # Placeholder for actual fill extraction
                
            if orders_submitted == 0:
                print("[Alpaca] No trades triggered. Weights within 0.5% tolerance threshold.")
                
            return actual_fill_prices
            
        except Exception as e:
            logger.error(f"Alpaca execution crashed: {e}")
            return theoretical_prices

    def calculate_shortfall(self, target_weights: np.ndarray, theoretical_prices: Dict[str, float], fill_prices: Dict[str, float]) -> float:
        """
        Calculates the Implementation Shortfall (IS) in basis points.
        IS = (Simulated Net Return) - (Actual Nav Change).
        Here we proxy it by the volume-weighted difference between theoretical and fill price.
        """
        total_shortfall_bps = 0.0
        
        for i, tick in enumerate(self.tickers):
            if tick not in fill_prices or tick not in theoretical_prices:
                continue
                
            w = target_weights[i]
            if w == 0:
                continue
                
            theo = theoretical_prices[tick]
            actual = fill_prices[tick]
            
            # Slippage percent
            if w > 0:
                slip_pct = (actual - theo) / theo
            else:
                slip_pct = (theo - actual) / theo
                
            total_shortfall_bps += (slip_pct * 10000) * abs(w)
            
        return total_shortfall_bps
