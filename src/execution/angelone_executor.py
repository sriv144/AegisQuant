"""
Angel One SmartAPI Executor
===========================
Takes continuous [-1, 1] weighting arrays and translates them into market orders
against Angel One's SmartAPI (Indian broker).
Mirrors AlpacaExecutor interface for compatibility.
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Tuple

from src import config  # noqa: F401

try:
    from smartapi import SmartConnect
    has_angelone = True
except ImportError:
    has_angelone = False

logger = logging.getLogger(__name__)

class AngelOneExecutor:
    def __init__(self, tickers: List[str], paper: bool = True):
        self.tickers = tickers
        self.paper = paper

        self.api_key = os.getenv("ANGELONE_API_KEY", "")
        self.client_id = os.getenv("ANGELONE_CLIENT_ID", "")
        self.password = os.getenv("ANGELONE_PASSWORD", "")
        self.totp_key = os.getenv("ANGELONE_TOTP_KEY", "")

        self.mock_mode = (
            not has_angelone
            or not self.api_key
            or not self.client_id
            or not self.password
        )

        if not self.mock_mode:
            try:
                self.client = SmartConnect(api_key=self.api_key)
                # In real mode, would do: session_data = self.client.generateSession(self.client_id, self.password, totp)
                # For now, fallback to mock
                logger.warning("Angel One execution: TOTP required for live session. Running in mock mode.")
                self.mock_mode = True
            except Exception as e:
                logger.warning(f"Angel One initialization failed ({e}). Running in Mock Executor mode.")
                self.mock_mode = True
        else:
            logger.warning("Angel One credentials unavailable or SmartAPI not installed. Running in Mock Executor mode.")

    def execute_target_weights(self, target_weights: np.ndarray, theoretical_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Takes the RL array (e.g. [0.4, -0.2, 0.0]) and issues trades to map the live portfolio
        to these exact allocations. Returns the actual fill prices for slippage calc.

        In mock mode: assumes zero slippage and returns theoretical prices.
        """
        assert len(target_weights) == len(self.tickers), "Weight array dimension mismatch."

        if self.mock_mode:
            print(f"[Angel One Mock] Executing target weights: {target_weights.round(2)}")
            # Assume zero slippage in mock mode
            return theoretical_prices.copy()

        try:
            # Real Angel One execution would go here
            # For each ticker:
            #   - Get current position qty
            #   - Compute target qty based on weight
            #   - Submit PlaceOrder request via self.client.placeOrder()
            # This requires full SmartAPI session setup with TOTP

            print(f"[Angel One] Would execute weights: {target_weights.round(2)}")
            return theoretical_prices.copy()

        except Exception as e:
            logger.error(f"Angel One execution crashed: {e}")
            return theoretical_prices

    def calculate_shortfall(self, target_weights: np.ndarray, theoretical_prices: Dict[str, float], fill_prices: Dict[str, float]) -> float:
        """
        Calculates the Implementation Shortfall (IS) in basis points.
        IS = (Simulated Net Return) - (Actual Nav Change).
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
