"""
Groww API Executor
===========================
Takes continuous [-1, 1] weighting arrays and translates them into market orders
against Groww API (Indian broker).
Mirrors AlpacaExecutor interface for compatibility.

Supports:
- Paper trading (mock mode, no credentials needed)
- Live trading via unofficial growwapi
- MIS (intraday) and CNC (delivery) product types
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Optional

from src import config  # noqa: F401

try:
    from growwapi import GrowwAPI
    has_groww = True
except ImportError:
    has_groww = False

logger = logging.getLogger(__name__)

# Groww Exchange mapping
NSE_EXCHANGE = "NSE"
BSE_EXCHANGE = "BSE"


class GrowwExecutor:
    def __init__(self, tickers: List[str], paper: bool = True):
        self.tickers = tickers
        self.paper = paper

        self.api_key    = os.getenv("GROWW_API_KEY", "")
        self.secret_key = os.getenv("GROWW_SECRET_KEY", "")

        enable_execution = os.getenv("ENABLE_BROKER_EXECUTION", "False").lower() == "true"

        self.mock_mode = (
            not has_groww
            or not self.api_key
            or not enable_execution
        )

        self.api = None

        if not self.mock_mode:
            self._login()
        else:
            if not has_groww:
                logger.warning("[Groww] growwapi not installed. Run: pip install growwapi")
            elif not enable_execution:
                logger.info("[Groww] ENABLE_BROKER_EXECUTION=False — running in paper mode (no real orders).")
            else:
                logger.warning("[Groww] Credentials incomplete. Running in Mock Executor mode.")

    def _login(self):
        """Authenticate with Groww."""
        try:
            # The get_access_token method returns a string OR potentially a dict depending on internal logic.
            # We handle both just in case.
            token_data = GrowwAPI.get_access_token(api_key=self.api_key, secret=self.secret_key)
            if isinstance(token_data, str):
                access_token = token_data
            elif isinstance(token_data, dict):
                access_token = token_data.get("access_token", self.api_key)
            else:
                access_token = self.api_key
                
            self.api = GrowwAPI(token=access_token)
            profile = self.api.get_user_profile()
            
            if profile and profile.get("vendor_user_id"):
                logger.info(f"[Groww] Logged in successfully for user ID {profile.get('vendor_user_id')}")
                print(f"[Groww] Live session established for user ID {profile.get('vendor_user_id')}")
            else:
                logger.error(f"[Groww] Login failed or profile empty: {profile}")
                self.mock_mode = True

        except Exception as e:
            logger.error(f"[Groww] Login error: {e}")
            print(f"[Groww] Login failed ({e}). Falling back to mock mode.")
            self.mock_mode = True

    def get_ltp(self, ticker: str) -> float:
        """Fetch Last Traded Price from Groww."""
        if self.mock_mode or not self.api:
            return 0.0
        try:
            symbol = ticker.replace(".NS", "").replace(".BO", "")
            exchange = "NSE" if ticker.endswith(".NS") else "BSE"
            payload_symbol = f"{exchange}_{symbol}"
            
            ltp_data = self.api.get_ltp(exchange_trading_symbols=payload_symbol, segment="CASH")
            if ltp_data and payload_symbol in ltp_data:
                return float(ltp_data[payload_symbol])
        except Exception as e:
            logger.error(f"[Groww] LTP fetch failed for {ticker}: {e}")
        return 0.0

    def get_portfolio_value(self) -> float:
        """Fetch live portfolio value (holdings + cash)."""
        if self.mock_mode or not self.api:
            return 250000.0
        try:
            # In Groww, available funds can be checked via get_available_margin_details
            # Total value = available cash + active holdings
            funds = self.api.get_available_margin_details()
            cash = funds.get("clear_cash", 0.0) + funds.get("collateral_available", 0.0)
            
            # Fetch holdings
            holdings_response = self.api.get_holdings_for_user()
            holdings = holdings_response.get("holdings", [])
            holdings_value = 0.0
            
            for h in holdings:
                # Based on typical Groww API response schema
                ltp = float(h.get("ltp", 0.0))
                qty = int(h.get("quantity", 0))
                holdings_value += ltp * qty
                
            return cash + holdings_value
        except Exception as e:
            logger.error(f"[Groww] Holdings/Funds fetch failed: {e}")
        return 250000.0

    def _place_order(self, ticker: str, qty: int, transaction_type: str, product_type: str = "CNC") -> Optional[str]:
        """
        Place a single order via Groww API.

        Args:
            ticker:           e.g. "RELIANCE.NS"
            qty:              number of shares
            transaction_type: "BUY" or "SELL"
            product_type:     "CNC" (delivery) or "MIS" (intraday)

        Returns:
            Order ID string, or None on failure
        """
        if self.mock_mode or not self.api:
            print(f"[Groww Mock] {transaction_type} {qty}x {ticker} ({product_type})")
            return f"MOCK_{ticker}_{int(time.time())}"

        try:
            symbol = ticker.replace(".NS", "").replace(".BO", "")
            exchange = "NSE" if ticker.endswith(".NS") else "BSE"

            response = self.api.place_order(
                validity="DAY",
                exchange=exchange,
                order_type="MARKET",
                product=product_type,
                quantity=qty,
                segment="CASH",
                trading_symbol=symbol,
                transaction_type=transaction_type
            )
            
            if response and "order_id" in response:
                order_id = response.get("order_id")
                logger.info(f"[Groww] Order placed: {transaction_type} {qty}x {symbol} ({product_type}) → ID {order_id}")
                return str(order_id)
            elif response and "id" in response:
                # Alternative reference from response
                order_id = response.get("id")
                logger.info(f"[Groww] Order placed: {transaction_type} {qty}x {symbol} ({product_type}) → ID {order_id}")
                return str(order_id)
            else:
                logger.error(f"[Groww] Order failed: {response}")
                return None

        except Exception as e:
            logger.error(f"[Groww] Order error for {ticker}: {e}")
            return None

    def execute_target_weights(
        self,
        target_weights: np.ndarray,
        theoretical_prices: Dict[str, float],
        trade_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Execute target weights as market orders.

        Args:
            target_weights:      Array of target weights (positive = long, 0 = flat)
            theoretical_prices:  Dict of ticker → price for qty calculation
            trade_types:         Dict of ticker → "MIS" or "CNC" (default CNC if not provided)

        Returns:
            Dict of ticker → fill price
        """
        assert len(target_weights) == len(self.tickers), "Weight array dimension mismatch."
        fills = {}

        if self.mock_mode:
            print(f"[Groww Mock] Executing target weights: {target_weights.round(2)}")
            return theoretical_prices.copy()

        # Get portfolio value for qty calculation
        portfolio_value = self.get_portfolio_value()

        for i, ticker in enumerate(self.tickers):
            weight = float(target_weights[i])
            price  = theoretical_prices.get(ticker, 0.0)
            product = (trade_types or {}).get(ticker, "CNC")

            if weight == 0 or price == 0:
                fills[ticker] = price
                continue

            target_rupees = portfolio_value * abs(weight)
            qty = max(1, int(target_rupees / price))
            txn = "BUY" if weight > 0 else "SELL"

            order_id = self._place_order(ticker, qty, txn, product_type=product)
            # In real mode, fill price would come from order book;
            # use theoretical price as approximation
            fills[ticker] = price
            time.sleep(0.3)  # Throttle to avoid rate limits

        return fills

    def calculate_shortfall(
        self,
        target_weights: np.ndarray,
        theoretical_prices: Dict[str, float],
        fill_prices: Dict[str, float],
    ) -> float:
        """Calculate Implementation Shortfall in basis points."""
        total_shortfall_bps = 0.0

        for i, tick in enumerate(self.tickers):
            if tick not in fill_prices or tick not in theoretical_prices:
                continue

            w = target_weights[i]
            if w == 0:
                continue

            theo   = theoretical_prices[tick]
            actual = fill_prices[tick]

            if theo == 0:
                continue

            slip_pct = (actual - theo) / theo if w > 0 else (theo - actual) / theo
            total_shortfall_bps += (slip_pct * 10000) * abs(w)

        return total_shortfall_bps
