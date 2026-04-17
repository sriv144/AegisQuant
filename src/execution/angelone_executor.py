"""
Angel One SmartAPI Executor
===========================
Takes continuous [-1, 1] weighting arrays and translates them into market orders
against Angel One's SmartAPI (Indian broker).
Mirrors AlpacaExecutor interface for compatibility.

Supports:
- Paper trading (mock mode, no credentials needed)
- Live trading via SmartAPI with TOTP authentication
- MIS (intraday) and CNC (delivery) product types
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Optional

from src import config  # noqa: F401

try:
    from smartapi import SmartConnect
    has_angelone = True
except ImportError:
    has_angelone = False

logger = logging.getLogger(__name__)

# Angel One SmartAPI token symbol lookup is needed for placeOrder
# Format: exchange_token -> trading_symbol mapping (simplified)
NSE_EXCHANGE = "NSE"
BSE_EXCHANGE = "BSE"


class AngelOneExecutor:
    def __init__(self, tickers: List[str], paper: bool = True):
        self.tickers = tickers
        self.paper = paper

        self.api_key    = os.getenv("ANGELONE_API_KEY", "")
        self.client_id  = os.getenv("ANGELONE_CLIENT_ID", "")
        self.password   = os.getenv("ANGELONE_PASSWORD", "")
        self.totp_key   = os.getenv("ANGELONE_TOTP_KEY", "")  # Base32 secret from Angel One TOTP setup

        enable_execution = os.getenv("ENABLE_BROKER_EXECUTION", "False").lower() == "true"

        self.mock_mode = (
            not has_angelone
            or not self.api_key
            or not self.client_id
            or not self.password
            or not enable_execution
        )

        self.client = None
        self.session_token = None

        if not self.mock_mode:
            self._login()
        else:
            if not has_angelone:
                logger.warning("[AngelOne] smartapi-python not installed. Run: pip install smartapi-python pyotp")
            elif not enable_execution:
                logger.info("[AngelOne] ENABLE_BROKER_EXECUTION=False — running in paper mode (no real orders).")
            else:
                logger.warning("[AngelOne] Credentials incomplete. Running in Mock Executor mode.")

    def _login(self):
        """Authenticate with Angel One SmartAPI using TOTP."""
        try:
            import pyotp

            totp = pyotp.TOTP(self.totp_key).now() if self.totp_key else ""

            self.client = SmartConnect(api_key=self.api_key)
            data = self.client.generateSession(self.client_id, self.password, totp)

            if data and data.get("status"):
                self.session_token = data["data"]["jwtToken"]
                logger.info(f"[AngelOne] Logged in successfully as {self.client_id}")
                print(f"[AngelOne] Live session established for {self.client_id}")
            else:
                logger.error(f"[AngelOne] Login failed: {data}")
                self.mock_mode = True

        except Exception as e:
            logger.error(f"[AngelOne] Login error: {e}")
            print(f"[AngelOne] Login failed ({e}). Falling back to mock mode.")
            self.mock_mode = True

    def get_ltp(self, ticker: str) -> float:
        """Fetch Last Traded Price from Angel One."""
        if self.mock_mode or not self.client:
            return 0.0
        try:
            symbol = ticker.replace(".NS", "").replace(".BO", "")
            ltp_data = self.client.ltpData("NSE", symbol, "")
            if ltp_data and ltp_data.get("data"):
                return float(ltp_data["data"]["ltp"])
        except Exception as e:
            logger.error(f"[AngelOne] LTP fetch failed for {ticker}: {e}")
        return 0.0

    def get_portfolio_value(self) -> float:
        """Fetch live portfolio value (holdings + cash)."""
        if self.mock_mode or not self.client:
            return 250000.0
        try:
            holdings = self.client.holding()
            if holdings and holdings.get("data"):
                total = sum(
                    float(h.get("ltp", 0)) * int(h.get("quantity", 0))
                    for h in holdings["data"]
                )
                return total
        except Exception as e:
            logger.error(f"[AngelOne] Holdings fetch failed: {e}")
        return 250000.0

    def _place_order(self, ticker: str, qty: int, transaction_type: str, product_type: str = "CNC") -> Optional[str]:
        """
        Place a single order via SmartAPI.

        Args:
            ticker:           e.g. "RELIANCE.NS"
            qty:              number of shares
            transaction_type: "BUY" or "SELL"
            product_type:     "CNC" (delivery) or "MIS" (intraday)

        Returns:
            Order ID string, or None on failure
        """
        if self.mock_mode or not self.client:
            print(f"[AngelOne Mock] {transaction_type} {qty}x {ticker} ({product_type})")
            return f"MOCK_{ticker}_{int(time.time())}"

        try:
            symbol = ticker.replace(".NS", "").replace(".BO", "")
            exchange = "NSE" if ticker.endswith(".NS") else "BSE"

            order_params = {
                "variety":          "NORMAL",
                "tradingsymbol":    symbol,
                "symboltoken":      "",       # Requires token lookup from Angel One scrip master
                "transactiontype":  transaction_type,  # "BUY" or "SELL"
                "exchange":         exchange,
                "ordertype":        "MARKET",
                "producttype":      product_type,  # "CNC" or "MIS"
                "duration":         "DAY",
                "price":            "0",
                "squareoff":        "0",
                "stoploss":         "0",
                "quantity":         str(qty),
            }

            response = self.client.placeOrder(order_params)
            if response and response.get("status"):
                order_id = response["data"]["orderid"]
                logger.info(f"[AngelOne] Order placed: {transaction_type} {qty}x {symbol} ({product_type}) → ID {order_id}")
                return order_id
            else:
                logger.error(f"[AngelOne] Order failed: {response}")
                return None

        except Exception as e:
            logger.error(f"[AngelOne] Order error for {ticker}: {e}")
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
            print(f"[AngelOne Mock] Executing target weights: {target_weights.round(2)}")
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
