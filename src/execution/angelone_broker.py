"""
Angel One (SmartAPI) Broker Adapter
====================================
Implements BaseBroker for Angel One's SmartAPI.
Requires: pip install smartapi-python pyotp

Setup:
  1. Create app at https://smartapi.angelbroking.com/
  2. Set env vars: ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
  3. TOTP is auto-generated from the secret — no manual entry needed.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.execution.broker_base import (
    BaseBroker, OrderRequest, OrderResult,
    OrderSide, OrderStatus, OrderType, ProductType,
)

logger = logging.getLogger(__name__)

try:
    from SmartApi import SmartConnect
    HAS_SMART = True
except ImportError:
    HAS_SMART = False

try:
    import pyotp
    HAS_PYOTP = True
except ImportError:
    HAS_PYOTP = False


class AngelOneBroker(BaseBroker):
    """
    Angel One SmartAPI broker adapter.
    Supports TOTP-based login (no manual 2FA).
    """

    _PRODUCT_MAP = {
        ProductType.CNC: "DELIVERY",
        ProductType.MIS: "INTRADAY",
        ProductType.NRML: "CARRYFORWARD",
    }

    _EXCHANGE_MAP = {
        "NSE": "NSE",
        "BSE": "BSE",
        "NFO": "NFO",
    }

    def __init__(self):
        self.api_key = os.getenv("ANGEL_API_KEY", "")
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "")
        self.password = os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

        self.smart: Optional[object] = None
        self._connected = False
        self._auth_token = ""
        self._feed_token = ""

    def connect(self) -> bool:
        if not HAS_SMART:
            logger.error("[AngelOne] smartapi-python not installed. Run: pip install smartapi-python")
            return False

        if not all([self.api_key, self.client_id, self.password]):
            logger.error("[AngelOne] Missing ANGEL_API_KEY, ANGEL_CLIENT_ID, or ANGEL_PASSWORD")
            return False

        try:
            self.smart = SmartConnect(api_key=self.api_key)

            # Generate TOTP
            totp = ""
            if self.totp_secret and HAS_PYOTP:
                totp = pyotp.TOTP(self.totp_secret).now()
            elif self.totp_secret:
                logger.warning("[AngelOne] pyotp not installed. Run: pip install pyotp")

            data = self.smart.generateSession(self.client_id, self.password, totp)

            if data.get("status"):
                self._auth_token = data["data"]["jwtToken"]
                self._feed_token = self.smart.getfeedToken()
                self._connected = True
                logger.info(f"[AngelOne] Connected as {self.client_id}")
                return True
            else:
                logger.error(f"[AngelOne] Session failed: {data.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"[AngelOne] Connection failed: {e}")
            return False

    def get_ltp(self, ticker: str) -> float:
        if not self._connected or not self.smart:
            return 0.0
        try:
            symbol = ticker.replace(".NS", "").replace(".BO", "")
            # Angel One requires symboltoken — this is a simplified lookup
            # In production, you'd maintain a symbol-to-token map
            data = self.smart.ltpData("NSE", symbol, "")
            if data.get("status") and data.get("data"):
                return float(data["data"]["ltp"])
        except Exception as e:
            logger.error(f"[AngelOne] LTP failed for {ticker}: {e}")
        return 0.0

    def get_portfolio_value(self) -> float:
        if not self._connected or not self.smart:
            return 0.0
        try:
            rms = self.smart.rmsLimit()
            if rms.get("status") and rms.get("data"):
                return float(rms["data"].get("net", 0))
        except Exception as e:
            logger.error(f"[AngelOne] Portfolio value failed: {e}")
        return 0.0

    def place_order(self, order: OrderRequest) -> OrderResult:
        if not self._connected or not self.smart:
            return self._rejected(order, "Not connected")

        try:
            symbol = order.ticker.replace(".NS", "").replace(".BO", "")
            product = self._PRODUCT_MAP.get(order.product, "DELIVERY")
            txn = "BUY" if order.side == OrderSide.BUY else "SELL"
            order_type = "MARKET" if order.order_type == OrderType.MARKET else "LIMIT"

            params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": "",  # Must be looked up from instrument master
                "transactiontype": txn,
                "exchange": "NSE",
                "ordertype": order_type,
                "producttype": product,
                "duration": "DAY",
                "quantity": order.quantity,
            }
            if order.order_type == OrderType.LIMIT and order.limit_price:
                params["price"] = order.limit_price

            response = self.smart.placeOrder(params)

            if response:
                order_id = str(response)
                logger.info(f"[AngelOne] Order: {txn} {order.quantity}x {symbol} -> {order_id}")

                # Wait for fill
                time.sleep(0.5)

                return OrderResult(
                    order_id=order_id,
                    ticker=order.ticker,
                    side=order.side,
                    requested_qty=order.quantity,
                    filled_qty=order.quantity,  # Assume full fill for market orders
                    fill_price=0.0,  # Would need order book query
                    status=OrderStatus.FILLED,
                    timestamp=datetime.utcnow().isoformat(),
                    raw_response={"order_id": order_id},
                )
            else:
                return self._rejected(order, "No response from API")

        except Exception as e:
            logger.error(f"[AngelOne] Order failed: {e}")
            return self._rejected(order, str(e))

    def get_positions(self) -> List[dict]:
        if not self._connected or not self.smart:
            return []
        try:
            data = self.smart.position()
            if data.get("status") and data.get("data"):
                return data["data"]
        except Exception as e:
            logger.error(f"[AngelOne] Positions failed: {e}")
        return []

    def get_order_history(self, limit: int = 50) -> List[dict]:
        if not self._connected or not self.smart:
            return []
        try:
            data = self.smart.orderBook()
            if data.get("status") and data.get("data"):
                return data["data"][-limit:]
        except Exception as e:
            logger.error(f"[AngelOne] Order history failed: {e}")
        return []

    @staticmethod
    def _rejected(order: OrderRequest, reason: str) -> OrderResult:
        return OrderResult(
            order_id="", ticker=order.ticker, side=order.side,
            requested_qty=order.quantity, filled_qty=0, fill_price=0.0,
            status=OrderStatus.REJECTED,
            timestamp=datetime.utcnow().isoformat(),
            raw_response={"error": reason},
        )
