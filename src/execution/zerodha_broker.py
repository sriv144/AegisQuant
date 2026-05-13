"""
Zerodha (Kite Connect) Broker Adapter
======================================
Implements BaseBroker for Zerodha's Kite Connect API.
Requires: pip install kiteconnect

Setup:
  1. Create app at https://developers.kite.trade/
  2. Set env vars: KITE_API_KEY, KITE_API_SECRET, KITE_REQUEST_TOKEN
  3. On first run, the adapter exchanges the request token for an access token.

For paper trading, use PaperBroker instead — Zerodha doesn't offer a paper API.
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
    from kiteconnect import KiteConnect
    HAS_KITE = True
except ImportError:
    HAS_KITE = False


class ZerodhaBroker(BaseBroker):
    """
    Zerodha Kite Connect broker adapter.
    Call connect() first, then use place_order / execute_target_weights.
    """

    # Kite product type mapping
    _PRODUCT_MAP = {
        ProductType.CNC: "CNC",
        ProductType.MIS: "MIS",
        ProductType.NRML: "NRML",
    }

    def __init__(self):
        self.api_key = os.getenv("KITE_API_KEY", "")
        self.api_secret = os.getenv("KITE_API_SECRET", "")
        self.request_token = os.getenv("KITE_REQUEST_TOKEN", "")
        self.access_token = os.getenv("KITE_ACCESS_TOKEN", "")

        self.kite: Optional[object] = None
        self._connected = False

    def connect(self) -> bool:
        if not HAS_KITE:
            logger.error("[Zerodha] kiteconnect not installed. Run: pip install kiteconnect")
            return False

        if not self.api_key:
            logger.error("[Zerodha] KITE_API_KEY not set")
            return False

        try:
            self.kite = KiteConnect(api_key=self.api_key)

            if self.access_token:
                self.kite.set_access_token(self.access_token)
            elif self.request_token and self.api_secret:
                data = self.kite.generate_session(
                    self.request_token, api_secret=self.api_secret
                )
                self.access_token = data["access_token"]
                self.kite.set_access_token(self.access_token)
                logger.info(f"[Zerodha] Session generated. Token: {self.access_token[:8]}...")
            else:
                logger.error("[Zerodha] Need KITE_ACCESS_TOKEN or KITE_REQUEST_TOKEN + KITE_API_SECRET")
                return False

            profile = self.kite.profile()
            logger.info(f"[Zerodha] Connected as {profile.get('user_name', 'Unknown')}")
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"[Zerodha] Connection failed: {e}")
            return False

    def get_ltp(self, ticker: str) -> float:
        if not self._connected or not self.kite:
            return 0.0
        try:
            symbol = self._to_kite_symbol(ticker)
            data = self.kite.ltp(symbol)
            return float(data[symbol]["last_price"])
        except Exception as e:
            logger.error(f"[Zerodha] LTP failed for {ticker}: {e}")
            return 0.0

    def get_portfolio_value(self) -> float:
        if not self._connected or not self.kite:
            return 0.0
        try:
            margins = self.kite.margins("equity")
            available = float(margins.get("available", {}).get("live_balance", 0))
            holdings = self.kite.holdings()
            holdings_value = sum(
                float(h.get("last_price", 0)) * int(h.get("quantity", 0))
                for h in holdings
            )
            return available + holdings_value
        except Exception as e:
            logger.error(f"[Zerodha] Portfolio value failed: {e}")
            return 0.0

    def place_order(self, order: OrderRequest) -> OrderResult:
        if not self._connected or not self.kite:
            return self._rejected(order, "Not connected")

        try:
            symbol = order.ticker.replace(".NS", "").replace(".BO", "")
            kite_product = self._PRODUCT_MAP.get(order.product, "CNC")
            txn = "BUY" if order.side == OrderSide.BUY else "SELL"

            params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",
                "transaction_type": txn,
                "quantity": order.quantity,
                "product": kite_product,
                "order_type": "MARKET" if order.order_type == OrderType.MARKET else "LIMIT",
                "validity": "DAY",
                "tag": order.tag[:20] if order.tag else "",
            }
            if order.order_type == OrderType.LIMIT and order.limit_price:
                params["price"] = order.limit_price

            order_id = self.kite.place_order(variety="regular", **params)
            logger.info(f"[Zerodha] Order placed: {txn} {order.quantity}x {symbol} -> {order_id}")

            # Wait briefly for fill
            time.sleep(0.5)
            order_details = self.kite.order_history(order_id)
            latest = order_details[-1] if order_details else {}

            fill_price = float(latest.get("average_price", 0))
            filled_qty = int(latest.get("filled_quantity", 0))
            status_str = latest.get("status", "").upper()

            if status_str == "COMPLETE":
                status = OrderStatus.FILLED
            elif filled_qty > 0:
                status = OrderStatus.PARTIAL
            elif status_str == "REJECTED":
                status = OrderStatus.REJECTED
            else:
                status = OrderStatus.PENDING

            return OrderResult(
                order_id=str(order_id),
                ticker=order.ticker,
                side=order.side,
                requested_qty=order.quantity,
                filled_qty=filled_qty,
                fill_price=fill_price,
                status=status,
                timestamp=datetime.utcnow().isoformat(),
                raw_response=latest,
            )

        except Exception as e:
            logger.error(f"[Zerodha] Order failed: {e}")
            return self._rejected(order, str(e))

    def get_positions(self) -> List[dict]:
        if not self._connected or not self.kite:
            return []
        try:
            positions = self.kite.positions()
            return positions.get("net", [])
        except Exception as e:
            logger.error(f"[Zerodha] Positions failed: {e}")
            return []

    def get_order_history(self, limit: int = 50) -> List[dict]:
        if not self._connected or not self.kite:
            return []
        try:
            orders = self.kite.orders()
            return orders[-limit:] if orders else []
        except Exception as e:
            logger.error(f"[Zerodha] Order history failed: {e}")
            return []

    @staticmethod
    def _to_kite_symbol(ticker: str) -> str:
        """Convert 'RELIANCE.NS' to 'NSE:RELIANCE'."""
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        return f"NSE:{symbol}"

    @staticmethod
    def _rejected(order: OrderRequest, reason: str) -> OrderResult:
        return OrderResult(
            order_id="", ticker=order.ticker, side=order.side,
            requested_qty=order.quantity, filled_qty=0, fill_price=0.0,
            status=OrderStatus.REJECTED,
            timestamp=datetime.utcnow().isoformat(),
            raw_response={"error": reason},
        )
