"""
Alpaca Broker Adapter (Paper + Live)
=====================================
Implements BaseBroker for Alpaca's trading API.
Supports both paper trading and live trading via ALPACA_BASE_URL.

Requires: pip install alpaca-trade-api

Setup:
  1. Create account at https://alpaca.markets/
  2. Get API keys from https://app.alpaca.markets/paper/dashboard/overview
  3. Set env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
  4. For paper trading (default): ALPACA_BASE_URL=https://paper-api.alpaca.markets
  5. For live trading: ALPACA_BASE_URL=https://api.alpaca.markets
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
    import alpaca_trade_api as tradeapi
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker adapter for US equities.
    Supports paper and live trading. Commission-free.
    """

    # Alpaca uses different time-in-force values
    _TIF_MAP = {
        ProductType.CNC: "gtc",   # Good Till Cancel (like delivery)
        ProductType.MIS: "day",   # Day order (like intraday)
        ProductType.NRML: "gtc",  # Default to GTC
    }

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv(
            "ALPACA_BASE_URL",
            "https://paper-api.alpaca.markets"  # Paper trading by default
        )

        self.api: Optional[object] = None
        self._connected = False
        self._is_paper = "paper" in self.base_url.lower()

    def connect(self) -> bool:
        if not HAS_ALPACA:
            logger.error("[Alpaca] alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            return False

        if not self.api_key or not self.secret_key:
            logger.error("[Alpaca] Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
            return False

        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2',
            )

            # Verify connection by fetching account
            account = self.api.get_account()
            mode = "PAPER" if self._is_paper else "LIVE"
            logger.info(
                f"[Alpaca] Connected ({mode}) — "
                f"equity=${float(account.equity):,.2f}, "
                f"buying_power=${float(account.buying_power):,.2f}, "
                f"status={account.status}"
            )
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"[Alpaca] Connection failed: {e}")
            return False

    def get_ltp(self, ticker: str) -> float:
        """Fetch last trade price from Alpaca."""
        if not self._connected or not self.api:
            return 0.0
        try:
            symbol = self._clean_symbol(ticker)
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            logger.error(f"[Alpaca] LTP failed for {ticker}: {e}")
            return 0.0

    def get_portfolio_value(self) -> float:
        """Total equity from Alpaca account."""
        if not self._connected or not self.api:
            return 0.0
        try:
            account = self.api.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"[Alpaca] Portfolio value failed: {e}")
            return 0.0

    def place_order(self, order: OrderRequest) -> OrderResult:
        if not self._connected or not self.api:
            return self._rejected(order, "Not connected")

        try:
            symbol = self._clean_symbol(order.ticker)
            side = "buy" if order.side == OrderSide.BUY else "sell"
            order_type = "market" if order.order_type == OrderType.MARKET else "limit"
            tif = self._TIF_MAP.get(order.product, "day")

            params = {
                "symbol": symbol,
                "qty": order.quantity,
                "side": side,
                "type": order_type,
                "time_in_force": tif,
            }
            if order.order_type == OrderType.LIMIT and order.limit_price:
                params["limit_price"] = order.limit_price

            alpaca_order = self.api.submit_order(**params)
            order_id = alpaca_order.id
            logger.info(f"[Alpaca] Order submitted: {side.upper()} {order.quantity}x {symbol} -> {order_id}")

            # Wait for fill (market orders fill almost instantly on Alpaca paper)
            fill_price = 0.0
            filled_qty = 0
            status = OrderStatus.PENDING

            for _ in range(10):  # Poll up to 5 seconds
                time.sleep(0.5)
                updated = self.api.get_order(order_id)
                if updated.status == "filled":
                    fill_price = float(updated.filled_avg_price)
                    filled_qty = int(updated.filled_qty)
                    status = OrderStatus.FILLED
                    break
                elif updated.status == "partially_filled":
                    fill_price = float(updated.filled_avg_price) if updated.filled_avg_price else 0.0
                    filled_qty = int(updated.filled_qty) if updated.filled_qty else 0
                    status = OrderStatus.PARTIAL
                    break
                elif updated.status in ("rejected", "canceled", "expired"):
                    return self._rejected(order, f"Order {updated.status}")

            logger.info(
                f"[Alpaca] {side.upper()} {filled_qty}x {symbol} "
                f"@ ${fill_price:.2f} — {status.value}"
            )

            return OrderResult(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                requested_qty=order.quantity,
                filled_qty=filled_qty,
                fill_price=fill_price,
                status=status,
                slippage_bps=0.0,  # Alpaca reports real fills
                commission=0.0,     # Alpaca is commission-free
                timestamp=datetime.utcnow().isoformat(),
                raw_response={"alpaca_status": updated.status if 'updated' in dir() else "unknown"},
            )

        except Exception as e:
            logger.error(f"[Alpaca] Order failed: {e}")
            return self._rejected(order, str(e))

    def get_positions(self) -> List[dict]:
        """Get all open positions from Alpaca."""
        if not self._connected or not self.api:
            return []
        try:
            positions = self.api.list_positions()
            return [
                {
                    "ticker": p.symbol,
                    "qty": int(p.qty),
                    "avg_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_plpc),
                    "side": p.side,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"[Alpaca] Positions failed: {e}")
            return []

    def get_order_history(self, limit: int = 50) -> List[dict]:
        """Get recent order history from Alpaca."""
        if not self._connected or not self.api:
            return []
        try:
            orders = self.api.list_orders(status="all", limit=limit)
            return [
                {
                    "order_id": o.id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": o.qty,
                    "filled_qty": o.filled_qty,
                    "type": o.type,
                    "status": o.status,
                    "filled_avg_price": o.filled_avg_price,
                    "submitted_at": str(o.submitted_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"[Alpaca] Order history failed: {e}")
            return []

    def get_account(self) -> dict:
        """Get full account details from Alpaca."""
        if not self._connected or not self.api:
            return {}
        try:
            acct = self.api.get_account()
            return {
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "status": acct.status,
                "pattern_day_trader": acct.pattern_day_trader,
                "day_trade_count": acct.daytrade_count,
            }
        except Exception as e:
            logger.error(f"[Alpaca] Account fetch failed: {e}")
            return {}

    @staticmethod
    def _clean_symbol(ticker: str) -> str:
        """Remove any exchange suffixes (shouldn't be needed for US, but safety net)."""
        return ticker.replace(".US", "").replace(".NYSE", "").replace(".NASDAQ", "").strip()

    @staticmethod
    def _rejected(order: OrderRequest, reason: str) -> OrderResult:
        return OrderResult(
            order_id="", ticker=order.ticker, side=order.side,
            requested_qty=order.quantity, filled_qty=0, fill_price=0.0,
            status=OrderStatus.REJECTED,
            timestamp=datetime.utcnow().isoformat(),
            raw_response={"error": reason},
        )
