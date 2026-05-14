"""
Alpaca Broker Adapter (Paper + Live)
=====================================
Implements BaseBroker using alpaca-py (the modern Alpaca SDK).
No websockets conflict — alpaca-py is dependency-clean.

Requires: pip install alpaca-py  (already in requirements.txt)

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
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaSide,
        TimeInForce,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker adapter for US equities using alpaca-py SDK.
    Supports paper and live trading. Commission-free.
    """

    _TIF_MAP = {
        ProductType.CNC: TimeInForce.GTC if HAS_ALPACA else "gtc",
        ProductType.MIS: TimeInForce.DAY if HAS_ALPACA else "day",
        ProductType.NRML: TimeInForce.GTC if HAS_ALPACA else "gtc",
    }

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv(
            "ALPACA_BASE_URL",
            "https://paper-api.alpaca.markets"
        )
        self._is_paper = "paper" in self.base_url.lower()
        self.client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self._connected = False

    def connect(self) -> bool:
        if not HAS_ALPACA:
            logger.error("[Alpaca] alpaca-py not installed. Run: pip install alpaca-py")
            return False
        if not self.api_key or not self.secret_key:
            logger.error("[Alpaca] Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
            return False
        try:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self._is_paper,
            )
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            account = self.client.get_account()
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
        if not self._connected or not self.data_client:
            return 0.0
        try:
            symbol = self._clean_symbol(ticker)
            req = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trade = self.data_client.get_stock_latest_trade(req)
            return float(trade[symbol].price)
        except Exception as e:
            logger.warning(f"[Alpaca] LTP failed for {ticker}: {e}")
            return 0.0

    def get_portfolio_value(self) -> float:
        if not self._connected or not self.client:
            return 0.0
        try:
            return float(self.client.get_account().equity)
        except Exception as e:
            logger.error(f"[Alpaca] Portfolio value failed: {e}")
            return 0.0

    def place_order(self, order: OrderRequest) -> OrderResult:
        if not self._connected or not self.client:
            return self._rejected(order, "Not connected")
        try:
            symbol = self._clean_symbol(order.ticker)
            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
            tif = self._TIF_MAP.get(order.product, TimeInForce.DAY)

            if order.order_type == OrderType.LIMIT and order.limit_price:
                req = LimitOrderRequest(
                    symbol=symbol, qty=order.quantity, side=side,
                    time_in_force=tif, limit_price=order.limit_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=symbol, qty=order.quantity, side=side,
                    time_in_force=tif,
                )

            alpaca_order = self.client.submit_order(req)
            order_id = str(alpaca_order.id)
            logger.info(f"[Alpaca] Order submitted: {side.value.upper()} {order.quantity}x {symbol} → {order_id}")

            # Poll for fill (market orders fill almost instantly on paper)
            fill_price, filled_qty, status = 0.0, 0, OrderStatus.PENDING
            final_order = alpaca_order

            for _ in range(10):
                time.sleep(0.5)
                final_order = self.client.get_order_by_id(order_id)
                if final_order.status.value == "filled":
                    fill_price = float(final_order.filled_avg_price or 0)
                    filled_qty = int(final_order.filled_qty or 0)
                    status = OrderStatus.FILLED
                    break
                elif final_order.status.value == "partially_filled":
                    fill_price = float(final_order.filled_avg_price or 0)
                    filled_qty = int(final_order.filled_qty or 0)
                    status = OrderStatus.PARTIAL
                    break
                elif final_order.status.value in ("rejected", "canceled", "expired"):
                    return self._rejected(order, f"Order {final_order.status.value}")

            logger.info(
                f"[Alpaca] {side.value.upper()} {filled_qty}x {symbol} "
                f"@ ${fill_price:.2f} — {status.value}"
            )
            return OrderResult(
                order_id=order_id, ticker=order.ticker, side=order.side,
                requested_qty=order.quantity, filled_qty=filled_qty,
                fill_price=fill_price, status=status,
                slippage_bps=0.0, commission=0.0,
                timestamp=datetime.utcnow().isoformat(),
                raw_response={"alpaca_status": final_order.status.value},
            )

        except Exception as e:
            logger.error(f"[Alpaca] Order failed: {e}")
            return self._rejected(order, str(e))

    def get_positions(self) -> List[dict]:
        if not self._connected or not self.client:
            return []
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "ticker": p.symbol,
                    "qty": int(p.qty),
                    "avg_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_plpc),
                    "side": p.side.value,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"[Alpaca] Positions failed: {e}")
            return []

    def get_order_history(self, limit: int = 50) -> List[dict]:
        if not self._connected or not self.client:
            return []
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            orders = self.client.get_orders(req)
            return [
                {
                    "order_id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": str(o.qty),
                    "filled_qty": str(o.filled_qty),
                    "type": o.type.value,
                    "status": o.status.value,
                    "filled_avg_price": str(o.filled_avg_price),
                    "submitted_at": str(o.submitted_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"[Alpaca] Order history failed: {e}")
            return []

    def get_account(self) -> dict:
        if not self._connected or not self.client:
            return {}
        try:
            acct = self.client.get_account()
            return {
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "status": str(acct.status),
                "pattern_day_trader": acct.pattern_day_trader,
                "day_trade_count": acct.daytrade_count,
            }
        except Exception as e:
            logger.error(f"[Alpaca] Account fetch failed: {e}")
            return {}

    @staticmethod
    def _clean_symbol(ticker: str) -> str:
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
