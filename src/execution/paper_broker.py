"""
Realistic Paper Trading Broker
===============================
Simulates order execution with:
  - Slippage modeling (volume-dependent + random noise)
  - Partial fill probability (large orders may not fill completely)
  - Market impact (square-root model from cost_model.py)
  - NSE transaction costs (STT, brokerage, GST, stamp duty, exchange fees)
  - Order latency simulation
  - Fill price jitter based on volatility

This replaces the old "mock mode" which just returned theoretical prices.
Now paper P&L actually reflects realistic execution friction.
"""

import time
import math
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.execution.broker_base import (
    BaseBroker, OrderRequest, OrderResult,
    OrderSide, OrderStatus, ProductType,
)

logger = logging.getLogger(__name__)


# ── NSE Transaction Costs (as of 2024-25) ────────────────────────────────────
# All rates are per-side (per trade)
NSE_COSTS = {
    "CNC": {
        "brokerage_pct": 0.0,        # Most discount brokers: zero for delivery
        "stt_buy_pct": 0.001,        # 0.1% on buy (delivery)
        "stt_sell_pct": 0.001,       # 0.1% on sell (delivery)
        "exchange_pct": 0.0000345,   # NSE transaction charge
        "gst_on_brokerage_pct": 0.18,  # 18% GST on brokerage + exchange charges
        "sebi_pct": 0.000001,        # SEBI turnover fee
        "stamp_duty_buy_pct": 0.00015,  # Stamp duty on buy (0.015%)
    },
    "MIS": {
        "brokerage_flat": 20.0,      # ₹20 per executed order (Zerodha-style)
        "stt_sell_pct": 0.00025,     # 0.025% on sell only (intraday)
        "exchange_pct": 0.0000345,
        "gst_on_brokerage_pct": 0.18,
        "sebi_pct": 0.000001,
        "stamp_duty_buy_pct": 0.00003,  # 0.003% stamp duty (intraday)
    },
}


class PaperBroker(BaseBroker):
    """
    Simulates realistic NSE execution for paper trading.
    Tracks positions, P&L, and order history internally.
    """

    def __init__(
        self,
        initial_capital: float = 250_000.0,
        slippage_model: str = "realistic",  # "none", "fixed", "realistic"
        fixed_slippage_bps: float = 5.0,     # Used when slippage_model="fixed"
        fill_rate: float = 0.95,             # Probability of full fill (0.0 to 1.0)
        latency_ms: int = 0,                 # Simulated latency (0 = instant)
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.slippage_model = slippage_model
        self.fixed_slippage_bps = fixed_slippage_bps
        self.fill_rate = fill_rate
        self.latency_ms = latency_ms

        self._positions: Dict[str, dict] = {}  # ticker -> {qty, avg_price, side}
        self._order_history: List[dict] = []
        self._order_counter = 0
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"[PaperBroker] Connected. Capital: ₹{self.initial_capital:,.0f}")
        return True

    def get_ltp(self, ticker: str) -> float:
        """Paper broker doesn't have live prices — caller provides them."""
        return 0.0

    def get_portfolio_value(self) -> float:
        """Cash + mark-to-market of open positions (caller must update prices)."""
        # In paper mode, we track cash; position MTM is handled by the position manager
        return self.cash

    def get_positions(self) -> List[dict]:
        return [
            {"ticker": t, **info}
            for t, info in self._positions.items()
            if info.get("qty", 0) != 0
        ]

    def get_order_history(self, limit: int = 50) -> List[dict]:
        return self._order_history[-limit:]

    def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Simulate order execution with realistic slippage and partial fills.
        """
        self._order_counter += 1
        order_id = f"PAPER_{self._order_counter}_{int(time.time())}"

        # Simulated latency
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        # We need a theoretical price — use the latest from theoretical_prices
        # The caller passes this through the weight executor; for direct place_order
        # calls, we use a sentinel that means "use the price provided by execute_target_weights"
        theo_price = getattr(order, '_theo_price', 0.0)
        if theo_price <= 0:
            # Can't simulate without a price
            return OrderResult(
                order_id=order_id, ticker=order.ticker, side=order.side,
                requested_qty=order.quantity, filled_qty=0, fill_price=0.0,
                status=OrderStatus.REJECTED, timestamp=datetime.utcnow().isoformat(),
            )

        # ── Slippage Calculation ──────────────────────────────────────────
        fill_price = self._compute_fill_price(theo_price, order)

        # ── Partial Fill Simulation ───────────────────────────────────────
        filled_qty = order.quantity
        status = OrderStatus.FILLED

        if random.random() > self.fill_rate:
            # Partial fill: 50-90% of requested quantity
            fill_pct = random.uniform(0.5, 0.9)
            filled_qty = max(1, int(order.quantity * fill_pct))
            status = OrderStatus.PARTIAL
            logger.info(f"[PaperBroker] Partial fill: {filled_qty}/{order.quantity} for {order.ticker}")

        # ── Transaction Costs ─────────────────────────────────────────────
        notional = fill_price * filled_qty
        commission = self._compute_transaction_costs(notional, order.side, order.product)

        # ── Update Internal State ─────────────────────────────────────────
        if order.side == OrderSide.BUY:
            self.cash -= (notional + commission)
        else:
            self.cash += (notional - commission)

        # Track position
        pos = self._positions.get(order.ticker, {"qty": 0, "avg_price": 0.0})
        if order.side == OrderSide.BUY:
            total_cost = pos["avg_price"] * pos["qty"] + fill_price * filled_qty
            pos["qty"] += filled_qty
            pos["avg_price"] = total_cost / pos["qty"] if pos["qty"] > 0 else 0.0
        else:
            pos["qty"] -= filled_qty
            if pos["qty"] <= 0:
                pos = {"qty": 0, "avg_price": 0.0}
        self._positions[order.ticker] = pos

        # Slippage in bps
        slippage_bps = abs(fill_price - theo_price) / theo_price * 10000 if theo_price > 0 else 0.0

        result = OrderResult(
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            requested_qty=order.quantity,
            filled_qty=filled_qty,
            fill_price=round(fill_price, 2),
            status=status,
            slippage_bps=round(slippage_bps, 2),
            commission=round(commission, 2),
            timestamp=datetime.utcnow().isoformat(),
        )

        self._order_history.append({
            "order_id": order_id,
            "ticker": order.ticker,
            "side": order.side.value,
            "qty": filled_qty,
            "price": fill_price,
            "slippage_bps": slippage_bps,
            "commission": commission,
            "status": status.value,
            "timestamp": result.timestamp,
        })

        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        logger.info(
            f"[PaperBroker] {action} {filled_qty}x {order.ticker} "
            f"@ ₹{fill_price:.2f} (theo: ₹{theo_price:.2f}, slip: {slippage_bps:.1f}bps, "
            f"cost: ₹{commission:.2f})"
        )

        return result

    def _compute_fill_price(self, theo_price: float, order: OrderRequest) -> float:
        """
        Compute realistic fill price with slippage.

        Models:
          - "none": fill at theoretical price (old behavior)
          - "fixed": fixed N bps adverse slippage
          - "realistic": volume-dependent spread + random noise + market impact
        """
        if self.slippage_model == "none":
            return theo_price

        if self.slippage_model == "fixed":
            slip = theo_price * (self.fixed_slippage_bps / 10000)
            if order.side == OrderSide.BUY:
                return theo_price + slip
            else:
                return theo_price - slip

        # ── Realistic model ───────────────────────────────────────────────
        # 1. Bid-ask spread: assume 3-8 bps for liquid NSE stocks
        spread_bps = random.uniform(3.0, 8.0)
        half_spread = theo_price * (spread_bps / 10000) / 2

        # 2. Market impact: sqrt model — larger orders have more impact
        #    Assume ADV ≈ ₹5Cr for a mid-cap NSE stock
        adv_rupees = 50_000_000.0
        notional = theo_price * order.quantity
        participation = notional / adv_rupees if adv_rupees > 0 else 0.0
        impact = 0.1 * math.sqrt(max(participation, 0)) * theo_price

        # 3. Random noise: normal distribution, ±2bps std dev
        noise = random.gauss(0, theo_price * 2.0 / 10000)

        # Combine: buys pay more, sells receive less
        if order.side == OrderSide.BUY:
            fill_price = theo_price + half_spread + impact + abs(noise)
        else:
            fill_price = theo_price - half_spread - impact - abs(noise)

        # Ensure fill price is positive and reasonable (within 1% of theo)
        fill_price = max(fill_price, theo_price * 0.99)
        fill_price = min(fill_price, theo_price * 1.01)

        return round(fill_price, 2)

    def _compute_transaction_costs(
        self, notional: float, side: OrderSide, product: ProductType
    ) -> float:
        """
        Compute NSE transaction costs (STT + brokerage + GST + exchange + SEBI + stamp duty).
        """
        product_key = "MIS" if product == ProductType.MIS else "CNC"
        rates = NSE_COSTS[product_key]

        costs = 0.0

        if product_key == "MIS":
            # Flat brokerage
            costs += rates["brokerage_flat"]
            # STT on sell only for intraday
            if side == OrderSide.SELL:
                costs += notional * rates["stt_sell_pct"]
            # Stamp duty on buy only
            if side == OrderSide.BUY:
                costs += notional * rates["stamp_duty_buy_pct"]
        else:
            # CNC: zero brokerage, STT on both sides
            if side == OrderSide.BUY:
                costs += notional * rates["stt_buy_pct"]
                costs += notional * rates["stamp_duty_buy_pct"]
            else:
                costs += notional * rates["stt_sell_pct"]

        # Exchange transaction charges
        exchange_charge = notional * rates["exchange_pct"]
        costs += exchange_charge

        # GST on brokerage + exchange charges
        brokerage_for_gst = rates.get("brokerage_flat", 0.0)
        costs += (brokerage_for_gst + exchange_charge) * rates["gst_on_brokerage_pct"]

        # SEBI turnover fee
        costs += notional * rates["sebi_pct"]

        return round(costs, 2)

    def execute_target_weights(
        self,
        tickers: List[str],
        target_weights: np.ndarray,
        theoretical_prices: Dict[str, float],
        portfolio_value: float,
        trade_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, OrderResult]:
        """
        Override to inject theoretical prices into order objects for slippage calculation.
        """
        assert len(target_weights) == len(tickers), "Weight array dimension mismatch"
        results = {}

        for i, ticker in enumerate(tickers):
            weight = float(target_weights[i])
            price = theoretical_prices.get(ticker, 0.0)
            product_str = (trade_types or {}).get(ticker, "CNC")

            if abs(weight) < 0.001 or price <= 0:
                continue

            target_rupees = portfolio_value * abs(weight)
            qty = max(1, int(target_rupees / price))
            side = OrderSide.BUY if weight > 0 else OrderSide.SELL

            try:
                product = ProductType[product_str]
            except (KeyError, ValueError):
                product = ProductType.CNC

            order = OrderRequest(
                ticker=ticker,
                side=side,
                quantity=qty,
                product=product,
            )
            # Inject theoretical price for slippage calculation
            order._theo_price = price  # type: ignore[attr-defined]
            results[ticker] = self.place_order(order)

        return results
