"""
Realistic Paper Trading Broker
===============================
Simulates order execution with:
  - Slippage modeling (volume-dependent + random noise)
  - Partial fill probability (large orders may not fill completely)
  - Market impact (square-root model from cost_model.py)
  - Market-aware transaction costs (US = commission-free, India = NSE fees)
  - Order latency simulation
  - Fill price jitter based on volatility

Supports both US and Indian markets via the `market` parameter.
"""

import os
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
NSE_COSTS = {
    "CNC": {
        "brokerage_pct": 0.0,
        "stt_buy_pct": 0.001,
        "stt_sell_pct": 0.001,
        "exchange_pct": 0.0000345,
        "gst_on_brokerage_pct": 0.18,
        "sebi_pct": 0.000001,
        "stamp_duty_buy_pct": 0.00015,
    },
    "MIS": {
        "brokerage_flat": 20.0,
        "stt_sell_pct": 0.00025,
        "exchange_pct": 0.0000345,
        "gst_on_brokerage_pct": 0.18,
        "sebi_pct": 0.000001,
        "stamp_duty_buy_pct": 0.00003,
    },
}

# ── US Transaction Costs (Alpaca / most US brokers) ──────────────────────────
# Commission-free for equities. Only SEC & FINRA fees apply (negligible).
US_COSTS = {
    "CNC": {
        "sec_fee_per_dollar_sold": 0.0000278,  # SEC fee ~$27.80 per $1M sold
        "finra_taf_per_share_sold": 0.000166,  # FINRA TAF ~$0.000166/share sold
    },
    "MIS": {
        "sec_fee_per_dollar_sold": 0.0000278,
        "finra_taf_per_share_sold": 0.000166,
    },
}


class PaperBroker(BaseBroker):
    """
    Simulates realistic execution for paper trading.
    Supports US (commission-free) and India (NSE fees) markets.
    Tracks positions, P&L, and order history internally.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_model: str = "realistic",  # "none", "fixed", "realistic"
        fixed_slippage_bps: float = 5.0,     # Used when slippage_model="fixed"
        fill_rate: float = 0.95,             # Probability of full fill (0.0 to 1.0)
        latency_ms: int = 0,                 # Simulated latency (0 = instant)
        market: str = "US",                  # "US" or "IN" — determines cost model
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.slippage_model = slippage_model
        self.fixed_slippage_bps = fixed_slippage_bps
        self.fill_rate = fill_rate
        self.latency_ms = latency_ms
        self.market = market.upper()

        self._positions: Dict[str, dict] = {}  # ticker -> {qty, avg_price, side}
        self._order_history: List[dict] = []
        self._order_counter = 0
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        currency = "$" if self.market == "US" else "₹"
        logger.info(f"[PaperBroker] Connected ({self.market} market). Capital: {currency}{self.initial_capital:,.0f}")
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
        commission = self._compute_transaction_costs(notional, order.side, order.product, qty=filled_qty)

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
        currency = "$" if self.market == "US" else "₹"
        logger.info(
            f"[PaperBroker] {action} {filled_qty}x {order.ticker} "
            f"@ {currency}{fill_price:.2f} (theo: {currency}{theo_price:.2f}, slip: {slippage_bps:.1f}bps, "
            f"cost: {currency}{commission:.2f})"
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
        # 1. Bid-ask spread: US stocks are tighter (1-3 bps) vs India (3-8 bps)
        if self.market == "US":
            spread_bps = random.uniform(1.0, 3.0)  # US large caps very tight
        else:
            spread_bps = random.uniform(3.0, 8.0)   # NSE mid/large caps
        half_spread = theo_price * (spread_bps / 10000) / 2

        # 2. Market impact: sqrt model — larger orders have more impact
        #    US: ADV ~$50M for mid-cap, India: ADV ~₹5Cr for mid-cap
        adv_rupees = 50_000_000.0 if self.market == "US" else 50_000_000.0
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
        self, notional: float, side: OrderSide, product: ProductType,
        qty: int = 0,
    ) -> float:
        """
        Compute transaction costs based on market:
        - US: Commission-free (only SEC + FINRA fees on sells, negligible)
        - India: NSE costs (STT + brokerage + GST + exchange + SEBI + stamp duty)
        """
        if self.market == "US":
            return self._compute_us_costs(notional, side, product, qty)
        else:
            return self._compute_india_costs(notional, side, product)

    def _compute_us_costs(
        self, notional: float, side: OrderSide, product: ProductType,
        qty: int = 0,
    ) -> float:
        """US transaction costs: SEC fee + FINRA TAF (sell-side only, negligible)."""
        product_key = "MIS" if product == ProductType.MIS else "CNC"
        rates = US_COSTS[product_key]
        costs = 0.0

        if side == OrderSide.SELL:
            # SEC fee on sell notional
            costs += notional * rates["sec_fee_per_dollar_sold"]
            # FINRA TAF per share sold
            costs += qty * rates["finra_taf_per_share_sold"]

        return round(costs, 4)

    def _compute_india_costs(
        self, notional: float, side: OrderSide, product: ProductType,
    ) -> float:
        """India/NSE transaction costs (STT + brokerage + GST + exchange + SEBI + stamp duty)."""
        product_key = "MIS" if product == ProductType.MIS else "CNC"
        rates = NSE_COSTS[product_key]
        costs = 0.0

        if product_key == "MIS":
            costs += rates["brokerage_flat"]
            if side == OrderSide.SELL:
                costs += notional * rates["stt_sell_pct"]
            if side == OrderSide.BUY:
                costs += notional * rates["stamp_duty_buy_pct"]
        else:
            if side == OrderSide.BUY:
                costs += notional * rates["stt_buy_pct"]
                costs += notional * rates["stamp_duty_buy_pct"]
            else:
                costs += notional * rates["stt_sell_pct"]

        exchange_charge = notional * rates["exchange_pct"]
        costs += exchange_charge
        brokerage_for_gst = rates.get("brokerage_flat", 0.0)
        costs += (brokerage_for_gst + exchange_charge) * rates["gst_on_brokerage_pct"]
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

            target_notional = portfolio_value * abs(weight)
            qty = max(1, int(target_notional / price))
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
