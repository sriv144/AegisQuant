"""
Broker Abstraction Layer
========================
Defines the interface that ALL brokers must implement.
Supports US (Alpaca) and Indian (Zerodha, Angel One) markets.

Usage:
    from src.execution.broker_base import BaseBroker
    class AlpacaBroker(BaseBroker): ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ProductType(Enum):
    CNC = "CNC"      # Delivery (Cash & Carry)
    MIS = "MIS"      # Intraday (Margin Intraday Settlement)
    NRML = "NRML"    # F&O Normal


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderRequest:
    """Standard order request — broker-agnostic."""
    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.CNC
    limit_price: Optional[float] = None
    tag: str = ""  # Strategy tag for attribution


@dataclass
class OrderResult:
    """Standard order result — broker-agnostic."""
    order_id: str
    ticker: str
    side: OrderSide
    requested_qty: int
    filled_qty: int
    fill_price: float
    status: OrderStatus
    slippage_bps: float = 0.0
    commission: float = 0.0
    timestamp: str = ""
    raw_response: dict = field(default_factory=dict)


class BaseBroker(ABC):
    """
    Abstract broker interface. Every broker adapter must implement these methods.
    The execution layer calls these — it never talks to a broker SDK directly.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish session. Returns True on success."""
        ...

    @abstractmethod
    def get_ltp(self, ticker: str) -> float:
        """Fetch last traded price for a single ticker."""
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Total portfolio value (cash + holdings at market value)."""
        ...

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place a single order. Returns fill details."""
        ...

    @abstractmethod
    def get_positions(self) -> List[dict]:
        """Return current open positions from the broker."""
        ...

    @abstractmethod
    def get_order_history(self, limit: int = 50) -> List[dict]:
        """Return recent order history."""
        ...

    def execute_target_weights(
        self,
        tickers: List[str],
        target_weights: np.ndarray,
        theoretical_prices: Dict[str, float],
        portfolio_value: float,
        trade_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, OrderResult]:
        """
        Delta-based weight executor. Queries current broker positions and only
        trades the difference between target and current holdings.

        Long-only: only BUY (increase) or SELL (decrease/exit), never short-sell.
        Skips trades where |delta| < 0.5% of portfolio value (noise filter).

        Returns: Dict[ticker -> OrderResult]
        """
        assert len(target_weights) == len(tickers), "Weight array dimension mismatch"
        results = {}

        # Build current position map from broker
        current_positions = {}
        try:
            positions = self.get_positions()
            for pos in positions:
                sym = pos.get("ticker", pos.get("symbol", ""))
                qty = int(pos.get("qty", 0))
                if sym and qty != 0:
                    current_positions[sym] = qty
        except Exception:
            pass  # If broker doesn't support get_positions, treat as empty

        noise_threshold = portfolio_value * 0.020  # 2% of portfolio — Buffett anti-churn threshold

        for i, ticker in enumerate(tickers):
            weight = float(target_weights[i])
            price = theoretical_prices.get(ticker, 0.0)
            product_str = (trade_types or {}).get(ticker, "CNC")

            if price <= 0:
                continue

            # Target qty from weight
            target_notional = portfolio_value * max(0.0, weight)  # long-only: floor at 0
            target_qty = int(target_notional / price) if price > 0 else 0

            # Current qty from broker
            current_qty = current_positions.get(ticker, 0)

            # Delta
            delta_qty = target_qty - current_qty

            # Skip if delta is noise (< 0.5% of portfolio)
            delta_value = abs(delta_qty * price)
            if delta_value < noise_threshold:
                continue

            # Determine side — long-only: BUY to increase, SELL to decrease
            if delta_qty > 0:
                side = OrderSide.BUY
                qty = delta_qty
            elif delta_qty < 0:
                side = OrderSide.SELL
                qty = abs(delta_qty)
            else:
                continue

            try:
                product = ProductType[product_str] if product_str in ProductType.__members__ else ProductType.CNC
            except (KeyError, ValueError):
                product = ProductType.CNC

            order = OrderRequest(
                ticker=ticker,
                side=side,
                quantity=qty,
                product=product,
            )
            results[ticker] = self.place_order(order)

        return results

    def calculate_shortfall(
        self,
        tickers: List[str],
        target_weights: np.ndarray,
        theoretical_prices: Dict[str, float],
        results: Dict[str, OrderResult],
    ) -> float:
        """Calculate implementation shortfall in basis points."""
        total_shortfall = 0.0
        for i, ticker in enumerate(tickers):
            w = target_weights[i]
            if abs(w) < 0.001 or ticker not in results:
                continue
            theo = theoretical_prices.get(ticker, 0.0)
            fill = results[ticker].fill_price
            if theo <= 0:
                continue
            slip = (fill - theo) / theo if w > 0 else (theo - fill) / theo
            total_shortfall += (slip * 10000) * abs(w)
        return round(total_shortfall, 2)
