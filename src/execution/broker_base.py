"""
Broker Abstraction Layer
========================
Defines the interface that ALL brokers (Zerodha, Angel One, Groww, paper sim) must implement.
Plug in any Indian broker by subclassing BaseBroker.

Usage:
    from src.execution.broker_base import BaseBroker
    class ZerodhaBroker(BaseBroker): ...
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
        High-level weight executor. Translates target weights into orders.
        This default implementation works for any broker — override only if needed.

        Returns: Dict[ticker -> OrderResult]
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
