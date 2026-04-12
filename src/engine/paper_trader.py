import random
from typing import Dict, Any

from src.engine.cost_model import cost_model


class PaperTrader:
    """
    Simulates order execution with realistic slippage and transaction costs.

    Phase-0 change: fee calculation now delegates to TransactionCostModel
    (commission + bid-ask spread + square-root market impact) instead of a
    flat percentage, so the RL agent cannot exploit unrealistically cheap trading.
    """

    # Assumed ADV in dollars when not provided by caller
    DEFAULT_ADV = 10_000_000.0

    def execute_order(
        self,
        asset: str,
        direction: str,
        quantity: float,
        market_price: float,
        algo: str = "MARKET",
        adv: float | None = None,
    ) -> Dict[str, Any]:
        """
        Executes a simulated trade and returns fill details including cost breakdown.

        Args:
            asset:        Ticker symbol.
            direction:    "LONG"/"BUY" or "SHORT"/"SELL".
            quantity:     Number of units to trade (must be > 0).
            market_price: Current mid-price.
            algo:         MARKET, LIMIT, TWAP, or VWAP. TWAP/VWAP reduce spread cost.
            adv:          Average daily dollar volume. Defaults to DEFAULT_ADV.
        """
        if quantity <= 0:
            return {"status": "FAILED", "reason": "Quantity must be > 0"}

        effective_adv = adv if adv is not None else self.DEFAULT_ADV

        # Compute realistic transaction costs
        total_cost, cost_breakdown = cost_model.compute_cost(
            price=market_price,
            quantity=quantity,
            adv=effective_adv,
            algo=algo,
            ticker=asset,
        )

        # Derive fill price from spread (half-spread applied directionally)
        half_spread_pct = cost_breakdown["spread_cost"] / (market_price * quantity)
        if direction.upper() in ("LONG", "BUY"):
            fill_price = market_price * (1 + half_spread_pct)
        else:
            fill_price = market_price * (1 - half_spread_pct)

        fill_price = round(fill_price, 4)
        notional_value = round(fill_price * quantity, 2)

        return {
            "status": "FILLED",
            "asset": asset,
            "direction": direction,
            "quantity": quantity,
            "fill_price": fill_price,
            "notional_value": notional_value,
            "cost_breakdown": cost_breakdown,
            "total_cost": total_cost,
            "algo": algo,
        }


paper_trader = PaperTrader()
