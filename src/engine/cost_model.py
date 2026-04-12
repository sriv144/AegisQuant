import math
from typing import Dict

# Asset type → bid-ask spread in basis points
SPREAD_BPS = {
    "equity": 2.0,
    "crypto": 5.0,
    "etf": 1.5,
}

# Tickers we treat as crypto (all others default to equity)
CRYPTO_TICKERS = {"BTC", "ETH", "BTC-USD", "ETH-USD", "BTCUSDT", "ETHUSDT"}


class TransactionCostModel:
    """
    Realistic three-component transaction cost model.

    Components:
      1. Commission  — flat $1.00 or 0.5bp of notional, whichever is larger.
      2. Bid-ask spread — spread_bps * notional / 2, halved for TWAP/VWAP.
      3. Market impact — square-root model: coeff * sqrt(notional / (adv * price)) * notional.

    Usage:
        cost_model = TransactionCostModel()
        total_cost, breakdown = cost_model.compute_cost(
            price=150.0, quantity=100, adv=5_000_000, algo="MARKET", ticker="AAPL"
        )
    """

    FLAT_COMMISSION = 1.00        # dollars
    COMMISSION_BPS = 0.5          # basis points (0.5bp = 0.00005)
    SQRT_IMPACT_COEFF = 0.1       # square-root market impact coefficient

    def compute_cost(
        self,
        price: float,
        quantity: float,
        adv: float = 10_000_000.0,
        algo: str = "MARKET",
        ticker: str = "",
    ) -> tuple[float, Dict[str, float]]:
        """
        Returns (total_cost_dollars, breakdown_dict).

        Args:
            price:    Fill price per share/unit.
            quantity: Number of shares/units traded (always positive).
            adv:      Average daily volume in dollars for the asset. Defaults to $10M.
            algo:     Execution algorithm — MARKET, LIMIT, TWAP, or VWAP.
            ticker:   Asset ticker used to determine spread category.
        """
        notional = price * abs(quantity)

        # 1. Commission
        commission = max(self.FLAT_COMMISSION, notional * self.COMMISSION_BPS / 10_000)

        # 2. Bid-ask spread
        asset_type = "crypto" if ticker.upper() in CRYPTO_TICKERS else "equity"
        spread_bps = SPREAD_BPS[asset_type]
        spread_cost = notional * spread_bps / 10_000 / 2
        if algo in ("TWAP", "VWAP"):
            spread_cost *= 0.5  # slicing reduces spread cost

        # 3. Market impact (square-root model)
        if adv > 0 and price > 0:
            participation = notional / adv
            market_impact = self.SQRT_IMPACT_COEFF * math.sqrt(participation) * notional
        else:
            market_impact = 0.0

        total = commission + spread_cost + market_impact
        breakdown = {
            "commission": round(commission, 4),
            "spread_cost": round(spread_cost, 4),
            "market_impact": round(market_impact, 4),
            "total": round(total, 4),
            "notional": round(notional, 4),
        }
        return round(total, 4), breakdown


# Singleton
cost_model = TransactionCostModel()
