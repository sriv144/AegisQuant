from typing import Dict, Any

class PortfolioTracker:
    """
    Maintains the state of the simulated portfolio, tracking cash balance and holdings.
    """
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.holdings = {}  # format: { 'AAPL': {'quantity': 100, 'avg_price': 150.0} }
        self.trade_history = []
        
        # Global risk limits for the Risk Officer
        self.risk_limits = {
            "max_drawdown_pct": 0.15,
            "max_single_asset_exposure": 0.20,
            "max_leverage": 1.0
        }

    def process_fill(self, fill_data: Dict[str, Any]):
        """
        Updates portfolio given a completed trade fill.
        """
        if fill_data.get("status") != "FILLED":
            return
            
        asset = fill_data["asset"]
        direction = fill_data["direction"].upper()
        quantity = fill_data["quantity"]
        fill_price = fill_data["fill_price"]
        fee = fill_data["fee_paid"]
        
        if asset not in self.holdings:
            self.holdings[asset] = {"quantity": 0, "avg_price": 0.0}
            
        holding = self.holdings[asset]
        
        if direction in ["BUY", "LONG"]:
            # Update cash and average price
            total_cost = (quantity * fill_price) + fee
            self.cash_balance -= total_cost
            
            new_qty = holding["quantity"] + quantity
            # VWAP of positions
            holding["avg_price"] = ((holding["quantity"] * holding["avg_price"]) + (quantity * fill_price)) / new_qty
            holding["quantity"] = new_qty
            
        elif direction in ["SELL", "SHORT"]:
            # Simplifying: Assume we only sell what we have (no naked shorting in this v1)
            total_revenue = (quantity * fill_price) - fee
            self.cash_balance += total_revenue
            holding["quantity"] -= quantity
            
            if holding["quantity"] <= 0:
                holding["quantity"] = 0
                holding["avg_price"] = 0.0
                
        self.trade_history.append(fill_data)

    def get_state(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates Mark-to-Market Total Value.
        """
        holdings_value = 0.0
        exposures = {}
        
        for asset, data in self.holdings.items():
            if data["quantity"] > 0:
                mkt_price = current_prices.get(asset, data["avg_price"])
                val = data["quantity"] * mkt_price
                holdings_value += val
                exposures[asset] = val
                
        total_value = self.cash_balance + holdings_value
        
        # Convert absolute exposures to percentages
        for asset in exposures.keys():
            exposures[asset] = exposures[asset] / total_value
            
        drawdown = (self.initial_capital - total_value) / self.initial_capital if total_value < self.initial_capital else 0.0
        
        return {
            "total_value": round(total_value, 2),
            "cash_balance": round(self.cash_balance, 2),
            "holdings_value": round(holdings_value, 2),
            "asset_exposures": exposures,
            "current_drawdown": drawdown,
            "risk_limits": self.risk_limits
        }

portfolio = PortfolioTracker()
