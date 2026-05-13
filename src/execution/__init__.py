"""
Execution Layer — Broker Factory
=================================
Auto-selects the appropriate broker based on environment configuration.

Priority:
  1. BROKER=alpaca + ALPACA_API_KEY set → AlpacaBroker (US paper/live)
  2. BROKER=zerodha + KITE_API_KEY set → ZerodhaBroker (India)
  3. BROKER=angelone + ANGEL_API_KEY set → AngelOneBroker (India)
  4. Default → PaperBroker (realistic simulation, market-aware costs)

Usage:
    from src.execution import get_broker
    broker = get_broker()
    broker.connect()
    result = broker.place_order(order)
"""

import os
import logging
from src.execution.broker_base import BaseBroker

logger = logging.getLogger(__name__)


def get_broker() -> BaseBroker:
    """
    Factory: returns the configured broker instance.
    Set BROKER env var to 'alpaca', 'zerodha', 'angelone', or 'paper' (default).
    """
    broker_name = os.getenv("BROKER", "paper").lower().strip()
    market = os.getenv("MARKET", "US").upper().strip()

    if broker_name == "alpaca" and os.getenv("ALPACA_API_KEY"):
        from src.execution.alpaca_broker import AlpacaBroker
        logger.info("[BrokerFactory] Using Alpaca (US Markets)")
        return AlpacaBroker()

    elif broker_name == "zerodha" and os.getenv("KITE_API_KEY"):
        from src.execution.zerodha_broker import ZerodhaBroker
        logger.info("[BrokerFactory] Using Zerodha (Kite Connect)")
        return ZerodhaBroker()

    elif broker_name == "angelone" and os.getenv("ANGEL_API_KEY"):
        from src.execution.angelone_broker import AngelOneBroker
        logger.info("[BrokerFactory] Using Angel One (SmartAPI)")
        return AngelOneBroker()

    else:
        from src.execution.paper_broker import PaperBroker
        initial_capital = float(os.getenv("INITIAL_CAPITAL", "100000" if market == "US" else "250000"))
        slippage = os.getenv("PAPER_SLIPPAGE", "realistic")
        currency = "$" if market == "US" else "₹"
        logger.info(f"[BrokerFactory] Using PaperBroker ({market} market, capital={currency}{initial_capital:,.0f}, slippage={slippage})")
        return PaperBroker(
            initial_capital=initial_capital,
            slippage_model=slippage,
            market=market,
        )
