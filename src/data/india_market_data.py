"""
India Market Data Collector
============================
NSE/BSE data via yfinance with .NS/.BO suffixes.
India VIX via ^INDIAVIX.
"""

import logging
from src.data.market_data import MarketDataCollector

logger = logging.getLogger(__name__)

class IndiaMarketDataCollector(MarketDataCollector):
    """Extends base MarketDataCollector for Indian markets."""

    def __init__(self):
        super().__init__()
        self.exchange_suffix = ".NS"  # Default to NSE

    def get_latest_quote(self, ticker: str) -> float:
        """
        Fetch latest quote for Indian ticker.
        Auto-appends .NS if not already present.
        """
        # Ensure .NS suffix
        if not ticker.endswith((".NS", ".BO")):
            ticker_with_suffix = ticker + self.exchange_suffix
        else:
            ticker_with_suffix = ticker

        return super().get_latest_quote(ticker_with_suffix)

    def get_historical_data(self, ticker: str, start_date: str = None, end_date: str = None, interval: str = "1d"):
        """
        Fetch historical data for Indian ticker.
        Auto-appends .NS if not already present.
        """
        # Ensure .NS suffix
        if not ticker.endswith((".NS", ".BO")):
            ticker_with_suffix = ticker + self.exchange_suffix
        else:
            ticker_with_suffix = ticker

        return super().get_historical_data(ticker_with_suffix, start_date, end_date, interval)

    def get_india_vix(self) -> float:
        """
        Fetch latest India VIX (^INDIAVIX) from yfinance.
        Returns 20.0 on any failure (neutral default).
        """
        try:
            import yfinance as yf
            from datetime import datetime, timedelta

            # Fetch last 5 days to ensure we get data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)

            data = yf.download(
                "^INDIAVIX",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )

            if data.empty:
                logger.warning("[IndiaMarketData] ^INDIAVIX returned empty, defaulting to 20.0")
                return 20.0

            vix_value = float(data["Close"].iloc[-1])
            logger.info(f"[IndiaMarketData] India VIX: {vix_value:.2f}")
            return vix_value

        except Exception as e:
            logger.error(f"[IndiaMarketData] VIX fetch failed ({e}), defaulting to 20.0")
            return 20.0

    def _generate_mock_sentiment(self, ticker: str):
        """Override mock prices for Indian tickers."""
        import random
        import numpy as np
        from datetime import datetime

        # Mock prices for Indian ETFs and NIFTY 50 stocks
        india_mock_prices = {
            "NIFTYBEES.NS": 250.0,
            "BANKBEES.NS": 480.0,
            "GOLDBEES.NS": 55.0,
            "LIQUIDBEES.NS": 1000.0,
            "RELIANCE.NS": 1280.0,
            "TCS.NS": 3800.0,
            "HDFCBANK.NS": 1650.0,
            "ICICIBANK.NS": 1120.0,
            "INFY.NS": 1900.0,
            "HINDUNILVR.NS": 2200.0,
            "BHARTIARTL.NS": 1000.0,
            "ITC.NS": 450.0,
            "KOTAKBANK.NS": 1800.0,
            "LT.NS": 2100.0,
        }

        # Normalize ticker
        if not ticker.endswith((".NS", ".BO")):
            ticker_key = ticker + ".NS"
        else:
            ticker_key = ticker

        base_price = india_mock_prices.get(ticker_key, 500.0)

        # Apply random walk
        price = base_price * (1 + random.uniform(-0.02, 0.02))

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "headline": f"Market moving normally on {ticker}",
            "sentiment_score": float(np.random.uniform(-1, 1))
        }

# Module-level singleton
india_market_data = IndiaMarketDataCollector()
