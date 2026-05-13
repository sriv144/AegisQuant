"""
US Market Data Collector
=========================
US equities via yfinance (no suffix needed — tickers are plain symbols like AAPL, MSFT).
CBOE VIX via ^VIX.
"""

import logging
from src.data.market_data import MarketDataCollector

logger = logging.getLogger(__name__)


class USMarketDataCollector(MarketDataCollector):
    """Extends base MarketDataCollector for US markets. No suffix needed."""

    def __init__(self):
        super().__init__()

    def get_latest_quote(self, ticker: str) -> float:
        """Fetch latest quote for US ticker. No suffix needed."""
        return super().get_latest_quote(ticker)

    def get_historical_data(self, ticker: str, start_date: str = None, end_date: str = None, interval: str = "1d"):
        """Fetch historical data for US ticker."""
        return super().get_historical_data(ticker, start_date, end_date, interval)

    def get_vix(self) -> float:
        """
        Fetch latest CBOE VIX (^VIX) from yfinance.
        Returns 20.0 on any failure (neutral default).
        """
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            from threading import Thread
            import queue as queue_module

            result_queue = queue_module.Queue()

            def fetch_vix():
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5)
                    data = yf.download(
                        "^VIX",
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        progress=False
                    )
                    result_queue.put(data)
                except Exception:
                    result_queue.put(None)

            thread = Thread(target=fetch_vix, daemon=True)
            thread.start()
            thread.join(timeout=5)

            try:
                data = result_queue.get_nowait()
            except queue_module.Empty:
                logger.warning("[USMarketData] ^VIX fetch timed out, defaulting to 20.0")
                return 20.0

            if data is None or data.empty:
                logger.warning("[USMarketData] ^VIX returned empty, defaulting to 20.0")
                return 20.0

            vix_value = float(data["Close"].values[-1])
            logger.info(f"[USMarketData] CBOE VIX: {vix_value:.2f}")
            return vix_value

        except Exception as e:
            logger.error(f"[USMarketData] VIX fetch failed ({e}), defaulting to 20.0")
            return 20.0


# Module-level singleton
us_market_data = USMarketDataCollector()
