import os
import random
import logging
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

import pandas as pd

from src import config  # noqa: F401

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 900  # 15 minutes
MAX_RETRIES = 2
RETRY_BACKOFF = 1.5  # seconds, doubles each retry


class MarketDataCollector:
    """
    Fetches OHLCV market data.

    Phase-0 changes:
    - Real data via yfinance when ENABLE_MOCK_DATA=False (auto_adjust=True handles splits/dividends).
    - get_historical_data() returns standardised list-of-dicts (same schema as mock data so
      downstream feature_engineering code is unchanged).
    - get_train_val_test_splits() creates chronological train / validation / test DataFrames.
      The test set must only be touched once — at the very end of the project.
    - Mock random-walk fallback is retained for offline / CI use.
    """

    def __init__(self):
        self.mock_mode = os.getenv("ENABLE_MOCK_DATA", "True").lower() == "true"
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key → (expiry_ts, data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_price_data(
        self, ticker: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper used by the simulation loop (main.py)."""
        return self.get_historical_data(
            ticker,
            start_time.strftime("%Y-%m-%d"),
            end_time.strftime("%Y-%m-%d"),
            interval="1d",
        )

    def get_latest_quote(self, ticker: str) -> float:
        """Returns the most recent close price (cached for 15 min)."""
        if self.mock_mode:
            base_prices = {"AAPL": 150.0, "BTC": 60000.0, "ETH": 3000.0, "SPY": 500.0}
            base = base_prices.get(ticker, 100.0)
            return round(base * (1 + random.uniform(-0.01, 0.01)), 2)

        cache_key = f"quote:{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        data = self.get_historical_data(ticker, interval="1d")
        if data:
            price = data[-1]["close"]
            self._set_cached(cache_key, price)
            return price
        return 100.0

    def _get_cached(self, key: str):
        entry = self._cache.get(key)
        if entry and time.time() < entry[0]:
            return entry[1]
        return None

    def _set_cached(self, key: str, data, ttl: float = CACHE_TTL_SECONDS):
        self._cache[key] = (time.time() + ttl, data)

    def get_historical_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV bars for *ticker* between *start_date* and *end_date*.
        Results are cached for 15 minutes to avoid rate-limiting.
        """
        if self.mock_mode:
            return self._generate_mock_price_data(ticker, start_date, end_date)

        cache_key = f"hist:{ticker}:{start_date}:{end_date}:{interval}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        for attempt in range(MAX_RETRIES + 1):
            try:
                import yfinance as yf

                kwargs: Dict[str, Any] = {"auto_adjust": True, "progress": False}
                if start_date:
                    kwargs["start"] = start_date
                if end_date:
                    kwargs["end"] = end_date

                df = yf.download(ticker, interval=interval, **kwargs)

                if df.empty:
                    logger.warning("yfinance returned empty DataFrame for %s — using mock", ticker)
                    return self._generate_mock_price_data(ticker, start_date, end_date)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]

                df.index.name = "timestamp"
                df = df.reset_index()

                records = []
                for _, row in df.iterrows():
                    records.append({
                        "timestamp": str(row["timestamp"]),
                        "ticker": ticker,
                        "open": round(float(row.get("open", 0)), 4),
                        "high": round(float(row.get("high", 0)), 4),
                        "low": round(float(row.get("low", 0)), 4),
                        "close": round(float(row.get("close", 0)), 4),
                        "volume": round(float(row.get("volume", 0)), 2),
                    })

                self._set_cached(cache_key, records)
                return records

            except Exception as exc:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning("yfinance fetch failed for %s (attempt %d/%d, retrying in %.1fs): %s",
                                   ticker, attempt + 1, MAX_RETRIES + 1, wait, exc)
                    time.sleep(wait)
                else:
                    logger.warning("yfinance fetch failed for %s after %d attempts — falling back to mock",
                                   ticker, MAX_RETRIES + 1)
                    return self._generate_mock_price_data(ticker, start_date, end_date)

    def get_macro_data(self, start_date: str = "2015-01-01") -> pd.DataFrame:
        """
        Fetches VIX and Treasury yield curves (^TNX = 10Y, ^IRX = 13W/3M).
        Returns a DataFrame with columns: vix_raw, vix_z, yield_curve_slope.
        """
        import yfinance as yf
        import numpy as np
        
        try:
            df_macro = yf.download(["^VIX", "^TNX", "^IRX"], start=start_date, auto_adjust=True, progress=False)["Close"]
            if isinstance(df_macro.columns, pd.MultiIndex):
               df_macro.columns = [c[0] for c in df_macro.columns]
               
            # Forward fill missing days (holidays, etc)
            df_macro = df_macro.fillna(method='ffill')
            
            macro_feats = pd.DataFrame(index=df_macro.index)
            macro_feats["vix_raw"] = df_macro.get("^VIX", 20.0)
            
            # Normalised VIX (rolling 63-day z-score)
            mean_vix = macro_feats["vix_raw"].rolling(63).mean()
            std_vix = macro_feats["vix_raw"].rolling(63).std().replace(0, np.nan)
            macro_feats["vix_z"] = (macro_feats["vix_raw"] - mean_vix) / std_vix
            
            # Yield curve slope: 10 year minus 3 month
            tnx = df_macro.get("^TNX", 2.0)
            irx = df_macro.get("^IRX", 1.0)
            macro_feats["yield_curve_slope"] = tnx - irx
            
            return macro_feats.bfill().ffill()
            
        except Exception as e:
            logger.error(f"Failed to fetch macro data: {e}. Using dummy values.")
            idx = pd.date_range(start=start_date, end=datetime.today())
            return pd.DataFrame({
                "vix_raw": 20.0,
                "vix_z": 0.0,
                "yield_curve_slope": 1.0
            }, index=idx)

    def get_train_val_test_splits(
        self,
        ticker: str,
        train_start: str = "2015-01-01",
        train_end: str = "2021-12-31",
        val_end: str = "2022-12-31",
        # Test set = val_end → today (locked; only touch at final evaluation)
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns (train_df, val_df, test_df) — all chronological, no shuffling.

        The test DataFrame should only be evaluated once — after all
        hyperparameter tuning is complete. Any earlier use invalidates results.
        """
        import yfinance as yf

        df = yf.download(ticker, start=train_start, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten MultiIndex if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        train_df = df.loc[:train_end].copy()
        val_df = df.loc[train_end:val_end].iloc[1:].copy()   # avoid overlap
        test_df = df.loc[val_end:].iloc[1:].copy()

        logger.info(
            "Splits for %s | train=%d bars | val=%d bars | test=%d bars",
            ticker, len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_mock_price_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict[str, Any]]:
        """Deterministic-ish random-walk mock data (only used when mock mode is on)."""
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
            except ValueError:
                start_dt = datetime.utcnow() - timedelta(days=30)
        else:
            start_dt = datetime.utcnow() - timedelta(days=30)

        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
            except ValueError:
                end_dt = datetime.utcnow()
        else:
            end_dt = datetime.utcnow()

        data = []
        current_dt = start_dt
        base_price = 100.0

        while current_dt <= end_dt:
            change = random.uniform(-0.02, 0.02)
            open_p = base_price * (1 + change)
            high_p = open_p * (1 + random.uniform(0, 0.01))
            low_p = open_p * (1 - random.uniform(0, 0.01))
            close_p = open_p * (1 + random.uniform(-0.01, 0.01))
            volume = random.uniform(1_000_000, 10_000_000)

            data.append({
                "timestamp": current_dt.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "volume": round(volume, 2),
            })
            base_price = close_p
            current_dt += timedelta(days=1)

        return data


# Singleton
market_data = MarketDataCollector()
