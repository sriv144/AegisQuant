import pandas as pd
import numpy as np
from typing import List, Dict, Any


class FeatureEngineer:

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_technical_indicators(self, price_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Takes raw OHLCV price data and computes technical indicators.

        Phase-0 changes:
        - Rolling z-score normalization (63-day window) applied to RSI, MACD,
          Volatility, and BB-position to prevent look-ahead bias from global
          normalization.
        - Deprecated fillna(method=...) replaced with .bfill().ffill().
        """
        df = pd.DataFrame(price_data)
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # ---- Moving Averages ----
        df["SMA_10"] = df["close"].rolling(window=10).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

        # ---- MACD ----
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ---- RSI (14-period) ----
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # ---- Bollinger Bands (20-period) ----
        df["BB_Mid"] = df["close"].rolling(window=20).mean()
        df["BB_Std"] = df["close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Mid"] + df["BB_Std"] * 2
        df["BB_Lower"] = df["BB_Mid"] - df["BB_Std"] * 2
        # BB position: where price sits in the band [0 = lower, 1 = upper]
        band_width = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
        df["BB_Position"] = (df["close"] - df["BB_Lower"]) / band_width

        # ---- Volatility & Return ----
        df["Daily_Return"] = df["close"].pct_change()
        df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

        # ---- Rolling Z-score normalization (63-day window) ----
        # Prevents look-ahead bias that global normalization introduces.
        for col in ("RSI_14", "MACD", "Volatility_20", "BB_Position"):
            if col in df.columns:
                df[f"{col}_Z"] = self._rolling_zscore(df[col], window=63)

        # ---- Fill NaNs (pandas 2.x compatible) ----
        df = df.bfill().ffill()

        return df

    def aggregate_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregates recent news sentiment into a score between -1 and 1."""
        if not news_data:
            return {"sentiment_score": 0.0, "news_volume": 0}

        total_score = sum(item.get("sentiment_score", 0.0) for item in news_data)
        count = len(news_data)
        return {
            "sentiment_score": total_score / count,
            "news_volume": count,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int = 63) -> pd.Series:
        """
        Compute a rolling z-score using only past data (no look-ahead).
        Returns NaN for the first `window` rows.
        """
        mean = series.rolling(window).mean()
        std = series.rolling(window).std().replace(0, np.nan)
        return (series - mean) / std


# Singleton
feature_engineer = FeatureEngineer()
