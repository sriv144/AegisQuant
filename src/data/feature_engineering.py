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

        # ---- 12-month momentum (rate of change) ----
        df["mom_12m"] = df["close"].pct_change(periods=min(252, len(df) - 1)) if len(df) > 20 else 0.0

        # ---- ADX (Average Directional Index, 14-period) for trend strength ----
        df["ADX_14"] = self._compute_adx(df, period=14)

        # ---- ATR (Average True Range, 14-period) for volatility sizing ----
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        df["ATR_14"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        # ---- Volume z-score (unusual volume detection) ----
        df["Volume_Z"] = self._rolling_zscore(df["volume"].astype(float), window=63)

        # ---- Rolling Z-score normalization (63-day window) ----
        for col in ("RSI_14", "MACD", "Volatility_20", "BB_Position", "mom_12m"):
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
        mean = series.rolling(window).mean()
        std = series.rolling(window).std().replace(0, np.nan)
        return (series - mean) / std

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1).rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        return dx.rolling(period).mean()


# Singleton
feature_engineer = FeatureEngineer()
