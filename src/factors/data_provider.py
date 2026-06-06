"""
Unified data provider for the factor engine.

- get_prices(tickers, start, end) → wide DataFrame (date index, ticker cols, adj close)
- get_fundamentals(ticker) → dict of latest available fundamentals from yfinance
- get_earnings(ticker) → DataFrame of earnings dates + EPS estimate/actual
- get_sector(ticker) → string sector classification

All calls are disk-cached under .cache/factors/ with TTL appropriate for the
data type. yfinance is rate-limited and flaky — caching is mandatory.

Free-only design: yfinance is the sole source. Fundamentals are point-in-time-ish
(yfinance returns trailing 12-month values that update on each earnings report).
For backtests, the caller must pass a historical as_of and we will lag fundamentals
by 45 days (typical 10-Q reporting delay).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "factors"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TTLs by data type — prices change daily, fundamentals quarterly
_TTL_PRICES_DAYS = 1
_TTL_FUNDAMENTALS_DAYS = 7
_TTL_EARNINGS_DAYS = 7
_TTL_SECTOR_DAYS = 30


@dataclass
class DataProvider:
    """Single point of access to all market data for the factor engine."""

    def get_prices(
        self,
        tickers: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for a list of tickers.
        Returns a wide DataFrame (date index, ticker cols).

        Strategy: one cache file per (period or date range). yfinance returns
        a multi-ticker frame natively if we pass a space-separated list.
        """
        import yfinance as yf

        key = self._cache_key("prices", tickers, start or period, end or "now")
        cached = self._read_df_cache(key, ttl_days=_TTL_PRICES_DAYS)
        if cached is not None:
            return cached

        try:
            if start or end:
                df = yf.download(
                    " ".join(tickers),
                    start=start, end=end,
                    progress=False, group_by="ticker", auto_adjust=True,
                    threads=True,
                )
            else:
                df = yf.download(
                    " ".join(tickers),
                    period=period,
                    progress=False, group_by="ticker", auto_adjust=True,
                    threads=True,
                )
        except Exception as e:
            logger.error(f"yfinance bulk download failed: {e}")
            return pd.DataFrame()

        # Normalize to wide adj-close frame
        out = self._to_wide_close(df, tickers)
        self._write_df_cache(key, out)
        return out

    def get_fundamentals(self, ticker: str) -> Dict[str, float]:
        """
        Fetch fundamental ratios from yfinance.Ticker.info.

        Returns a dict with whatever keys yfinance provides; standardised subset:
          - trailingPE, priceToBook, enterpriseToEbitda, freeCashflowYield (computed)
          - returnOnEquity, debtToEquity, profitMargins, operatingMargins
          - beta, sharesOutstanding, marketCap
          - sector, industry

        Returns {} on failure.
        """
        key = self._cache_key("fundamentals", [ticker])
        cached = self._read_json_cache(key, ttl_days=_TTL_FUNDAMENTALS_DAYS)
        if cached is not None:
            return cached

        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            info = dict(tk.info or {})
        except Exception as e:
            logger.warning(f"yfinance info fetch failed for {ticker}: {e}")
            info = {}

        # Derive FCF yield where possible
        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        if fcf and mcap and mcap > 0:
            info["freeCashflowYield"] = float(fcf) / float(mcap)

        # Only persist the fields we actually use — keeps cache small + grep-able
        keep = {
            "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
            "enterpriseToEbitda", "enterpriseToRevenue", "freeCashflowYield",
            "returnOnEquity", "returnOnAssets", "debtToEquity",
            "profitMargins", "operatingMargins", "grossMargins",
            "beta", "sharesOutstanding", "marketCap",
            "sector", "industry", "country",
            "earningsGrowth", "revenueGrowth",
            "currentRatio", "quickRatio",
        }
        clean = {k: info.get(k) for k in keep if info.get(k) is not None}
        self._write_json_cache(key, clean)
        return clean

    def get_earnings(self, ticker: str) -> pd.DataFrame:
        """
        Fetch earnings calendar + EPS surprises.
        Returns DataFrame with columns: date, eps_estimate, eps_actual, surprise_pct.
        Empty frame on failure.
        """
        key = self._cache_key("earnings", [ticker])
        cached = self._read_df_cache(key, ttl_days=_TTL_EARNINGS_DAYS)
        if cached is not None:
            return cached

        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            df = tk.get_earnings_dates(limit=12)
            if df is None or df.empty:
                df = pd.DataFrame(columns=["EPS Estimate", "Reported EPS", "Surprise(%)"])
            df = df.reset_index().rename(columns={
                "Earnings Date": "date",
                "EPS Estimate": "eps_estimate",
                "Reported EPS": "eps_actual",
                "Surprise(%)": "surprise_pct",
            })
        except Exception as e:
            logger.warning(f"yfinance earnings fetch failed for {ticker}: {e}")
            df = pd.DataFrame(columns=["date", "eps_estimate", "eps_actual", "surprise_pct"])

        self._write_df_cache(key, df)
        return df

    def get_sector(self, ticker: str) -> str:
        """Sector classification from yfinance. Returns 'Unknown' on failure."""
        f = self.get_fundamentals(ticker)
        return f.get("sector") or "Unknown"

    # ── cache plumbing ──────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(kind: str, tickers: List[str], *args) -> str:
        joined = "_".join(sorted(tickers)) if len(tickers) <= 5 else f"bulk_{len(tickers)}_{hash(tuple(sorted(tickers))) & 0xFFFFFFFF:x}"
        suffix = "_".join(str(a) for a in args) if args else ""
        return f"{kind}__{joined}__{suffix}".strip("_")

    def _read_df_cache(self, key: str, ttl_days: int) -> Optional[pd.DataFrame]:
        path = _CACHE_DIR / f"{key}.parquet"
        if not path.exists():
            return None
        if (time.time() - path.stat().st_mtime) > ttl_days * 86400:
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"parquet read failed for {path}: {e}")
            return None

    def _write_df_cache(self, key: str, df: pd.DataFrame):
        path = _CACHE_DIR / f"{key}.parquet"
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"parquet write failed for {path}: {e}")

    def _read_json_cache(self, key: str, ttl_days: int) -> Optional[dict]:
        path = _CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        if (time.time() - path.stat().st_mtime) > ttl_days * 86400:
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"json read failed for {path}: {e}")
            return None

    def _write_json_cache(self, key: str, payload: dict):
        path = _CACHE_DIR / f"{key}.json"
        try:
            path.write_text(json.dumps(payload, default=str))
        except Exception as e:
            logger.warning(f"json write failed for {path}: {e}")

    @staticmethod
    def _to_wide_close(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Normalize yfinance multi-ticker download into wide adj-close frame.

        Handles:
          - empty df / missing Close column → returns empty frame
          - single-ticker flat column layout
          - multi-ticker (ticker, field) MultiIndex
          - multi-ticker (field, ticker) MultiIndex
          - partial failures (some tickers missing from response)
        """
        if df is None or df.empty:
            return pd.DataFrame()

        try:
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = df.columns.get_level_values(0)
                lvl1 = df.columns.get_level_values(1) if df.columns.nlevels >= 2 else pd.Index([])
                if "Close" in set(lvl0):
                    # (field, ticker) layout — pick the Close sub-frame
                    close = df["Close"]
                    if isinstance(close, pd.Series):
                        close = close.to_frame()
                    return close.dropna(how="all")
                if "Close" in set(lvl1):
                    # (ticker, field) layout — extract each ticker's Close
                    cols = {}
                    for t in tickers:
                        if t in lvl0:
                            try:
                                cols[t] = df[t]["Close"]
                            except (KeyError, Exception):
                                continue
                    if not cols:
                        return pd.DataFrame()
                    return pd.concat(cols, axis=1).dropna(how="all")
                return pd.DataFrame()

            # Single-ticker flat-column layout
            if "Close" not in df.columns:
                return pd.DataFrame()
            close = df["Close"]
            if isinstance(close, pd.Series):
                return close.to_frame(name=tickers[0] if tickers else "value").dropna(how="all")
            # If multiple columns ended up under "Close", just return as-is
            return close.dropna(how="all")
        except Exception as e:
            logger.warning(f"_to_wide_close failed: {e}")
            return pd.DataFrame()


_singleton: Optional[DataProvider] = None


def get_data_provider() -> DataProvider:
    """Process-level singleton DataProvider."""
    global _singleton
    if _singleton is None:
        _singleton = DataProvider()
    return _singleton
