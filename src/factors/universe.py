"""
Universe loaders — free sources only.

- sp500_tickers(): S&P 500 constituents from Wikipedia. Cached on disk for 7d.
- sp100_tickers(): S&P 100 (faster for development / unit tests).

Survivorship-bias note
----------------------
Wikipedia gives only the CURRENT constituents — anything delisted in the past
is gone. For *live trading* that's fine; for backtests it inflates results.
Phase 5 (backtest discipline) will add a point-in-time constituent list from
a free source like the historical-constituents repos on GitHub if needed.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "factors"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL = 7 * 24 * 3600  # 7 days


def _cached(path: Path, fetch_fn, ttl: int = _CACHE_TTL) -> List[str]:
    """Read a newline-delimited cache file; refresh via fetch_fn if stale."""
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
        except Exception as e:
            logger.warning(f"universe cache read failed for {path}: {e}; refetching")

    tickers = fetch_fn()
    try:
        path.write_text("\n".join(tickers))
    except Exception as e:
        logger.warning(f"failed to write universe cache {path}: {e}")
    return tickers


def _fetch_sp500() -> List[str]:
    import pandas as pd
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
        # Wikipedia uses BRK.B; Yahoo expects BRK-B. We normalize to '-' which yfinance accepts.
        return sorted(set(tickers))
    except Exception as e:
        logger.error(f"sp500 fetch from wikipedia failed: {e}")
        return _SP100_FALLBACK[:]  # at least give SOMETHING


# Fallback static list — used if wikipedia is unreachable
_SP100_FALLBACK = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM",
    "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LIN", "LLY", "LMT", "LOW",
    "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS",
    "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM",
    "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO",
    "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA",
    "WFC", "WMT", "XOM",
]


def sp500_tickers() -> List[str]:
    """Return current S&P 500 constituents (cached 7d). Falls back to S&P 100 list."""
    path = _CACHE_DIR / "sp500.txt"
    return _cached(path, _fetch_sp500)


def sp100_tickers() -> List[str]:
    """Static S&P 100 list — useful for fast unit tests / development."""
    return _SP100_FALLBACK[:]
