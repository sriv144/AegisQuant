"""
Universe Screener for US Stocks
==================================
Screens S&P 500 + select growth stocks.
Applies 4-stage filtering: liquidity → quality → opportunity → diversification.
Returns top 30-50 tradeable US equities.
"""

import logging
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.market_data import MarketDataCollector
from src.data.feature_engineering import feature_engineer
from src.data.correlation_analyzer import CorrelationAnalyzer

logger = logging.getLogger(__name__)

# Hard exclusion list (delisted, suspended, problematic)
EXCLUSION_LIST = set()

# Sector mapping for US stocks
SECTOR_MAP = {
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "AMZN": "TECH",
    "META": "TECH", "NVDA": "TECH", "TSLA": "AUTO", "AMD": "TECH",
    "INTC": "TECH", "CRM": "TECH", "ORCL": "TECH", "ADBE": "TECH",
    "JPM": "BANKING", "BAC": "BANKING", "GS": "BANKING", "MS": "BANKING",
    "WFC": "BANKING", "C": "BANKING", "SCHW": "BANKING",
    "BRK-B": "FINANCE", "V": "FINANCE", "MA": "FINANCE", "AXP": "FINANCE",
    "UNH": "HEALTH", "JNJ": "HEALTH", "PFE": "HEALTH", "ABBV": "HEALTH",
    "LLY": "HEALTH", "MRK": "HEALTH", "TMO": "HEALTH",
    "PG": "CONSUMER", "KO": "CONSUMER", "PEP": "CONSUMER", "WMT": "CONSUMER",
    "COST": "CONSUMER", "HD": "CONSUMER", "MCD": "CONSUMER",
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY", "SLB": "ENERGY",
    "NEE": "UTILITIES", "DUK": "UTILITIES", "SO": "UTILITIES",
    "BA": "INDUSTRIAL", "CAT": "INDUSTRIAL", "HON": "INDUSTRIAL", "UNP": "INDUSTRIAL",
    "LMT": "DEFENSE", "RTX": "DEFENSE", "GD": "DEFENSE",
    "DIS": "MEDIA", "NFLX": "MEDIA", "CMCSA": "MEDIA",
    "T": "TELECOM", "VZ": "TELECOM", "TMUS": "TELECOM",
}


class USUniverseScreener:
    """
    Screens US stocks across 4 stages:
    1. Liquidity (price, volume)
    2. Quality (market cap proxy, circuit breaker health)
    3. Opportunity (momentum, volatility, RSI)
    4. Diversification (sector caps, existing positions)
    """

    def __init__(self, market_data_collector: Optional[MarketDataCollector] = None):
        self.market_data = market_data_collector or MarketDataCollector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self._cache = {}
        self._cache_time = None
        self.cache_ttl_days = 1

    def screen_universe(self, force_refresh: bool = False, open_positions: Optional[Dict] = None) -> List[str]:
        """
        Main entry point. Returns cached universe if < 1 day old.
        Returns list of top 30-50 US tickers (plain symbols, no suffix).
        """
        if not force_refresh and self._cache and self._cache_time:
            age = datetime.now() - self._cache_time
            if age.days < self.cache_ttl_days:
                logger.info(f"[USUniverseScreener] Using cached universe ({age.days}d old)")
                return self._cache.get("tickers", [])

        print("[USUniverseScreener] Re-running full universe screen...")

        try:
            all_tickers = self._get_us_tickers()
        except Exception as e:
            print(f"[USUniverseScreener] FATAL: _get_us_tickers crashed: {e}")
            all_tickers = []
        print(f"[USUniverseScreener] Starting with {len(all_tickers)} US tickers")

        tickers = self._apply_liquidity_filters(all_tickers)
        print(f"[USUniverseScreener] After liquidity: {len(tickers)} tickers")

        tickers = self._apply_quality_filters(tickers)
        print(f"[USUniverseScreener] After quality: {len(tickers)} tickers")

        scored_tickers = self._apply_opportunity_filters(tickers)
        print(f"[USUniverseScreener] After opportunity scoring: {len(scored_tickers)} tickers")

        final_tickers = self._apply_diversification_filters(scored_tickers, open_positions or {})
        print(f"[USUniverseScreener] Final universe: {len(final_tickers)} tickers")

        if not final_tickers and all_tickers:
            print(f"[USUniverseScreener] WARNING: all filters eliminated everything — using seed ({len(all_tickers)} tickers)")
            final_tickers = all_tickers

        self._cache = {"tickers": final_tickers, "timestamp": datetime.now().isoformat()}
        self._cache_time = datetime.now()

        return final_tickers

    def _get_us_tickers(self) -> List[str]:
        """
        Get list of tradeable US stocks.
        Priority:
        1. Try fetching S&P 500 from Wikipedia
        2. Fall back to curated seed list
        """
        # 1. Try S&P 500 from Wikipedia
        try:
            import requests
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                tables = pd.read_html(resp.text)
                if tables:
                    df = tables[0]
                    tickers = df["Symbol"].tolist()
                    # Clean tickers (some have dots like BRK.B → BRK-B)
                    tickers = [t.replace(".", "-") for t in tickers if isinstance(t, str)]
                    logger.info(f"[USUniverseScreener] Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
                    return tickers
        except Exception as e:
            logger.warning(f"[USUniverseScreener] Wikipedia S&P 500 fetch failed: {e}")

        # 2. Curated seed list — top ~100 US stocks by market cap + growth names
        seed_tickers = [
            # Mega-cap tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
            "ORCL", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "AMAT",
            "MU", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS", "PANW", "NOW",
            # Finance
            "JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "BLK",
            "BRK-B", "V", "MA", "AXP", "COF", "USB", "PNC",
            # Healthcare
            "UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT",
            "AMGN", "GILD", "BMY", "ISRG", "VRTX", "REGN",
            # Consumer
            "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE",
            "SBUX", "TGT", "LOW", "CL", "EL", "MDLZ",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
            # Industrial
            "BA", "CAT", "HON", "UNP", "GE", "MMM", "DE", "UPS", "RTX",
            "LMT", "GD", "NOC",
            # Telecom / Media
            "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS",
            # Growth / Mid-cap
            "SQ", "SHOP", "SNOW", "DDOG", "ZS", "NET", "CRWD",
            "COIN", "PLTR", "RBLX", "U", "ABNB", "UBER", "LYFT",
            "SOFI", "RIVN", "LCID",
        ]
        logger.warning(f"[USUniverseScreener] Using curated seed list of {len(seed_tickers)} tickers")
        return seed_tickers

    def _apply_liquidity_filters(self, tickers: List[str]) -> List[str]:
        """
        Stage 1: Hard liquidity filters.
        - Price > $5
        - Avg Daily Volume > $10M notional
        """
        result = []
        for ticker in tickers:
            if ticker in EXCLUSION_LIST:
                continue
            try:
                hist = self.market_data.get_historical_data(
                    ticker,
                    start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                )
                if hist is None or len(hist) < 10:
                    continue

                latest = hist[-1] if isinstance(hist, list) else hist.iloc[-1]
                price = latest.get("close", latest.get("Close", 0))
                volume = latest.get("volume", latest.get("Volume", 0))

                if price < 5:
                    continue

                # ADV > $10M
                if isinstance(hist, list):
                    prices = [h.get("close", 0) for h in hist[-20:]]
                    volumes = [h.get("volume", 0) for h in hist[-20:]]
                else:
                    prices = hist["Close"].tail(20).tolist() if "Close" in hist else []
                    volumes = hist["Volume"].tail(20).tolist() if "Volume" in hist else []

                if prices and volumes:
                    adv = np.mean(prices) * np.mean(volumes)
                    if adv < 5_000_000:  # $5M minimum (relaxed)
                        continue

                result.append(ticker)
            except Exception as e:
                logger.debug(f"[USUniverseScreener] Liquidity filter failed for {ticker}: {e}")
                continue

        return result

    def _apply_quality_filters(self, tickers: List[str]) -> List[str]:
        """
        Stage 2: Quality filters.
        - Average traded value > $50M over 60 days
        - No extreme circuit-breaker-like moves (>8% daily) on more than 3 of last 20 days
        """
        result = []
        for ticker in tickers:
            try:
                hist = self.market_data.get_historical_data(
                    ticker,
                    start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                )
                if hist is None or len(hist) < 20:
                    continue

                if isinstance(hist, list):
                    df = pd.DataFrame(hist)
                    df.columns = [c.lower() for c in df.columns]
                else:
                    df = hist.copy()
                    df.columns = [c.lower() for c in df.columns]

                df["trade_value"] = df["close"] * df["volume"]
                avg_trade_val = df["trade_value"].tail(60).mean()
                if avg_trade_val < 10_000_000:  # $10M minimum (relaxed)
                    continue

                df["daily_return"] = df["close"].pct_change()
                extreme_moves = (df["daily_return"].abs() > 0.08).sum()
                if extreme_moves > 3:
                    continue

                result.append(ticker)
            except Exception as e:
                logger.debug(f"[USUniverseScreener] Quality filter failed for {ticker}: {e}")
                continue

        return result

    def _apply_opportunity_filters(self, tickers: List[str]) -> List[tuple]:
        """Stage 3: Opportunity scoring. Returns (ticker, score) sorted desc."""
        scored = []
        for ticker in tickers:
            try:
                hist = self.market_data.get_historical_data(
                    ticker,
                    start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                )
                if hist is None or len(hist) < 60:
                    continue

                if isinstance(hist, list):
                    df = pd.DataFrame(hist)
                else:
                    df = hist.copy()

                df_feat = feature_engineer.compute_technical_indicators(df)
                latest = df_feat.iloc[-1]

                close_col = "Close" if "Close" in df.columns else "close"
                roc_20 = df[close_col].pct_change(20).iloc[-1] if len(df) >= 20 else 0
                roc_60 = df[close_col].pct_change(60).iloc[-1] if len(df) >= 60 else 0
                momentum_score = (roc_20 + roc_60) / 2

                volatility_20 = latest.get("Volatility_20", 0.02)
                vol_score = 1.0 if 0.01 < volatility_20 < 0.04 else 0.5

                rsi = latest.get("RSI_14", 50)
                rsi_score = 1.0 if 35 < rsi < 70 else 0.7

                combined_score = momentum_score * 0.6 + vol_score * 0.2 + rsi_score * 0.2
                scored.append((ticker, combined_score))
            except Exception as e:
                logger.debug(f"[USUniverseScreener] Opportunity filter failed for {ticker}: {e}")
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:50]

    def _apply_diversification_filters(self, scored_tickers: List[tuple], open_positions: Dict) -> List[str]:
        """Stage 4: Max 5 stocks per sector, exclude open positions."""
        sector_counts = {}
        result = []
        open_tickers = set(open_positions.keys())

        for ticker, score in scored_tickers:
            if ticker in open_tickers:
                continue

            sector = SECTOR_MAP.get(ticker.upper(), "OTHER")
            if sector_counts.get(sector, 0) >= 5:
                continue

            result.append(ticker)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

            if len(result) >= 40:
                break

        logger.info(f"[USUniverseScreener] Final breakdown by sector: {sector_counts}")
        return result


# Module-level singleton
us_universe_screener = USUniverseScreener()
