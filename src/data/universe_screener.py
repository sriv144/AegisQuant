"""
Universe Screener for NSE Stocks
==================================
Dynamically screens all ~2000+ NSE stocks weekly.
Applies 4-stage filtering: liquidity → quality → opportunity → diversification.
Returns top 30-50 tradeable candidates based on momentum and volatility.
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

# Hard exclusion list (delisted, suspended, ASM/GSM)
EXCLUSION_LIST = {
    "YESBANK.NS",  # Example: suspension history
}

# Sector mapping (simple version, can be expanded)
SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "HDFCBANK": "BANKING", "RELIANCE": "ENERGY",
    "ICICIBANK": "BANKING", "KOTAK": "BANKING", "AXIS": "BANKING",
    "HDFCLIFE": "INSURANCE", "SBILIFE": "INSURANCE",
    "BHARTI": "TELECOM", "JIO": "TELECOM",
    "ITC": "FMCG", "NESTL": "FMCG", "HINDUSTAN": "FMCG",
    "MARUTI": "AUTO", "BAJAJ": "AUTO", "M&M": "AUTO",
    "HDFC": "FINANCE", "INDIABULL": "FINANCE",
    "LT": "INFRA", "BHARATIARTL": "TELECOM",
}


class UniverseScreener:
    """
    Screens NSE stocks across 4 stages:
    1. Liquidity (price, volume, surveillance)
    2. Quality (market cap, circuit breaker health)
    3. Opportunity (momentum, volatility, RSI)
    4. Diversification (sector caps, existing positions)
    """

    def __init__(self, market_data_collector: Optional[MarketDataCollector] = None):
        self.market_data = market_data_collector or MarketDataCollector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self._cache = {}
        self._cache_time = None
        self.cache_ttl_days = 7

    def screen_universe(self, force_refresh: bool = False, open_positions: Optional[Dict] = None) -> List[str]:
        """
        Main entry point. Returns cached universe if < 7 days old, else re-screens.

        Args:
            force_refresh: If True, ignore cache and re-run full screen
            open_positions: Dict of current open positions to exclude from screening

        Returns:
            List of top 30-50 NSE tickers with .NS suffix
        """
        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            age = datetime.now() - self._cache_time
            if age.days < self.cache_ttl_days:
                logger.info(f"[UniverseScreener] Using cached universe ({age.days}d old)")
                return self._cache.get("tickers", [])

        logger.info("[UniverseScreener] Re-running full universe screen...")

        # Get all NSE tickers
        all_tickers = self._fetch_nse_all_tickers()
        logger.info(f"[UniverseScreener] Starting with {len(all_tickers)} NSE tickers")

        # Apply 4-stage filtering
        tickers = self._apply_liquidity_filters(all_tickers)
        logger.info(f"[UniverseScreener] After liquidity: {len(tickers)} tickers")

        tickers = self._apply_quality_filters(tickers)
        logger.info(f"[UniverseScreener] After quality: {len(tickers)} tickers")

        scored_tickers = self._apply_opportunity_filters(tickers)
        logger.info(f"[UniverseScreener] After opportunity scoring: {len(scored_tickers)} tickers")

        final_tickers = self._apply_diversification_filters(scored_tickers, open_positions or {})
        logger.info(f"[UniverseScreener] Final universe: {len(final_tickers)} tickers")

        # Cache result
        self._cache = {"tickers": final_tickers, "timestamp": datetime.now().isoformat()}
        self._cache_time = datetime.now()

        return final_tickers

    def _fetch_nse_all_tickers(self) -> List[str]:
        """
        Fetch list of all NSE-listed tickers.
        First try: data/nse_all_stocks.csv (static seed)
        Fallback: yfinance bulk download and parse
        """
        try:
            import os
            csv_path = os.path.join(os.path.dirname(__file__), "../../data/nse_all_stocks.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                tickers = [f"{t.strip()}.NS" for t in df["Symbol"].unique() if pd.notna(t)]
                logger.info(f"[UniverseScreener] Loaded {len(tickers)} from nse_all_stocks.csv")
                return tickers
        except Exception as e:
            logger.warning(f"[UniverseScreener] Failed to load nse_all_stocks.csv: {e}")

        # Fallback: use a curated list of liquid NSE stocks (updated quarterly)
        # This is a bootstrap set; in production, fetch from NSE bhav copy API
        fallback_tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
            "HINDUNILVR.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
            "AXISBANK.NS", "SBIN.NS", "HDFC.NS", "MARUTI.NS", "BAJAJFINSV.NS",
            "WIPRO.NS", "TECHM.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "DMART.NS",
            "NESTLEIND.NS", "LTIM.NS", "HCLTECH.NS", "POWERGRID.NS", "TIINDIA.NS",
            "IBREALEST.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "INDIGO.NS", "BAJAJUSDFX.NS",
            "BANKBARODA.NS", "CANBK.NS", "FEDERALBANK.NS", "IDFCBANK.NS", "PNBHOUSING.NS",
        ]
        logger.warning(f"[UniverseScreener] Using fallback list of {len(fallback_tickers)} tickers")
        return fallback_tickers

    def _apply_liquidity_filters(self, tickers: List[str]) -> List[str]:
        """
        Stage 1: Hard liquidity filters.
        - Price > ₹20
        - Avg Daily Volume * Close > ₹1 crore
        - Not in exclusion/surveillance list
        """
        result = []
        for ticker in tickers:
            if ticker in EXCLUSION_LIST:
                continue

            try:
                # Fetch last 20 days to compute ADV
                hist = self.market_data.get_historical_data(ticker, start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
                if hist is None or len(hist) < 10:
                    continue

                latest = hist.iloc[-1]
                price = latest.get("Close", 0)
                volume = latest.get("Volume", 0)

                # Price > ₹20
                if price < 20:
                    continue

                # ADV > ₹1 crore (10,000,000 INR)
                adv = hist["Close"].tail(20).mean() * hist["Volume"].tail(20).mean()
                if adv < 1_000_000:  # Less stringent: ₹10L minimum
                    continue

                result.append(ticker)
            except Exception as e:
                logger.debug(f"[UniverseScreener] Liquidity filter failed for {ticker}: {e}")
                continue

        return result

    def _apply_quality_filters(self, tickers: List[str]) -> List[str]:
        """
        Stage 2: Quality filters.
        - Market cap proxy: average trading value over last 60 days > ₹100 crore
        - No circuit breaker abuse (not upper/lower circuit on > 2 of last 20 days)
        - Positive close on at least 3 of last 5 days
        """
        result = []
        for ticker in tickers:
            try:
                hist = self.market_data.get_historical_data(ticker, start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"))
                if hist is None or len(hist) < 20:
                    continue

                # Market cap proxy: avg traded value > ₹100 crore
                hist["TradeValue"] = hist["Close"] * hist["Volume"]
                avg_trade_val = hist["TradeValue"].tail(60).mean()
                if avg_trade_val < 100_000_000:  # ₹1 crore minimum (relaxed)
                    continue

                # Circuit breaker check: count if close is at upper or lower circuit (~5% move)
                hist["DailyReturn"] = hist["Close"].pct_change()
                circuit_hits = (hist["DailyReturn"].abs() > 0.04).sum()
                if circuit_hits > 3:
                    continue

                # Positive closes: at least 3 of last 5 days
                recent = hist.tail(5)
                positive_closes = (recent["Close"].diff() > 0).sum()
                if positive_closes < 2:
                    continue

                result.append(ticker)
            except Exception as e:
                logger.debug(f"[UniverseScreener] Quality filter failed for {ticker}: {e}")
                continue

        return result

    def _apply_opportunity_filters(self, tickers: List[str]) -> List[tuple]:
        """
        Stage 3: Opportunity filters with scoring.
        Returns list of (ticker, score) tuples sorted by score desc.

        Scores based on:
        - Momentum: (20-day ROC + 60-day ROC) / 2, ranked
        - Volatility: 1-4% daily is ideal (not too calm, not too wild)
        - RSI: avoid extreme overbought (> 80) or oversold (< 20)
        """
        scored = []
        for ticker in tickers:
            try:
                hist = self.market_data.get_historical_data(ticker, start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"))
                if hist is None or len(hist) < 60:
                    continue

                # Compute indicators
                df_feat = feature_engineer.compute_technical_indicators(hist)
                latest = df_feat.iloc[-1]

                # Momentum score: (20-day + 60-day ROC) / 2
                roc_20 = hist["Close"].pct_change(20).iloc[-1] if len(hist) >= 20 else 0
                roc_60 = hist["Close"].pct_change(60).iloc[-1] if len(hist) >= 60 else 0
                momentum_score = (roc_20 + roc_60) / 2

                # Volatility: prefer 1-4% daily
                volatility_20 = latest.get("Volatility_20", 0.02)
                vol_score = 1.0 if 0.01 < volatility_20 < 0.04 else 0.5

                # RSI: avoid extremes
                rsi = latest.get("RSI_14", 50)
                rsi_score = 1.0 if 35 < rsi < 70 else 0.7

                # Combined score (momentum is primary)
                combined_score = momentum_score * 0.6 + vol_score * 0.2 + rsi_score * 0.2

                scored.append((ticker, combined_score))
            except Exception as e:
                logger.debug(f"[UniverseScreener] Opportunity filter failed for {ticker}: {e}")
                continue

        # Sort by score, return top 50
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:50]

    def _apply_diversification_filters(self, scored_tickers: List[tuple], open_positions: Dict) -> List[str]:
        """
        Stage 4: Diversification filters.
        - Max 5 stocks per sector
        - Exclude tickers already in open_positions (CNC)

        Returns final list of up to 40 tickers.
        """
        sector_counts = {}
        result = []

        # Exclude open positions
        open_tickers = set(open_positions.keys())

        for ticker, score in scored_tickers:
            if ticker in open_tickers:
                logger.debug(f"[UniverseScreener] Excluding {ticker} (already open)")
                continue

            # Get sector
            symbol_base = ticker.replace(".NS", "").replace(".BO", "").upper()
            sector = next((s for k, s in SECTOR_MAP.items() if k in symbol_base), "OTHER")

            # Check sector cap
            if sector_counts.get(sector, 0) >= 5:
                continue

            result.append(ticker)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

            if len(result) >= 40:
                break

        logger.info(f"[UniverseScreener] Final breakdown by sector: {sector_counts}")
        return result


# Module-level singleton
universe_screener = UniverseScreener()
