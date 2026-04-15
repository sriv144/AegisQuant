"""
Alternative Data Engineering
============================
Hooks into Real-time APIs to extract news, sentiment, or macroeconomic alt-data.
"""

import os
import requests
import logging
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

from src import config  # noqa: F401

logger = logging.getLogger(__name__)

class AlternativeDataCollector:
    def __init__(self):
        # Supports API keys from NewsAPI.org or Alpaca News
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.mock_mode = not bool(self.newsapi_key)
        
    def get_recent_news(self, ticker: str, days_back: int = 3) -> List[Dict[str, Any]]:
        """
        Fetches global news articles mentioning the ticker.
        If no API key is found, safely falls back to standard dummy data to avoid execution loops crashing.
        """
        if self.mock_mode:
            logger.debug(f"[AltData] API Key missing, returning mock sentiment for {ticker}")
            return self._generate_mock_sentiment(ticker)
            
        try:
            # We attempt a generic NewsAPI query
            from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            url = f"https://newsapi.org/v2/everything?q={ticker}&from={from_date}&sortBy=publishedAt&apiKey={self.newsapi_key}"
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                logger.warning(f"News API returned {response.status_code}. Reverting to mock.")
                return self._generate_mock_sentiment(ticker)
                
            articles = response.json().get("articles", [])
            structured_news = []
            
            for art in articles[:10]: # cap at 10 to limit LLM context bloat
                # A simple mock explicit sentiment score generator based on keyword matching
                # since full inference requires the actual LLM Agents defined in Phase 2
                text = ((art.get("title") or "") + " " + (art.get("description") or "")).lower()
                
                score = 0.0
                if any(w in text for w in ["soar", "jump", "record", "profit", "beat"]):
                    score += 0.5
                if any(w in text for w in ["plunge", "crash", "miss", "loss", "bankrupt"]):
                    score -= 0.5
                    
                structured_news.append({
                    "timestamp": art.get("publishedAt", ""),
                    "headline": art.get("title", ""),
                    "sentiment_score": float(np.clip(score, -1.0, 1.0))
                })
                
            return structured_news
            
        except Exception as e:
            logger.error(f"Alt Data Pipeline crashed on {ticker}: {e}")
            return self._generate_mock_sentiment(ticker)
            
    def _generate_mock_sentiment(self, ticker: str) -> List[Dict[str, Any]]:
        return [{
            "timestamp": datetime.utcnow().isoformat(),
            "headline": f"AegisQuant Synthetic Update: Market moving normally on {ticker}.",
            "sentiment_score": float(np.random.uniform(-1, 1))
        }]
    
alt_data = AlternativeDataCollector()
