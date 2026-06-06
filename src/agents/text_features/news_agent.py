"""
News sentiment agent.

Source: yfinance Ticker(t).news — free, returns ~10 recent headlines per ticker
with publisher, date, and link.

For each headline we run a small LLM call asking for a sentiment score in
[-3, +3] and confidence in [0, 1]. We aggregate the headlines as a
confidence-weighted mean for the ticker.

Fallback (no OPENAI_API_KEY): keyword-based sentiment using a small lexicon
of bullish/bearish finance terms. Less accurate but deterministic.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import List

import numpy as np

from src.agents.text_features.base import TextFeature, TextFeatureAgent

logger = logging.getLogger(__name__)


# Tiny lexicon — used only when no OpenAI key is configured
_BULLISH_TERMS = {
    "beats", "beat", "surge", "rally", "soar", "jump", "upgrade", "raises",
    "outperform", "strong", "growth", "profit", "record", "approval", "wins",
    "expansion", "breakthrough", "exceeds",
}
_BEARISH_TERMS = {
    "miss", "misses", "tumble", "plunge", "crash", "fall", "drop", "downgrade",
    "cuts", "weak", "loss", "warning", "lawsuit", "probe", "scandal",
    "decline", "concerns", "recall", "delays", "fails",
}


class NewsAgent(TextFeatureAgent):
    """Per-ticker news sentiment from yfinance headlines + LLM (or lexicon fallback)."""

    name = "news_sentiment"
    cache_ttl_seconds = 6 * 3600   # news ages fast — 6h cache
    max_headlines = 10
    max_age_days = 14

    def compute(self, ticker: str) -> TextFeature:
        cache_key = f"news::{ticker}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        headlines = self._fetch_headlines(ticker)
        if not headlines:
            feat = TextFeature.empty(self.name, f"no recent headlines for {ticker}")
            self._cache_set(cache_key, feat)
            return feat

        client = self._get_openai()
        if client is not None:
            feat = self._llm_aggregate(ticker, headlines)
        else:
            feat = self._lexicon_aggregate(ticker, headlines)

        self._cache_set(cache_key, feat)
        return feat

    # ── fetchers ────────────────────────────────────────────────────────────

    def _fetch_headlines(self, ticker: str) -> List[dict]:
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            news = tk.news or []
        except Exception as e:
            logger.warning(f"NewsAgent: yfinance news fetch failed for {ticker}: {e}")
            return []

        cutoff_ts = (datetime.utcnow() - timedelta(days=self.max_age_days)).timestamp()
        out = []
        for item in news[: self.max_headlines * 2]:   # buffer for filtering
            # yfinance schema changed a few times — be defensive
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            pub_date = content.get("pubDate") or item.get("providerPublishTime")
            if not title:
                continue
            # Normalise pub_date to unix ts
            ts = None
            if isinstance(pub_date, (int, float)):
                ts = float(pub_date)
            elif isinstance(pub_date, str):
                try:
                    ts = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp()
                except Exception:
                    pass
            if ts and ts < cutoff_ts:
                continue
            out.append({"title": title, "ts": ts})
            if len(out) >= self.max_headlines:
                break
        return out

    # ── aggregators ─────────────────────────────────────────────────────────

    def _llm_aggregate(self, ticker: str, headlines: List[dict]) -> TextFeature:
        joined = "\n".join(f"- {h['title']}" for h in headlines)
        prompt = (
            f"Recent news headlines for {ticker}:\n\n{joined}\n\n"
            "Assess the OVERALL sentiment for the stock. Return JSON: "
            '{"score": float in [-3, 3], "confidence": float in [0, 1], '
            '"rationale": "brief"}'
        )
        out = self._llm_score(prompt)
        if not out:
            # LLM failure → lexicon fallback so we still return something
            return self._lexicon_aggregate(ticker, headlines)

        score = float(out.get("score", 0.0))
        conf = float(out.get("confidence", 0.0))
        # Recency-weight confidence: more recent + more headlines = higher confidence
        coverage_mult = min(1.0, len(headlines) / self.max_headlines)
        return TextFeature(
            feature_name=self.name,
            score=max(-3.0, min(3.0, score)),
            confidence=max(0.0, min(1.0, conf * coverage_mult)),
            as_of=datetime.utcnow(),
            metadata={"n_headlines": len(headlines), "source": "llm"},
            rationale=str(out.get("rationale", ""))[:200],
        )

    def _lexicon_aggregate(self, ticker: str, headlines: List[dict]) -> TextFeature:
        net = 0
        total = 0
        for h in headlines:
            words = set(h["title"].lower().split())
            bull = len(words & _BULLISH_TERMS)
            bear = len(words & _BEARISH_TERMS)
            if bull or bear:
                net += bull - bear
                total += bull + bear
        if total == 0:
            return TextFeature(
                feature_name=self.name, score=0.0, confidence=0.1,
                as_of=datetime.utcnow(),
                metadata={"n_headlines": len(headlines), "source": "lexicon", "neutral": True},
                rationale="No lexicon matches in headlines",
            )
        # Scale: net=+3 strongly bullish, net=-3 strongly bearish
        score = max(-3.0, min(3.0, net))
        # Confidence: scale with # of matches, capped
        conf = min(0.7, total / 8.0)   # lexicon caps at 0.7 (less reliable than LLM)
        return TextFeature(
            feature_name=self.name,
            score=score,
            confidence=conf,
            as_of=datetime.utcnow(),
            metadata={"n_headlines": len(headlines), "n_matches": total, "source": "lexicon"},
            rationale=f"Lexicon: {total} matches across {len(headlines)} headlines",
        )
