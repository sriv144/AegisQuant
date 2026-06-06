"""
10-K Filing Risk-Factor Delta Agent.

Detects YoY changes in the "Item 1A — Risk Factors" section of a company's
annual report (10-K). New risks added between reports often precede material
stock moves — companies are legally required to disclose them when they
become material.

Approach
--------
1. Fetch the most recent 10-K and the prior 10-K from SEC EDGAR (free).
2. Extract the "Item 1A. Risk Factors" section from each.
3. Diff at the sentence level — what sentences are NEW in the latest filing?
4. Ask the LLM: "Are these new risks bearish, neutral, or material?"
5. Emit a score in [-3, +3] where negative = newly-disclosed risks are bearish.

Fallback (no OPENAI_API_KEY): simple count of new risk-section words.
A large increase (>15%) is treated as mildly bearish.

Cache: 30 days per ticker (10-Ks don't change between annual reports).
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import List, Optional, Tuple

from src.agents.text_features.base import TextFeature, TextFeatureAgent
from src.factors.insider import SEC_USER_AGENT, SEC_RATE_LIMIT_SECS

logger = logging.getLogger(__name__)


class Filing10KAgent(TextFeatureAgent):
    name = "filing_10k_delta"
    cache_ttl_seconds = 30 * 86400   # 30 days — 10-Ks update annually
    HTTP_TIMEOUT = 15
    MAX_NEW_CHARS_FOR_LLM = 6000     # truncate to keep token cost bounded

    def compute(self, ticker: str) -> TextFeature:
        cache_key = f"f10k::{ticker}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        latest, prior = self._fetch_two_recent_10ks(ticker)
        if latest is None:
            feat = TextFeature.empty(self.name, f"no recent 10-K for {ticker}")
            self._cache_set(cache_key, feat)
            return feat

        latest_risks = self._extract_risk_section(latest)
        prior_risks = self._extract_risk_section(prior) if prior else ""

        if not latest_risks:
            feat = TextFeature.empty(self.name, "could not parse risk section")
            self._cache_set(cache_key, feat)
            return feat

        new_text = self._diff_new_text(latest_risks, prior_risks)
        if not new_text or len(new_text) < 200:
            # No meaningful new risks → neutral
            feat = TextFeature(
                feature_name=self.name, score=0.0, confidence=0.3,
                as_of=datetime.utcnow(),
                metadata={"new_chars": len(new_text), "neutral": True},
                rationale="No material risk-factor additions YoY",
            )
            self._cache_set(cache_key, feat)
            return feat

        client = self._get_openai()
        if client is not None:
            feat = self._llm_evaluate(ticker, new_text)
        else:
            feat = self._heuristic_evaluate(ticker, latest_risks, prior_risks)

        self._cache_set(cache_key, feat)
        return feat

    # ── SEC EDGAR fetchers ─────────────────────────────────────────────────

    def _fetch_two_recent_10ks(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            import requests
        except ImportError:
            return None, None
        cik = self._lookup_cik(ticker, requests)
        if cik is None:
            return None, None

        try:
            url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
            r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT},
                             timeout=self.HTTP_TIMEOUT)
            time.sleep(SEC_RATE_LIMIT_SECS)
            if r.status_code != 200:
                return None, None
            data = r.json()
        except Exception as e:
            logger.warning(f"Filing10K: submissions fetch failed for {ticker}: {e}")
            return None, None

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        ten_ks = []
        for i, f in enumerate(forms):
            if f == "10-K":
                ten_ks.append((accs[i], docs[i]))
            if len(ten_ks) >= 2:
                break
        if not ten_ks:
            return None, None

        def fetch(acc, doc):
            acc_clean = acc.replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
            try:
                d = requests.get(doc_url, headers={"User-Agent": SEC_USER_AGENT},
                                 timeout=self.HTTP_TIMEOUT)
                time.sleep(SEC_RATE_LIMIT_SECS)
                if d.status_code == 200:
                    return d.text
            except Exception as e:
                logger.warning(f"Filing10K: fetch failed: {e}")
            return None

        latest = fetch(*ten_ks[0])
        prior = fetch(*ten_ks[1]) if len(ten_ks) > 1 else None
        return latest, prior

    def _lookup_cik(self, ticker: str, requests_mod) -> Optional[int]:
        # Reuse the same disk cache as InsiderFactor
        from pathlib import Path
        cache_path = Path(__file__).resolve().parents[3] / ".cache" / "factors" / "sec_ticker_to_cik.txt"
        if not cache_path.exists() or (time.time() - cache_path.stat().st_mtime) > 30 * 86400:
            try:
                r = requests_mod.get(
                    "https://www.sec.gov/include/ticker.txt",
                    headers={"User-Agent": SEC_USER_AGENT},
                    timeout=self.HTTP_TIMEOUT,
                )
                if r.status_code == 200:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(r.text)
            except Exception:
                pass
        if not cache_path.exists():
            return None
        for line in cache_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0].upper() == ticker.upper():
                try:
                    return int(parts[1])
                except ValueError:
                    return None
        return None

    # ── parsing ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_risk_section(html: str) -> str:
        """Best-effort extraction of Item 1A. Risk Factors text."""
        if not html:
            return ""
        # Strip HTML tags brutally
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        # Find "Item 1A" through next "Item 1B" or "Item 2"
        m = re.search(
            r"Item\s*1A[\s\.\-:]+Risk Factors(.*?)(Item\s*1B|Item\s*2[\s\.\-:])",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        # Fallback: just find "Risk Factors" heading
        m2 = re.search(r"Risk Factors(.*?)(Unresolved Staff Comments|Properties)",
                       text, re.IGNORECASE | re.DOTALL)
        return m2.group(1).strip() if m2 else ""

    @staticmethod
    def _diff_new_text(latest: str, prior: str) -> str:
        """Return sentences present in latest but not in prior (rough sentence-level diff)."""
        if not prior:
            return latest[:8000]
        prior_sents = set(s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", prior) if len(s) > 30)
        new = []
        for s in re.split(r"(?<=[.!?])\s+", latest):
            s_clean = s.strip()
            if len(s_clean) > 30 and s_clean.lower() not in prior_sents:
                new.append(s_clean)
        return " ".join(new)

    # ── evaluators ─────────────────────────────────────────────────────────

    def _llm_evaluate(self, ticker: str, new_text: str) -> TextFeature:
        snippet = new_text[: self.MAX_NEW_CHARS_FOR_LLM]
        prompt = (
            f"The following sentences are NEW additions to {ticker}'s 10-K risk-factor "
            f"section vs the prior year's filing:\n\n{snippet}\n\n"
            "Assess whether these newly-disclosed risks are material and bearish for the stock. "
            'Return JSON: {"score": float in [-3, 3] where negative = bearish new risks, '
            '"confidence": float in [0, 1], "rationale": "brief"}'
        )
        out = self._llm_score(prompt, max_tokens=250)
        if not out:
            return TextFeature.empty(self.name, "LLM call failed")
        return TextFeature(
            feature_name=self.name,
            score=max(-3.0, min(3.0, float(out.get("score", 0.0)))),
            confidence=max(0.0, min(1.0, float(out.get("confidence", 0.0)))),
            as_of=datetime.utcnow(),
            metadata={"new_chars": len(new_text), "source": "llm"},
            rationale=str(out.get("rationale", ""))[:300],
        )

    def _heuristic_evaluate(self, ticker: str, latest: str, prior: str) -> TextFeature:
        if not prior:
            return TextFeature.empty(self.name, "no prior 10-K to compare")
        delta_pct = (len(latest) - len(prior)) / max(len(prior), 1)
        # >15% growth in risk section = mildly bearish, >30% = clearly bearish
        if delta_pct > 0.30:
            score, conf = -1.5, 0.5
        elif delta_pct > 0.15:
            score, conf = -0.5, 0.3
        elif delta_pct < -0.15:
            score, conf = 0.5, 0.3   # risks shrank — mildly bullish
        else:
            score, conf = 0.0, 0.2
        return TextFeature(
            feature_name=self.name, score=score, confidence=conf,
            as_of=datetime.utcnow(),
            metadata={"delta_pct": delta_pct, "source": "heuristic"},
            rationale=f"Risk section size changed {delta_pct*100:+.1f}% YoY (heuristic)",
        )
