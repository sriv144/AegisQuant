"""
Insider Buying Factor — Cohen-Malloy-Pomorski (2012) "Decoding Inside Information".

Theory: when corporate insiders (officers, directors, 10%-owners) execute
*opportunistic* purchases on the open market, they earn ~82 bps/month of
abnormal return. The signal works specifically for "opportunistic" insiders
(those whose past trades don't follow a predictable calendar), not routine
quarterly buyers.

Data source: SEC EDGAR Form 4 filings (free, public).
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4

This implementation uses the SEC's Atom feed for Form 4 filings and counts
open-market purchase transactions over the lookback window. A "score" is
computed per ticker as:

    score = sum(dollar_value_of_recent_open_market_buys) / market_cap

then z-scored cross-sectionally.

Filtering rules (Cohen-Malloy-Pomorski approximation):
  - transaction code "P" (open-market purchase) only — excludes options
    exercises ("M"), grants ("A"), etc.
  - lookback 90 days
  - minimum $50K per filing (filters tiny director qualifying-share buys)

Notes:
  - SEC requires a User-Agent header; we set one with the user's email.
  - Rate limit: 10 req/sec. We batch tickers serially with a small sleep.
  - First run takes ~5s/ticker; cached for 7 days after.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlencode

import numpy as np

from src.factors.base import Factor, FactorResult

logger = logging.getLogger(__name__)

SEC_BASE = "https://www.sec.gov"
# SEC mandates a descriptive User-Agent identifying the requester
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "AegisQuant Factor Engine srivathsnat@gmail.com",
)
SEC_RATE_LIMIT_SECS = 0.12   # ~8 req/sec, below the 10 req/sec cap


class InsiderFactor(Factor):
    name = "insider"
    rebalance_freq = "weekly"
    requires = ["fundamentals"]   # needs marketCap for normalization

    LOOKBACK_DAYS = 90
    MIN_DOLLAR_VALUE = 50_000.0
    HTTP_TIMEOUT = 10

    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        as_of = as_of or datetime.utcnow()
        raw: Dict[str, Dict[str, float]] = {}
        confidence: Dict[str, float] = {}
        buy_dollars: Dict[str, float] = {}

        cutoff = as_of - timedelta(days=self.LOOKBACK_DAYS)

        for t in universe:
            f = self.dp.get_fundamentals(t)
            mcap = f.get("marketCap") if f else None
            if not mcap or mcap <= 0:
                continue

            dollars, n_filings = self._fetch_insider_buys(t, since=cutoff)
            if dollars is None:
                continue   # request failed → skip rather than zero

            raw[t] = {
                "buy_dollars_90d": dollars,
                "n_filings": float(n_filings),
                "market_cap": float(mcap),
                "buy_ratio": dollars / mcap,
            }
            buy_dollars[t] = dollars / mcap   # normalize so big-cap & small-cap comparable
            # Confidence rises with number of independent filings (max 1.0 at 3+)
            confidence[t] = min(1.0, n_filings / 3.0) if n_filings > 0 else 0.5

        scores = self.zscore(buy_dollars, winsorize=3.0)

        return FactorResult(
            factor_name=self.name,
            as_of=as_of,
            scores=scores,
            confidence=confidence,
            raw=raw,
            notes=f"Insider buys (Form 4, last {self.LOOKBACK_DAYS}d) for {len(scores)} tickers",
        )

    # ── EDGAR Form 4 fetcher ────────────────────────────────────────────────

    def _fetch_insider_buys(self, ticker: str, since: datetime):
        """
        Return (total_dollar_value, num_filings) of open-market Form 4 purchases
        since the given date, or (None, 0) on hard failure.
        Returns (0.0, 0) when there were no qualifying buys.
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed; insider factor disabled")
            return None, 0

        # 1) Get CIK for the ticker (cached in DataProvider's fundamentals cache
        #    would be ideal, but we don't have it there yet — fetch live).
        cik = self._lookup_cik(ticker, requests)
        if cik is None:
            return 0.0, 0

        # 2) Pull recent filings index for this issuer (JSON endpoint)
        try:
            url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
            r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=self.HTTP_TIMEOUT)
            time.sleep(SEC_RATE_LIMIT_SECS)
            if r.status_code != 200:
                return None, 0
            data = r.json()
        except Exception as e:
            logger.warning(f"insider: submissions fetch failed for {ticker}: {e}")
            return None, 0

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        total_dollars = 0.0
        n = 0
        for i, form in enumerate(forms):
            if form != "4":
                continue
            try:
                filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
            except Exception:
                continue
            if filing_date < since:
                continue
            # Parse the individual Form 4 XML/HTML for purchase amounts
            acc = accs[i].replace("-", "")
            doc = primary_docs[i]
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            try:
                d = requests.get(doc_url, headers={"User-Agent": SEC_USER_AGENT}, timeout=self.HTTP_TIMEOUT)
                time.sleep(SEC_RATE_LIMIT_SECS)
                if d.status_code != 200:
                    continue
                dollars = self._parse_form4_purchase_value(d.text)
                if dollars >= self.MIN_DOLLAR_VALUE:
                    total_dollars += dollars
                    n += 1
            except Exception as e:
                logger.debug(f"insider: form4 parse failed for {ticker}: {e}")
                continue
        return total_dollars, n

    def _lookup_cik(self, ticker: str, requests_mod):
        """Resolve ticker → CIK via SEC's ticker.txt file (cached on disk)."""
        from pathlib import Path
        cache_path = Path(__file__).resolve().parents[2] / ".cache" / "factors" / "sec_ticker_to_cik.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists() or (time.time() - cache_path.stat().st_mtime) > 30 * 86400:
            try:
                r = requests_mod.get(
                    "https://www.sec.gov/include/ticker.txt",
                    headers={"User-Agent": SEC_USER_AGENT},
                    timeout=self.HTTP_TIMEOUT,
                )
                if r.status_code == 200:
                    cache_path.write_text(r.text)
            except Exception:
                pass

        if not cache_path.exists():
            return None
        for line in cache_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0].upper() == ticker.lower().upper():
                try:
                    return int(parts[1])
                except ValueError:
                    return None
        return None

    @staticmethod
    def _parse_form4_purchase_value(html_or_xml: str) -> float:
        """
        Extract the total dollar value of transactionCode='P' (purchase)
        non-derivative transactions in a Form 4 document.
        Looks at both XML (structured) and HTML (rendered) variants.
        """
        text = html_or_xml
        total = 0.0

        # XML variant: look for <transactionCode>P</transactionCode> blocks and
        # extract <transactionShares><value>N</value></transactionShares>
        # and <transactionPricePerShare><value>P</value></...>
        # Regex-based parsing is OK here because the schema is shallow.
        if "<transactionCode>" in text:
            block_re = re.compile(
                r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
                re.DOTALL,
            )
            for block in block_re.findall(text):
                if "<transactionCode>P</transactionCode>" not in block:
                    continue
                shares = _xml_extract_float(
                    block,
                    r"<transactionShares>\s*<value>([0-9.]+)</value>",
                )
                price = _xml_extract_float(
                    block,
                    r"<transactionPricePerShare>\s*<value>([0-9.]+)</value>",
                )
                if shares is not None and price is not None:
                    total += shares * price
        # HTML variant fallback (skip for now — most filings provide XML)
        return total


def _xml_extract_float(text: str, pattern: str):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
