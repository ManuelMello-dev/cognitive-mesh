"""
NewsPlugin
==========
Feeds financial news headline signals into the cognitive mesh.

Sources (all free, no API key required):
  - Reuters Markets RSS
  - Yahoo Finance RSS (top stories, crypto, markets)
  - CoinDesk RSS
  - SEC EDGAR latest filings RSS (8-K, 10-K, earnings)
  - Federal Reserve press releases RSS

Each headline is scored for sentiment using a simple keyword lexicon
(no external NLP dependency) and ingested as an observation.

Observation schema (domain = "news:<source>"):
  entity_id     : source identifier (e.g. "reuters_markets")
  value         : sentiment score (-1.0 to +1.0)
  headline      : raw headline text
  url           : article URL
  sentiment     : "positive" | "negative" | "neutral"
  timestamp     : UTC epoch float
  domain_prefix : "news"

Fetch cadence: every 5 minutes
"""
import asyncio
import logging
import re
import time
from typing import List, Tuple, Dict, Any, Optional
from xml.etree import ElementTree as ET

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("NewsPlugin")

# Simple financial sentiment lexicon — no external dependency
_POSITIVE_WORDS = {
    "surge", "surges", "surging", "rally", "rallies", "rallying", "rise", "rises",
    "rising", "gain", "gains", "gained", "record", "high", "beat", "beats",
    "exceeds", "strong", "growth", "profit", "upgrade", "bullish", "recovery",
    "rebound", "outperform", "breakout", "boost", "boosts", "boosted", "approve",
    "approved", "deal", "partnership", "launch", "launches", "expansion", "hire",
    "hires", "dividend", "buyback", "acquisition", "merger",
}
_NEGATIVE_WORDS = {
    "fall", "falls", "falling", "drop", "drops", "dropped", "decline", "declines",
    "declining", "crash", "crashes", "crashing", "loss", "losses", "miss", "misses",
    "missed", "weak", "warning", "downgrade", "bearish", "recession", "inflation",
    "layoff", "layoffs", "bankrupt", "bankruptcy", "fraud", "investigation", "fine",
    "penalty", "lawsuit", "default", "crisis", "collapse", "collapses", "sell-off",
    "selloff", "plunge", "plunges", "plunging", "tumble", "tumbles", "tumbling",
    "concern", "concerns", "risk", "risks", "fear", "fears", "uncertainty",
    "volatile", "volatility", "debt", "deficit",
}

_RSS_FEEDS = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "yahoo_finance_crypto": "https://finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
    "yahoo_finance_markets": "https://finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "fed_press_releases": "https://www.federalreserve.gov/feeds/press_all.xml",
    "sec_edgar_8k": "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom&startdt=2020-01-01&forms=8-K&_source=efts&hits.hits._source=period_of_report,entity_name,file_date,form_type&hits.hits.total=true",
}


def _score_headline(text: str) -> float:
    """Score a headline from -1.0 (very negative) to +1.0 (very positive)."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


class NewsPlugin(DataPlugin):
    name = "news"
    FETCH_INTERVAL = 300  # 5 minutes
    MAX_HEADLINES_PER_SOURCE = 5  # Only ingest the freshest N per source

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch: float = 0.0
        self._seen_urls: set = set()
        self._cache: List[Tuple[Dict, str]] = []

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "CognitiveMesh/1.0 (news aggregator)"},
        )
        logger.info("NewsPlugin initialized")

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        if now - self._last_fetch < self.FETCH_INTERVAL and self._cache:
            return self._cache

        tasks = [
            self._fetch_rss(source_id, url)
            for source_id, url in _RSS_FEEDS.items()
        ]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Tuple[Dict, str]] = []
        for item in fetched:
            if isinstance(item, list):
                results.extend(item)

        # Only return genuinely new headlines (not seen before)
        new_results = [(obs, domain) for obs, domain in results if obs.get("url") not in self._seen_urls]
        for obs, _ in new_results:
            if obs.get("url"):
                self._seen_urls.add(obs["url"])

        # Trim seen_urls to prevent unbounded growth
        if len(self._seen_urls) > 10_000:
            self._seen_urls = set(list(self._seen_urls)[-5_000:])

        if new_results:
            self._cache = new_results
            self._last_fetch = now
            logger.info(f"NewsPlugin: {len(new_results)} new headlines ingested")
        elif not self._cache:
            self._last_fetch = now  # Avoid hammering on empty result

        return new_results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────
    # RSS Parser
    # ──────────────────────────────────────────

    async def _fetch_rss(self, source_id: str, url: str) -> List[Tuple[Dict, str]]:
        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()

            root = ET.fromstring(text)
            # Handle both RSS 2.0 and Atom formats
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)

            results = []
            for item in items[:self.MAX_HEADLINES_PER_SOURCE]:
                title_el = item.find("title") or item.find("atom:title", ns)
                link_el = item.find("link") or item.find("atom:link", ns)
                pub_el = item.find("pubDate") or item.find("atom:published", ns)

                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                link = link_el.text.strip() if link_el is not None and link_el.text else (
                    link_el.get("href", "") if link_el is not None else ""
                )

                if not title:
                    continue

                score = _score_headline(title)
                sentiment = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")

                obs = {
                    "entity_id": source_id,
                    "value": score,
                    "headline": title[:200],
                    "url": link,
                    "sentiment": sentiment,
                    "source": source_id,
                    "timestamp": time.time(),
                }
                results.append((obs, f"news:{source_id}"))

            return results
        except Exception as e:
            logger.debug(f"RSS fetch error for {source_id}: {e}")
            return []
