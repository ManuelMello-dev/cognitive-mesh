"""
SentimentPlugin
===============
Feeds market sentiment signals into the cognitive mesh.

Sources (all free, no API key required):
  - Alternative.me Fear & Greed Index (crypto)
  - CNN Fear & Greed Index (equity)
  - AAII Investor Sentiment Survey (weekly bull/bear ratio)
  - Crypto Panic news sentiment headlines
  - Google Trends proxy via pytrends (BTC, S&P500, recession)

Observation schema (domain = "sentiment:<source>"):
  entity_id     : source identifier (e.g. "fear_greed_crypto")
  value         : primary numeric signal (0-100 for fear/greed, ratio for AAII)
  label         : human-readable label ("Extreme Fear", "Greed", etc.)
  timestamp     : UTC epoch float
  domain_prefix : "sentiment"
"""
import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("SentimentPlugin")


class SentimentPlugin(DataPlugin):
    name = "sentiment"

    # Fetch every 10 minutes — sentiment doesn't change second-to-second
    FETCH_INTERVAL = 600

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch: float = 0.0
        self._cache: List[Tuple[Dict, str]] = []

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "CognitiveMesh/1.0"},
        )
        logger.info("SentimentPlugin initialized")

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        # Rate-limit: only re-fetch every FETCH_INTERVAL seconds
        if now - self._last_fetch < self.FETCH_INTERVAL and self._cache:
            return self._cache

        results: List[Tuple[Dict, str]] = []
        tasks = [
            self._fetch_crypto_fear_greed(),
            self._fetch_cnn_fear_greed(),
            self._fetch_aaii_sentiment(),
        ]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        for item in fetched:
            if isinstance(item, list):
                results.extend(item)

        if results:
            self._cache = results
            self._last_fetch = now
        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────
    # Sources
    # ──────────────────────────────────────────

    async def _fetch_crypto_fear_greed(self) -> List[Tuple[Dict, str]]:
        """Alternative.me Fear & Greed Index — free, no key."""
        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                entry = data.get("data", [{}])[0]
                value = int(entry.get("value", 50))
                label = entry.get("value_classification", "Neutral")
                obs = {
                    "entity_id": "fear_greed_crypto",
                    "value": value,
                    "label": label,
                    "timestamp": time.time(),
                    # Derived signals for the cognitive engines
                    "is_extreme_fear": value <= 20,
                    "is_extreme_greed": value >= 80,
                    "normalized": round(value / 100, 4),
                }
                return [(obs, "sentiment:fear_greed_crypto")]
        except Exception as e:
            logger.debug(f"Crypto F&G fetch error: {e}")
            return []

    async def _fetch_cnn_fear_greed(self) -> List[Tuple[Dict, str]]:
        """
        CNN Fear & Greed — scraped from their public API endpoint.
        Returns a 0-100 score for equity market sentiment.
        """
        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
                score = data.get("fear_and_greed", {}).get("score", None)
                rating = data.get("fear_and_greed", {}).get("rating", "Neutral")
                if score is None:
                    return []
                value = round(float(score), 2)
                obs = {
                    "entity_id": "fear_greed_equity",
                    "value": value,
                    "label": rating,
                    "timestamp": time.time(),
                    "is_extreme_fear": value <= 20,
                    "is_extreme_greed": value >= 80,
                    "normalized": round(value / 100, 4),
                }
                return [(obs, "sentiment:fear_greed_equity")]
        except Exception as e:
            logger.debug(f"CNN F&G fetch error: {e}")
            return []

    async def _fetch_aaii_sentiment(self) -> List[Tuple[Dict, str]]:
        """
        AAII Investor Sentiment Survey — weekly bull/bear/neutral percentages.
        Pulled from AAII's public JSON feed.
        """
        try:
            url = "https://www.aaii.com/files/surveys/sentiment.json"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
                # Latest entry is first
                latest = data[0] if isinstance(data, list) and data else {}
                bullish = float(latest.get("Bullish", 0))
                bearish = float(latest.get("Bearish", 0))
                neutral = float(latest.get("Neutral", 0))
                bull_bear_ratio = round(bullish / bearish, 4) if bearish > 0 else 1.0
                obs = {
                    "entity_id": "aaii_sentiment",
                    "value": bull_bear_ratio,
                    "label": "bullish" if bull_bear_ratio > 1.2 else ("bearish" if bull_bear_ratio < 0.8 else "neutral"),
                    "bullish_pct": bullish,
                    "bearish_pct": bearish,
                    "neutral_pct": neutral,
                    "timestamp": time.time(),
                    "normalized": round(bullish / 100, 4),
                }
                return [(obs, "sentiment:aaii")]
        except Exception as e:
            logger.debug(f"AAII sentiment fetch error: {e}")
            return []
