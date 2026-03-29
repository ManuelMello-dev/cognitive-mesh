"""
SocialPlugin
============
Feeds social media and search trend signals into the cognitive mesh.

Sources (all free, no API key required):
  - Reddit public JSON API (r/wallstreetbets, r/CryptoCurrency, r/investing hot posts)
  - CoinGecko trending searches (proxy for social attention)
  - Pytrends Google Trends (BTC, recession, inflation, stock market)

These signals capture retail attention and narrative momentum — often
leading indicators of price moves in both crypto and equities.

Observation schema (domain = "social:<source>"):
  entity_id     : source/topic identifier
  value         : attention/engagement score (normalized 0-1)
  topic         : what is being discussed
  score         : raw upvote/engagement count
  timestamp     : UTC epoch float
  domain_prefix : "social"

Fetch cadence: every 10 minutes
"""
import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("SocialPlugin")

_SUBREDDITS = [
    ("wallstreetbets", "wsb"),
    ("CryptoCurrency", "crypto_reddit"),
    ("investing", "investing_reddit"),
    ("stocks", "stocks_reddit"),
    ("Bitcoin", "bitcoin_reddit"),
]

# Crypto tickers to scan for in Reddit titles
_CRYPTO_TICKERS = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "MATIC", "DOT"}
_STOCK_TICKERS = {"AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "SPY", "QQQ"}


class SocialPlugin(DataPlugin):
    name = "social"
    FETCH_INTERVAL = 600  # 10 minutes

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch: float = 0.0
        self._cache: List[Tuple[Dict, str]] = []

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            headers={
                "User-Agent": "CognitiveMesh/1.0 (market research bot)",
                "Accept": "application/json",
            },
        )
        logger.info("SocialPlugin initialized")

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        if now - self._last_fetch < self.FETCH_INTERVAL and self._cache:
            return self._cache

        tasks = [
            self._fetch_reddit_hot(sub, entity_id)
            for sub, entity_id in _SUBREDDITS
        ]
        tasks.append(self._fetch_coingecko_trending())

        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[Tuple[Dict, str]] = []
        for item in fetched:
            if isinstance(item, list):
                results.extend(item)

        if results:
            self._cache = results
            self._last_fetch = now
            logger.info(f"SocialPlugin: fetched {len(results)} social observations")
        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────
    # Sources
    # ──────────────────────────────────────────

    async def _fetch_reddit_hot(self, subreddit: str, entity_id: str) -> List[Tuple[Dict, str]]:
        """Reddit public JSON API — top 10 hot posts, extract ticker mentions."""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            posts = data.get("data", {}).get("children", [])
            if not posts:
                return []

            # Count ticker mentions across all hot posts
            ticker_mentions: Dict[str, int] = {}
            total_score = 0
            total_comments = 0

            for post in posts:
                pdata = post.get("data", {})
                title = pdata.get("title", "").upper()
                score = pdata.get("score", 0)
                comments = pdata.get("num_comments", 0)
                total_score += score
                total_comments += comments

                for ticker in _CRYPTO_TICKERS | _STOCK_TICKERS:
                    if ticker in title.split() or f"${ticker}" in title:
                        ticker_mentions[ticker] = ticker_mentions.get(ticker, 0) + 1

            # Normalize engagement score (log scale)
            import math
            engagement = round(math.log1p(total_score + total_comments) / 20, 4)
            engagement = min(1.0, engagement)

            results = []

            # Overall subreddit activity observation
            obs = {
                "entity_id": entity_id,
                "value": engagement,
                "total_score": total_score,
                "total_comments": total_comments,
                "post_count": len(posts),
                "top_tickers": sorted(ticker_mentions.items(), key=lambda x: -x[1])[:5],
                "timestamp": time.time(),
            }
            results.append((obs, f"social:{entity_id}"))

            # Individual ticker attention observations
            for ticker, count in ticker_mentions.items():
                ticker_obs = {
                    "entity_id": f"{ticker.lower()}_reddit_mentions",
                    "value": count,
                    "ticker": ticker,
                    "source_subreddit": subreddit,
                    "normalized": min(1.0, count / 10),
                    "viral": count >= 5,
                    "timestamp": time.time(),
                }
                results.append((ticker_obs, f"social:reddit_mention:{ticker.lower()}"))

            return results
        except Exception as e:
            logger.debug(f"Reddit fetch error for r/{subreddit}: {e}")
            return []

    async def _fetch_coingecko_trending(self) -> List[Tuple[Dict, str]]:
        """CoinGecko trending searches — proxy for crypto social attention."""
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            coins = data.get("coins", [])
            results = []
            for rank, coin_entry in enumerate(coins[:10], 1):
                coin = coin_entry.get("item", {})
                symbol = coin.get("symbol", "").upper()
                name = coin.get("name", "")
                market_cap_rank = coin.get("market_cap_rank", 999)

                obs = {
                    "entity_id": f"{symbol.lower()}_trending",
                    "value": round(1.0 - (rank - 1) / 10, 2),  # rank 1 = 1.0, rank 10 = 0.1
                    "ticker": symbol,
                    "name": name,
                    "trend_rank": rank,
                    "market_cap_rank": market_cap_rank,
                    "low_cap_trending": market_cap_rank > 200,  # small cap going viral
                    "timestamp": time.time(),
                }
                results.append((obs, f"social:coingecko_trending:{symbol.lower()}"))

            return results
        except Exception as e:
            logger.debug(f"CoinGecko trending fetch error: {e}")
            return []
