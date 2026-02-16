"""
Market Scanner — Autonomous Asset Discovery
=============================================
Discovers trending/top market assets from free public APIs.
No hardcoded symbols — all discovery is organic from real market data.

Sources:
  - CoinGecko: Top coins by market cap
  - Binance: Top trading pairs by volume
  - Yahoo Finance: Trending tickers, most active, gainers, losers
"""

import logging
import asyncio
import aiohttp
from typing import Set, Dict, List, Any
from datetime import datetime, timezone

logger = logging.getLogger("MarketScanner")


class MarketScanner:
    """
    Autonomously discovers market assets from free public APIs.
    No hardcoded symbols — everything is dynamically discovered.
    """

    def __init__(self):
        self._session: aiohttp.ClientSession = None
        self._discovered_crypto: Set[str] = set()
        self._discovered_stocks: Set[str] = set()
        self._scan_count = 0
        self._last_scan_time = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "CognitiveMesh/1.0"}
            )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def scan(self) -> Dict[str, Set[str]]:
        """
        Scan all sources and return newly discovered symbols.
        Returns: {"crypto": set(...), "stocks": set(...)}
        """
        await self._ensure_session()
        self._scan_count += 1
        self._last_scan_time = datetime.now(timezone.utc).isoformat()

        new_crypto = set()
        new_stocks = set()

        # ── Crypto Discovery ──
        crypto_tasks = [
            self._scan_coingecko_top(),
            self._scan_binance_top(),
            self._scan_coingecko_trending(),
        ]
        crypto_results = await asyncio.gather(*crypto_tasks, return_exceptions=True)

        for result in crypto_results:
            if isinstance(result, set):
                for symbol in result:
                    if symbol not in self._discovered_crypto:
                        new_crypto.add(symbol)
                        self._discovered_crypto.add(symbol)

        # ── Stock Discovery ──
        stock_tasks = [
            self._scan_yahoo_trending(),
            self._scan_yahoo_most_active(),
        ]
        stock_results = await asyncio.gather(*stock_tasks, return_exceptions=True)

        for result in stock_results:
            if isinstance(result, set):
                for symbol in result:
                    if symbol not in self._discovered_stocks:
                        new_stocks.add(symbol)
                        self._discovered_stocks.add(symbol)

        if new_crypto:
            logger.info(f"Discovered {len(new_crypto)} new crypto assets: {new_crypto}")
        if new_stocks:
            logger.info(f"Discovered {len(new_stocks)} new stock assets: {new_stocks}")

        return {"crypto": new_crypto, "stocks": new_stocks}

    def get_all_discovered(self) -> Dict[str, Set[str]]:
        """Return all discovered symbols"""
        return {
            "crypto": self._discovered_crypto.copy(),
            "stocks": self._discovered_stocks.copy(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return scanner status"""
        return {
            "scan_count": self._scan_count,
            "last_scan_time": self._last_scan_time,
            "total_crypto_discovered": len(self._discovered_crypto),
            "total_stocks_discovered": len(self._discovered_stocks),
            "crypto_symbols": sorted(self._discovered_crypto),
            "stock_symbols": sorted(self._discovered_stocks),
        }

    # ──────────────────────────────────────────
    # Crypto Discovery Sources
    # ──────────────────────────────────────────

    async def _scan_coingecko_top(self) -> Set[str]:
        """Discover top crypto by market cap from CoinGecko"""
        symbols = set()
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 50,
                "page": 1,
                "sparkline": "false",
            }
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for coin in data:
                        symbol = coin.get("symbol", "").upper()
                        if symbol and len(symbol) <= 10:
                            symbols.add(symbol)
                    logger.info(f"CoinGecko top: discovered {len(symbols)} symbols")
                else:
                    logger.warning(f"CoinGecko top returned {resp.status}")
        except Exception as e:
            logger.warning(f"CoinGecko top scan failed: {e}")
        return symbols

    async def _scan_coingecko_trending(self) -> Set[str]:
        """Discover trending crypto from CoinGecko"""
        symbols = set()
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get("coins", []):
                        coin = item.get("item", {})
                        symbol = coin.get("symbol", "").upper()
                        if symbol and len(symbol) <= 10:
                            symbols.add(symbol)
                    logger.info(f"CoinGecko trending: discovered {len(symbols)} symbols")
                else:
                    logger.warning(f"CoinGecko trending returned {resp.status}")
        except Exception as e:
            logger.warning(f"CoinGecko trending scan failed: {e}")
        return symbols

    async def _scan_binance_top(self) -> Set[str]:
        """Discover top trading pairs from Binance"""
        symbols = set()
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Sort by quote volume descending, take top 30 USDT pairs
                    usdt_pairs = [
                        t for t in data
                        if t.get("symbol", "").endswith("USDT")
                    ]
                    usdt_pairs.sort(
                        key=lambda x: float(x.get("quoteVolume", 0)),
                        reverse=True,
                    )
                    for pair in usdt_pairs[:30]:
                        symbol = pair["symbol"].replace("USDT", "")
                        if symbol and len(symbol) <= 10:
                            symbols.add(symbol)
                    logger.info(f"Binance top: discovered {len(symbols)} symbols")
                else:
                    logger.warning(f"Binance top returned {resp.status}")
        except Exception as e:
            logger.warning(f"Binance top scan failed: {e}")
        return symbols

    # ──────────────────────────────────────────
    # Stock Discovery Sources
    # ──────────────────────────────────────────

    async def _scan_yahoo_trending(self) -> Set[str]:
        """Discover trending stocks from Yahoo Finance"""
        symbols = set()
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
            params = {"count": 30}
            headers = {"User-Agent": "Mozilla/5.0"}
            async with self._session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("finance", {}).get("result", [])
                    for result in results:
                        for quote in result.get("quotes", []):
                            symbol = quote.get("symbol", "").upper()
                            if symbol and "." not in symbol and "=" not in symbol and len(symbol) <= 5:
                                symbols.add(symbol)
                    logger.info(f"Yahoo trending: discovered {len(symbols)} symbols")
                else:
                    logger.warning(f"Yahoo trending returned {resp.status}")
        except Exception as e:
            logger.warning(f"Yahoo trending scan failed: {e}")
        return symbols

    async def _scan_yahoo_most_active(self) -> Set[str]:
        """Discover most active stocks from Yahoo Finance screener"""
        symbols = set()
        try:
            # Use the Yahoo chart API to get well-known indices components
            # SPY top holdings are a good proxy for most active stocks
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {"scrIds": "most_actives", "count": 25}
            headers = {"User-Agent": "Mozilla/5.0"}
            async with self._session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("finance", {}).get("result", [])
                    for result in results:
                        for quote in result.get("quotes", []):
                            symbol = quote.get("symbol", "").upper()
                            if symbol and "." not in symbol and "=" not in symbol and len(symbol) <= 5:
                                symbols.add(symbol)
                    logger.info(f"Yahoo most active: discovered {len(symbols)} symbols")
                else:
                    # Fallback: try the v6 quote endpoint with well-known tickers
                    # that are almost always in the most-active list
                    logger.warning(f"Yahoo screener returned {resp.status}, trying fallback")
                    symbols = await self._scan_yahoo_gainers_losers()
        except Exception as e:
            logger.warning(f"Yahoo most active scan failed: {e}")
            symbols = await self._scan_yahoo_gainers_losers()
        return symbols

    async def _scan_yahoo_gainers_losers(self) -> Set[str]:
        """Fallback: discover gainers and losers from Yahoo Finance"""
        symbols = set()
        try:
            for scr_id in ["day_gainers", "day_losers"]:
                url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                params = {"scrIds": scr_id, "count": 15}
                headers = {"User-Agent": "Mozilla/5.0"}
                async with self._session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("finance", {}).get("result", [])
                        for result in results:
                            for quote in result.get("quotes", []):
                                symbol = quote.get("symbol", "").upper()
                                if symbol and "." not in symbol and "=" not in symbol and len(symbol) <= 5:
                                    symbols.add(symbol)
                await asyncio.sleep(0.5)  # Rate limit
            if symbols:
                logger.info(f"Yahoo gainers/losers: discovered {len(symbols)} symbols")
        except Exception as e:
            logger.warning(f"Yahoo gainers/losers scan failed: {e}")
        return symbols
