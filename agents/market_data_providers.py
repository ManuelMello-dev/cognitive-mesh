"""
Market Data Providers — Circuit Breaker Architecture
=====================================================
Every provider is wrapped in a CircuitBreaker. If a provider fails N times
consecutively it is tripped OPEN and skipped for a cooldown period, then
re-tested (HALF_OPEN). On success it resets to CLOSED.

Provider cascade order (first success wins):
  STOCKS:  Yahoo Finance Chart → CNBC → Alpha Vantage* → Polygon* → Twelve Data* → FMP* → Tiingo*
  CRYPTO:  Binance → CoinGecko → CryptoCompare → Kraken → KuCoin → MEXC

  * = requires env-var API key; auto-disabled when key is absent.
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("MarketDataProviders")

# ──────────────────────────────────────────────
# Circuit Breaker
# ──────────────────────────────────────────────

class CBState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Per-provider circuit breaker with configurable thresholds."""

    def __init__(self, name: str, failure_threshold: int = 3, cooldown: float = 60.0):
        self.name = name
        self.state = CBState.CLOSED
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._opened_at: float = 0.0

    @property
    def allow_request(self) -> bool:
        if self.state == CBState.CLOSED:
            return True
        if self.state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self.cooldown:
                self.state = CBState.HALF_OPEN
                logger.info(f"CircuitBreaker [{self.name}] → HALF_OPEN (testing)")
                return True
            return False
        # HALF_OPEN — allow one probe
        return True

    def record_success(self):
        if self.state != CBState.CLOSED:
            logger.info(f"CircuitBreaker [{self.name}] → CLOSED (recovered)")
        self.failures = 0
        self.state = CBState.CLOSED

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = CBState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                f"CircuitBreaker [{self.name}] → OPEN after {self.failures} failures "
                f"(cooldown {self.cooldown}s)"
            )


# ──────────────────────────────────────────────
# Base Provider
# ──────────────────────────────────────────────

class BaseProvider:
    """Abstract base for all market data providers."""

    NAME: str = "base"
    REQUIRES_KEY: bool = False
    KEY_ENV: str = ""

    def __init__(self):
        self.api_key: Optional[str] = None
        if self.REQUIRES_KEY:
            self.api_key = os.getenv(self.KEY_ENV, "").strip() or None
        self.breaker = CircuitBreaker(self.NAME, failure_threshold=3, cooldown=90.0)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def enabled(self) -> bool:
        if self.REQUIRES_KEY and not self.api_key:
            return False
        return True

    @property
    def available(self) -> bool:
        return self.enabled and self.breaker.allow_request

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=12)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def _tick(self, symbol: str, price: float, volume: float = 0.0,
              change: float = 0.0, extra: Optional[Dict] = None) -> Dict[str, Any]:
        """Normalised tick output."""
        tick = {
            "symbol": symbol.upper(),
            "price": price,
            "volume": volume,
            "change_24h": change,
            "source": self.NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            tick.update(extra)
        return tick


# ──────────────────────────────────────────────
# FREE STOCK PROVIDERS
# ──────────────────────────────────────────────

class YahooFinanceChartProvider(BaseProvider):
    """Yahoo Finance v8 chart API — free, no key."""
    NAME = "yahoo_chart"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": "1d", "range": "5d"}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            result = data["chart"]["result"][0]
            meta = result["meta"]
            price = meta.get("regularMarketPrice", 0)
            volume = meta.get("regularMarketVolume", 0)
            prev = meta.get("chartPreviousClose", price)
            change = ((price - prev) / prev * 100) if prev else 0
            return self._tick(symbol, price, volume, change, {
                "market_state": meta.get("marketState", "UNKNOWN"),
                "exchange": meta.get("exchangeName", ""),
                "currency": meta.get("currency", "USD"),
            })


class CNBCQuoteProvider(BaseProvider):
    """CNBC REST quote API — free, no key."""
    NAME = "cnbc"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
        params = {
            "symbols": symbol, "requestMethod": "itv", "noform": "1",
            "partnerId": "2", "fund": "1", "exthrs": "1", "output": "json", "events": "1"
        }
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            quotes = data.get("FormattedQuoteResult", {}).get("FormattedQuote", [])
            if not quotes:
                raise Exception("No quote data")
            q = quotes[0]
            price = float(q.get("last", 0))
            volume = float(q.get("volume", "0").replace(",", ""))
            change = float(q.get("change_pct", "0").replace("%", ""))
            return self._tick(symbol, price, volume, change, {
                "name": q.get("name", ""),
                "exchange": q.get("exchange", ""),
            })


# ──────────────────────────────────────────────
# API-KEY STOCK PROVIDERS
# ──────────────────────────────────────────────

class AlphaVantageProvider(BaseProvider):
    NAME = "alphavantage"
    REQUIRES_KEY = True
    KEY_ENV = "ALPHA_VANTAGE_API_KEY"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = "https://www.alphavantage.co/query"
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self.api_key}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            gq = data.get("Global Quote", {})
            if not gq:
                raise Exception("Empty response or rate limited")
            price = float(gq.get("05. price", 0))
            volume = float(gq.get("06. volume", 0))
            change = float(gq.get("10. change percent", "0%").replace("%", ""))
            return self._tick(symbol, price, volume, change)


class PolygonProvider(BaseProvider):
    NAME = "polygon"
    REQUIRES_KEY = True
    KEY_ENV = "POLYGON_API_KEY"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        params = {"adjusted": "true", "apiKey": self.api_key}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            results = data.get("results", [])
            if not results:
                raise Exception("No results")
            r = results[0]
            price = r.get("c", 0)
            volume = r.get("v", 0)
            change = ((r["c"] - r["o"]) / r["o"] * 100) if r.get("o") else 0
            return self._tick(symbol, price, volume, change)


class TwelveDataProvider(BaseProvider):
    NAME = "twelvedata"
    REQUIRES_KEY = True
    KEY_ENV = "TWELVE_DATA_API_KEY"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = "https://api.twelvedata.com/quote"
        params = {"symbol": symbol, "apikey": self.api_key}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            if "code" in data:
                raise Exception(data.get("message", "API error"))
            price = float(data.get("close", 0))
            volume = float(data.get("volume", 0))
            change = float(data.get("percent_change", 0))
            return self._tick(symbol, price, volume, change, {
                "name": data.get("name", ""),
                "exchange": data.get("exchange", ""),
            })


class FMPProvider(BaseProvider):
    """Financial Modeling Prep — free tier 250 calls/day."""
    NAME = "fmp"
    REQUIRES_KEY = True
    KEY_ENV = "FMP_API_KEY"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}"
        params = {"apikey": self.api_key}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            if not data:
                raise Exception("Empty response")
            q = data[0]
            return self._tick(symbol, q.get("price", 0), q.get("volume", 0))


class TiingoProvider(BaseProvider):
    """Tiingo — free tier 500 calls/hour."""
    NAME = "tiingo"
    REQUIRES_KEY = True
    KEY_ENV = "TIINGO_API_KEY"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = f"https://api.tiingo.com/iex/{symbol}"
        headers = {"Authorization": f"Token {self.api_key}"}
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            if not data:
                raise Exception("Empty response")
            q = data[0]
            price = q.get("last") or q.get("tngoLast", 0)
            volume = q.get("volume", 0)
            prev = q.get("prevClose", price)
            change = ((price - prev) / prev * 100) if prev else 0
            return self._tick(symbol, price, volume, change)


# ──────────────────────────────────────────────
# FREE CRYPTO PROVIDERS
# ──────────────────────────────────────────────

# CoinGecko symbol mapping
_CG_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin", "DOT": "polkadot",
    "AVAX": "avalanche-2", "MATIC": "matic-network", "LINK": "chainlink",
    "UNI": "uniswap", "ATOM": "cosmos", "LTC": "litecoin", "SHIB": "shiba-inu",
    "TRX": "tron", "NEAR": "near", "APT": "aptos", "ARB": "arbitrum",
    "OP": "optimism", "SUI": "sui", "SEI": "sei-network", "FIL": "filecoin",
    "PEPE": "pepe", "WIF": "dogwifcoin", "RENDER": "render-token",
    "FET": "fetch-ai", "INJ": "injective-protocol", "BONK": "bonk",
}


class BinanceProvider(BaseProvider):
    """Binance public ticker — free, no key, very reliable."""
    NAME = "binance"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        pair = f"{symbol.upper()}USDT"
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {"symbol": pair}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            if "code" in data:
                raise Exception(data.get("msg", "API error"))
            price = float(data["lastPrice"])
            volume = float(data["volume"]) * price  # convert to USD volume
            change = float(data["priceChangePercent"])
            return self._tick(symbol, price, volume, change, {
                "high_24h": float(data["highPrice"]),
                "low_24h": float(data["lowPrice"]),
                "trades_24h": int(data["count"]),
            })


class CoinGeckoProvider(BaseProvider):
    """CoinGecko — free, no key, rate limited (10-30 req/min)."""
    NAME = "coingecko"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        cg_id = _CG_MAP.get(symbol.upper())
        if not cg_id:
            raise Exception(f"Unknown CoinGecko mapping for {symbol}")
        session = await self._get_session()
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": cg_id, "vs_currencies": "usd",
            "include_24hr_vol": "true", "include_24hr_change": "true",
            "include_market_cap": "true",
        }
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            coin = data.get(cg_id, {})
            if not coin:
                raise Exception(f"No data for {cg_id}")
            return self._tick(
                symbol,
                coin["usd"],
                coin.get("usd_24h_vol", 0),
                coin.get("usd_24h_change", 0),
                {"market_cap": coin.get("usd_market_cap", 0)},
            )


class CryptoCompareProvider(BaseProvider):
    """CryptoCompare — free, no key required for basic endpoints."""
    NAME = "cryptocompare"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = "https://min-api.cryptocompare.com/data/pricemultifull"
        params = {"fsyms": symbol.upper(), "tsyms": "USD"}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            raw = data.get("RAW", {}).get(symbol.upper(), {}).get("USD", {})
            if not raw:
                raise Exception(f"No data for {symbol}")
            return self._tick(
                symbol,
                raw.get("PRICE", 0),
                raw.get("TOTALVOLUME24HTO", 0),
                raw.get("CHANGEPCT24HOUR", 0),
                {
                    "market_cap": raw.get("MKTCAP", 0),
                    "high_24h": raw.get("HIGH24HOUR", 0),
                    "low_24h": raw.get("LOW24HOUR", 0),
                },
            )


class KrakenProvider(BaseProvider):
    """Kraken public ticker — free, no key."""
    NAME = "kraken"

    _PAIR_MAP = {
        "BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "ADA": "ADAUSD",
        "DOGE": "DOGEUSD", "DOT": "DOTUSD", "LINK": "LINKUSD", "AVAX": "AVAXUSD",
        "MATIC": "MATICUSD", "XRP": "XRPUSD", "LTC": "LTCUSD", "UNI": "UNIUSD",
        "ATOM": "ATOMUSD", "NEAR": "NEARUSD", "APT": "APTUSD", "ARB": "ARBUSD",
        "FIL": "FILUSD", "INJ": "INJUSD", "FET": "FETUSD", "TRX": "TRXUSD",
    }

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        pair = self._PAIR_MAP.get(symbol.upper())
        if not pair:
            raise Exception(f"Unknown Kraken pair for {symbol}")
        session = await self._get_session()
        url = "https://api.kraken.com/0/public/Ticker"
        params = {"pair": pair}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            if data.get("error"):
                raise Exception(str(data["error"]))
            result = list(data["result"].values())[0]
            price = float(result["c"][0])
            volume = float(result["v"][1]) * price
            open_price = float(result["o"])
            change = ((price - open_price) / open_price * 100) if open_price else 0
            return self._tick(symbol, price, volume, change, {
                "high_24h": float(result["h"][1]),
                "low_24h": float(result["l"][1]),
            })


class KuCoinProvider(BaseProvider):
    """KuCoin public ticker — free, no key."""
    NAME = "kucoin"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        pair = f"{symbol.upper()}-USDT"
        url = "https://api.kucoin.com/api/v1/market/stats"
        params = {"symbol": pair}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            d = data.get("data", {})
            if not d or not d.get("last"):
                raise Exception(f"No data for {pair}")
            price = float(d["last"])
            volume = float(d.get("volValue", 0))
            change = float(d.get("changeRate", 0)) * 100
            return self._tick(symbol, price, volume, change, {
                "high_24h": float(d.get("high", 0)),
                "low_24h": float(d.get("low", 0)),
            })


class MEXCProvider(BaseProvider):
    """MEXC public ticker — free, no key."""
    NAME = "mexc"

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        pair = f"{symbol.upper()}USDT"
        url = "https://api.mexc.com/api/v3/ticker/24hr"
        params = {"symbol": pair}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            data = await resp.json()
            if "code" in data:
                raise Exception(data.get("msg", "API error"))
            price = float(data["lastPrice"])
            volume = float(data.get("quoteVolume", 0))
            change = float(data.get("priceChangePercent", 0))
            return self._tick(symbol, price, volume, change)


# ──────────────────────────────────────────────
# Known crypto symbols (for auto-classification)
# ──────────────────────────────────────────────

_KNOWN_CRYPTO = set(_CG_MAP.keys()) | {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "WBTC", "WETH", "STETH",
    "TON", "ICP", "HBAR", "VET", "ALGO", "EGLD", "SAND", "MANA",
    "AXS", "GALA", "ENJ", "THETA", "XTZ", "EOS", "AAVE", "MKR",
    "CRV", "COMP", "SNX", "SUSHI", "YFI", "1INCH", "BAL", "RUNE",
    "CAKE", "JOE", "GMX", "DYDX", "LDO", "RPL", "SSV", "BLUR",
    "APE", "FLOKI", "JASMY", "CHZ", "ENS", "GRT", "RNDR",
}


# ──────────────────────────────────────────────
# Multi-Source Data Provider (Orchestrator)
# ──────────────────────────────────────────────

class MultiSourceDataProvider:
    """
    Orchestrates all providers with circuit breaker cascade.
    For each symbol, tries providers in order until one succeeds.
    """

    def __init__(self):
        # Stock providers — order = priority
        self.stock_providers: List[BaseProvider] = [
            YahooFinanceChartProvider(),
            CNBCQuoteProvider(),
            AlphaVantageProvider(),
            PolygonProvider(),
            TwelveDataProvider(),
            FMPProvider(),
            TiingoProvider(),
        ]

        # Crypto providers — order = priority
        self.crypto_providers: List[BaseProvider] = [
            BinanceProvider(),
            CoinGeckoProvider(),
            CryptoCompareProvider(),
            KrakenProvider(),
            KuCoinProvider(),
            MEXCProvider(),
        ]

        # Log status
        active_stock = [p.NAME for p in self.stock_providers if p.enabled]
        active_crypto = [p.NAME for p in self.crypto_providers if p.enabled]
        disabled_stock = [f"{p.NAME} ({p.KEY_ENV})" for p in self.stock_providers if not p.enabled]

        logger.info(f"MultiSourceDataProvider initialized")
        logger.info(f"  Stock providers:  {', '.join(active_stock)} ({len(active_stock)} active)")
        logger.info(f"  Crypto providers: {', '.join(active_crypto)} ({len(active_crypto)} active)")
        if disabled_stock:
            logger.warning(f"  Disabled (no key): {', '.join(disabled_stock)}")

    def is_crypto(self, symbol: str) -> bool:
        return symbol.upper() in _KNOWN_CRYPTO

    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch a single symbol through the circuit breaker cascade."""
        symbol = symbol.upper()
        providers = self.crypto_providers if self.is_crypto(symbol) else self.stock_providers

        for provider in providers:
            if not provider.available:
                continue
            try:
                result = await provider.fetch(symbol)
                if result and result.get("price", 0) > 0:
                    provider.breaker.record_success()
                    return result
                else:
                    provider.breaker.record_failure()
            except Exception as e:
                provider.breaker.record_failure()
                logger.debug(f"  [{provider.NAME}] failed for {symbol}: {e}")

        logger.warning(f"All providers failed for {symbol}")
        return None

    async def fetch_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple symbols concurrently."""
        if not symbols:
            return []

        tasks = [self.fetch(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ticks = []
        success = 0
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Exception fetching {sym}: {result}")
            elif result:
                logger.info(f"✓ Fetched {sym} from {result.get('source', '?')}: ${result['price']:,.2f}")
                ticks.append(result)
                success += 1

        logger.info(f"Batch fetch: {success}/{len(symbols)} successful")
        return ticks

    async def close(self):
        """Close all provider sessions."""
        all_providers = self.stock_providers + self.crypto_providers
        for p in all_providers:
            await p.close()

    def get_provider_status(self) -> Dict[str, Any]:
        """Return status of all providers for monitoring."""
        status = {"stock": [], "crypto": []}
        for p in self.stock_providers:
            status["stock"].append({
                "name": p.NAME,
                "enabled": p.enabled,
                "state": p.breaker.state.value,
                "failures": p.breaker.failures,
                "requires_key": p.REQUIRES_KEY,
                "key_env": p.KEY_ENV if p.REQUIRES_KEY else None,
            })
        for p in self.crypto_providers:
            status["crypto"].append({
                "name": p.NAME,
                "enabled": p.enabled,
                "state": p.breaker.state.value,
                "failures": p.breaker.failures,
            })
        return status
