"""
Multi-Source Market Data Providers
Implements multiple data sources with automatic fallback for resilient data collection
"""
import os
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("MarketDataProviders")

class CoinGeckoProvider:
    """Free crypto data from CoinGecko - no API key required"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Symbol mapping for CoinGecko IDs
    SYMBOL_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "MATIC": "matic-network",
        "DOT": "polkadot",
        "AVAX": "avalanche-2"
    }
    
    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current price and volume for a crypto symbol"""
        try:
            coin_id = self.SYMBOL_MAP.get(symbol.upper())
            if not coin_id:
                return None
            
            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if coin_id in data:
                            return {
                                "symbol": symbol,
                                "price": data[coin_id].get("usd"),
                                "volume": data[coin_id].get("usd_24h_vol"),
                                "change_24h": data[coin_id].get("usd_24h_change"),
                                "source": "coingecko",
                                "timestamp": datetime.now().isoformat()
                            }
            return None
        except Exception as e:
            logger.error(f"CoinGecko fetch error for {symbol}: {e}")
            return None


class BinanceProvider:
    """Free crypto data from Binance - no API key required for public endpoints"""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current price and volume for a crypto symbol"""
        try:
            # Binance uses USDT pairs
            pair = f"{symbol.upper()}USDT"
            
            # Get 24hr ticker data
            url = f"{self.BASE_URL}/ticker/24hr"
            params = {"symbol": pair}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "symbol": symbol,
                            "price": float(data.get("lastPrice", 0)),
                            "volume": float(data.get("volume", 0)),
                            "change_24h": float(data.get("priceChangePercent", 0)),
                            "source": "binance",
                            "timestamp": datetime.now().isoformat()
                        }
            return None
        except Exception as e:
            logger.error(f"Binance fetch error for {symbol}: {e}")
            return None


class AlphaVantageProvider:
    """Stock data from Alpha Vantage - requires API key (free tier: 25 calls/day)"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("AlphaVantage disabled: ALPHA_VANTAGE_API_KEY not set")
    
    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current price and volume for a stock symbol"""
        if not self.enabled:
            return None
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper(),
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        quote = data.get("Global Quote", {})
                        if quote:
                            return {
                                "symbol": symbol,
                                "price": float(quote.get("05. price", 0)),
                                "volume": float(quote.get("06. volume", 0)),
                                "change_percent": float(quote.get("10. change percent", "0").replace("%", "")),
                                "source": "alphavantage",
                                "timestamp": datetime.now().isoformat()
                            }
            return None
        except Exception as e:
            logger.error(f"AlphaVantage fetch error for {symbol}: {e}")
            return None


class PolygonProvider:
    """Stock data from Polygon.io - requires API key (free tier available)"""
    
    BASE_URL = "https://api.polygon.io/v2"
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("Polygon disabled: POLYGON_API_KEY not set")
    
    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current price and volume for a stock symbol"""
        if not self.enabled:
            return None
        
        try:
            url = f"{self.BASE_URL}/aggs/ticker/{symbol.upper()}/prev"
            params = {"apiKey": self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        if results:
                            r = results[0]
                            return {
                                "symbol": symbol,
                                "price": float(r.get("c", 0)),  # close price
                                "volume": float(r.get("v", 0)),
                                "open": float(r.get("o", 0)),
                                "high": float(r.get("h", 0)),
                                "low": float(r.get("l", 0)),
                                "source": "polygon",
                                "timestamp": datetime.now().isoformat()
                            }
            return None
        except Exception as e:
            logger.error(f"Polygon fetch error for {symbol}: {e}")
            return None


class TwelveDataProvider:
    """Stock data from Twelve Data - requires API key (free tier: 800 calls/day)"""
    
    BASE_URL = "https://api.twelvedata.com"
    
    def __init__(self):
        self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("TwelveData disabled: TWELVE_DATA_API_KEY not set")
    
    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current price and volume for a stock symbol"""
        if not self.enabled:
            return None
        
        try:
            url = f"{self.BASE_URL}/quote"
            params = {
                "symbol": symbol.upper(),
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "close" in data:
                            return {
                                "symbol": symbol,
                                "price": float(data.get("close", 0)),
                                "volume": float(data.get("volume", 0)),
                                "change_percent": float(data.get("percent_change", 0)),
                                "source": "twelvedata",
                                "timestamp": datetime.now().isoformat()
                            }
            return None
        except Exception as e:
            logger.error(f"TwelveData fetch error for {symbol}: {e}")
            return None


class MultiSourceDataProvider:
    """
    Orchestrates multiple data providers with automatic fallback.
    Tries providers in priority order until one succeeds.
    """
    
    def __init__(self):
        # Crypto providers (no API key required)
        self.crypto_providers = [
            CoinGeckoProvider(),
            BinanceProvider()
        ]
        
        # Stock providers (API keys optional but recommended)
        self.stock_providers = [
            AlphaVantageProvider(),
            PolygonProvider(),
            TwelveDataProvider()
        ]
        
        self.crypto_symbols = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "MATIC", "DOT", "AVAX"}
        
        logger.info("MultiSourceDataProvider initialized")
        logger.info(f"Crypto providers: {len(self.crypto_providers)} active")
        logger.info(f"Stock providers: {sum(1 for p in self.stock_providers if getattr(p, 'enabled', True))} enabled")
    
    def is_crypto(self, symbol: str) -> bool:
        """Determine if a symbol is crypto or stock"""
        return symbol.upper() in self.crypto_symbols
    
    async def fetch(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data for a symbol using appropriate providers with fallback.
        Returns the first successful result.
        """
        providers = self.crypto_providers if self.is_crypto(symbol) else self.stock_providers
        
        for provider in providers:
            try:
                result = await provider.fetch(symbol)
                if result and result.get("price"):
                    logger.info(f"✓ Fetched {symbol} from {result.get('source')}: ${result.get('price')}")
                    return result
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed for {symbol}: {e}")
                continue
        
        logger.warning(f"All providers failed for {symbol}")
        return None
    
    async def fetch_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch data for multiple symbols concurrently"""
        tasks = [self.fetch(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None and exceptions
        valid_results = [r for r in results if r and isinstance(r, dict)]
        
        logger.info(f"Batch fetch: {len(valid_results)}/{len(symbols)} successful")
        return valid_results
