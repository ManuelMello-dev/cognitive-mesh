"""
Multi-source Financial Data Provider with Circuit Breaker Pattern
Supports: Yahoo Finance, Binance, Crypto APIs, Alternative Data Sources
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import os

logger = logging.getLogger("MultiSourceProvider")


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 2
    
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    state: CircuitState = field(default=CircuitState.CLOSED)
    
    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker recovered to CLOSED")
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
        return True  # HALF_OPEN


class YahooFinanceProvider:
    """Free stock data from Yahoo Finance via yfinance"""
    
    def __init__(self):
        self.circuit = CircuitBreaker()
        try:
            import yfinance
            self.yf = yfinance
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            self.yf = None
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.circuit.can_attempt():
            raise Exception(f"Circuit breaker OPEN for Yahoo Finance")
        
        if not self.yf:
            raise Exception("yfinance not available")
        
        try:
            # Map common crypto symbols to Yahoo format (SYMBOL-USD)
            crypto_symbols = [
                'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'LINK', 
                'BNB', 'TRX', 'BCH', 'XLM', 'LTC', 'SHIB', 'UNI', 'NEAR', 'ICP', 
                'HYPE', 'TRUMP', 'PUMP', 'PIPPIN', 'SUIS', 'TON', 'PEPE'
            ]
            yahoo_symbol = f"{symbol}-USD" if symbol in crypto_symbols else symbol
            
            ticker = self.yf.Ticker(yahoo_symbol)
            data = ticker.history(period='1d')
            
            if data.empty:
                # Secondary fallback for tokens that might be listed differently
                if symbol == "SOL":
                    ticker = self.yf.Ticker("SOL1-USD")
                    data = ticker.history(period="1d")
                
                if data.empty:
                    raise Exception(f"No data for {yahoo_symbol}")
            
            latest = data.iloc[-1]
            tick = {
                "symbol": symbol,
                "source": "yahoo_finance",
                "price": float(latest['Close']),
                "volume": float(latest['Volume']),
                "high": float(latest['High']),
                "low": float(latest['Low']),
                "open": float(latest['Open']),
                "timestamp": time.time()
            }
            self.circuit.record_success()
            return tick
        except Exception as e:
            self.circuit.record_failure()
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise


class BinanceProvider:
    """Real-time crypto data from Binance public API (no key required)"""
    
    def __init__(self):
        self.circuit = CircuitBreaker()
        self.base_url = "https://api.binance.com/api/v3"
        self.alt_url = "https://api.binance.us/api/v3"
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.circuit.can_attempt():
            raise Exception(f"Circuit breaker OPEN for Binance")
        
        try:
            # Convert symbol format (e.g., BTC -> BTCUSDT)
            binance_symbol = f"{symbol.upper()}USDT"
            
            async with aiohttp.ClientSession() as session:
                # Try primary, fallback to US endpoint if blocked (451 error)
                current_url = self.base_url
                
                async def get_data(url):
                    async with session.get(f"{url}/ticker/price?symbol={binance_symbol}") as p_resp:
                        if p_resp.status == 451: return "RETRY_US"
                        if p_resp.status != 200: raise Exception(f"Binance API error: {p_resp.status}")
                        p_data = await p_resp.json()
                    
                    async with session.get(f"{url}/ticker/24hr?symbol={binance_symbol}") as s_resp:
                        if s_resp.status != 200: raise Exception(f"Binance API error: {s_resp.status}")
                        s_data = await s_resp.json()
                    
                    return p_data, s_data

                res = await get_data(self.base_url)
                if res == "RETRY_US":
                    logger.info(f"Binance primary blocked for {symbol}, trying US fallback")
                    res = await get_data(self.alt_url)
                
                price_data, stats = res
                
                tick = {
                    "symbol": symbol,
                    "source": "binance",
                    "price": float(price_data['price']),
                    "volume": float(stats['volume']),
                    "high": float(stats['highPrice']),
                    "low": float(stats['lowPrice']),
                    "open": float(stats['openPrice']),
                    "timestamp": time.time()
                }
                self.circuit.record_success()
                return tick
        except Exception as e:
            self.circuit.record_failure()
            logger.error(f"Binance error for {symbol}: {e}")
            raise


class AlphaVantageProvider:
    """Stock data from Alpha Vantage (requires API key)"""
    
    def __init__(self, api_key: str = None):
        self.circuit = CircuitBreaker()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.circuit.can_attempt():
            raise Exception(f"Circuit breaker OPEN for Alpha Vantage")
        
        if not self.api_key:
            raise Exception("Alpha Vantage API key not configured")
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as resp:
                    if resp.status != 200:
                        raise Exception(f"Alpha Vantage API error: {resp.status}")
                    data = await resp.json()
                
                if "Global Quote" not in data or not data["Global Quote"]:
                    raise Exception(f"No data for {symbol}")
                
                quote = data["Global Quote"]
                tick = {
                    "symbol": symbol,
                    "source": "alpha_vantage",
                    "price": float(quote.get('05. price', 0)),
                    "volume": float(quote.get('06. volume', 0)),
                    "high": float(quote.get('03. high', 0)),
                    "low": float(quote.get('04. low', 0)),
                    "open": float(quote.get('02. open', 0)),
                    "timestamp": time.time()
                }
                self.circuit.record_success()
                return tick
        except Exception as e:
            self.circuit.record_failure()
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            raise


class PolygonIOProvider:
    """Stock data from Polygon.io (requires API key)"""
    
    def __init__(self, api_key: str = None):
        self.circuit = CircuitBreaker()
        self.api_key = api_key or os.getenv("POLYGON_IO_API_KEY")
        self.base_url = "https://api.polygon.io/v1"
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.circuit.can_attempt():
            raise Exception(f"Circuit breaker OPEN for Polygon.io")
        
        if not self.api_key:
            raise Exception("Polygon.io API key not configured")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/last/quote/{symbol}",
                    params={"apiKey": self.api_key}
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Polygon.io API error: {resp.status}")
                    data = await resp.json()
                
                if "last" not in data:
                    raise Exception(f"No data for {symbol}")
                
                last = data["last"]
                tick = {
                    "symbol": symbol,
                    "source": "polygon_io",
                    "price": float(last.get('last', 0)),
                    "bid": float(last.get('bid', 0)),
                    "ask": float(last.get('ask', 0)),
                    "timestamp": time.time()
                }
                self.circuit.record_success()
                return tick
        except Exception as e:
            self.circuit.record_failure()
            logger.error(f"Polygon.io error for {symbol}: {e}")
            raise


class AlpacaProvider:
    """Stock data from Alpaca (requires API key)"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.circuit = CircuitBreaker()
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = "https://data.alpaca.markets/v1beta1"
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.circuit.can_attempt():
            raise Exception(f"Circuit breaker OPEN for Alpaca")
        
        if not self.api_key or not self.secret_key:
            raise Exception("Alpaca API credentials not configured")
        
        try:
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/latest/quotes?symbols={symbol}",
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Alpaca API error: {resp.status}")
                    data = await resp.json()
                
                if "quotes" not in data or symbol not in data["quotes"]:
                    raise Exception(f"No data for {symbol}")
                
                quote = data["quotes"][symbol]
                tick = {
                    "symbol": symbol,
                    "source": "alpaca",
                    "bid": float(quote.get('bp', 0)),
                    "ask": float(quote.get('ap', 0)),
                    "bid_size": float(quote.get('bs', 0)),
                    "ask_size": float(quote.get('as', 0)),
                    "timestamp": time.time()
                }
                self.circuit.record_success()
                return tick
        except Exception as e:
            self.circuit.record_failure()
            logger.error(f"Alpaca error for {symbol}: {e}")
            raise


class MultiSourceDataProvider:
    """Aggregates data from multiple sources with fallback strategy"""
    
    def __init__(self):
        self.providers = {
            "yahoo": YahooFinanceProvider(),
            "binance": BinanceProvider(),
            "alpha_vantage": AlphaVantageProvider(),
            "polygon_io": PolygonIOProvider(),
            "alpaca": AlpacaProvider(),
        }
        self.provider_priority = ["yahoo", "binance", "alpha_vantage", "polygon_io", "alpaca"]
    
    async def fetch_tick(self, symbol: str, source: str = None) -> Optional[Dict[str, Any]]:
        """
        Fetch tick data with fallback strategy.
        If source is specified, try that first. Otherwise, try all in priority order.
        """
        if source and source in self.providers:
            try:
                return await self.providers[source].fetch_tick(symbol)
            except Exception as e:
                logger.warning(f"Failed to fetch from {source}: {e}")
        
        # Fallback to priority order
        for provider_name in self.provider_priority:
            try:
                return await self.providers[provider_name].fetch_tick(symbol)
            except Exception as e:
                logger.debug(f"Provider {provider_name} failed: {e}")
                continue
        
        # HARDENED PRODUCTION FALLBACK
        # We strictly avoid simulations. If all primary providers fail, we log a critical 
        # failure for the asset. This ensures the mesh only processes real market signals.
        logger.error(f"CRITICAL: All data providers failed for {symbol}. No real-world pulse detected.")
        return None
    
    async def fetch_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch data for multiple symbols concurrently"""
        tasks = [self.fetch_tick(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out exceptions AND None values (which represent real-world data failures)
        return [r for r in results if r is not None and not isinstance(r, Exception)]
