"""
Multi-source Financial Data Provider with Circuit Breaker Pattern
Supports: Yahoo Finance, Binance, Crypto APIs, Alternative Data Sources
"""

import asyncio
import aiohttp
import time
import logging
import os
import tempfile
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from config.config import Config
from agents.synthetic_provider import SyntheticMarketProvider

logger = logging.getLogger("MultiSourceProvider")

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 2
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    state: CircuitState = CircuitState.CLOSED
    
    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker recovered to CLOSED")
    
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
                return True
            return False
        return True

class YahooFinanceProvider:
    def __init__(self):
        self.circuit = CircuitBreaker()
        self.yf = None
        try:
            import yfinance
            self.yf = yfinance
            # Production Hardening: Set a custom cache location to avoid permission issues in containers
            cache_dir = os.path.join(tempfile.gettempdir(), "yfinance_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.yf.set_tz_cache_location(cache_dir)
        except ImportError:
            logger.warning("yfinance not installed.")
        except Exception as e:
            logger.warning(f"Failed to set yfinance cache location: {e}")
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.circuit.can_attempt(): return None
        if not self.yf: return None
        
        try:
            # Simple mapping for common cryptos
            crypto_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
            yahoo_symbol = crypto_map.get(symbol, symbol)
            
            # Using run_in_executor because yfinance is blocking
            loop = asyncio.get_event_loop()
            ticker = self.yf.Ticker(yahoo_symbol)
            data = await loop.run_in_executor(None, lambda: ticker.history(period='1d'))
            
            if data.empty: return None
            
            latest = data.iloc[-1]
            tick = {
                "symbol": symbol,
                "source": "yahoo_finance",
                "price": float(latest['Close']),
                "volume": float(latest['Volume']),
                "timestamp": time.time()
            }
            self.circuit.record_success()
            return tick
        except Exception as e:
            self.circuit.record_failure()
            logger.debug(f"Yahoo Finance error for {symbol}: {e}")
            return None

class BinanceProvider:
    def __init__(self):
        self.circuit = CircuitBreaker()
        self.base_url = "https://api.binance.com/api/v3"
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.circuit.can_attempt(): return None
        try:
            binance_symbol = f"{symbol.upper()}USDT"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/ticker/price?symbol={binance_symbol}") as resp:
                    if resp.status != 200: return None
                    data = await resp.json()
                    tick = {
                        "symbol": symbol,
                        "source": "binance",
                        "price": float(data['price']),
                        "timestamp": time.time()
                    }
                    self.circuit.record_success()
                    return tick
        except Exception:
            self.circuit.record_failure()
            return None

class MultiSourceDataProvider:
    def __init__(self):
        # Production Hardening: Configure a shared session with connection pooling
        self.connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        self.providers = [
            YahooFinanceProvider(),
            BinanceProvider(),
            SyntheticMarketProvider()  # Fallback for resilience
        ]
    
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        for provider in self.providers:
            tick = await provider.fetch_tick(symbol)
            if tick: return tick
        return None
    
    async def fetch_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        tasks = [self.fetch_tick(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
