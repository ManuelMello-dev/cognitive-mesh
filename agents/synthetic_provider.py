"""
Synthetic Data Provider - Fallback for when external APIs fail
Generates realistic market-like data for testing and resilience
"""
import asyncio
import time
import random
import math
from typing import Dict, Any, List, Optional

class SyntheticMarketProvider:
    """
    Generates synthetic market data that mimics real market behavior.
    Used as a fallback when yfinance or other providers are unavailable.
    """
    
    def __init__(self):
        self.base_prices = {
            "BTC": 65000.0,
            "ETH": 3500.0,
            "SOL": 150.0,
            "AAPL": 180.0,
            "NVDA": 900.0,
            "TSLA": 250.0,
            "DBGI": 12.5,  # User's requested symbol
            "MSFT": 420.0,
            "GOOGL": 140.0,
            "META": 480.0
        }
        self.time_offset = 0
        
    async def fetch_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate a synthetic tick for a given symbol"""
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Simulate market oscillation with noise
        t = time.time() + self.time_offset
        oscillation = math.sin(t / 60) * 0.02  # 2% oscillation
        noise = random.gauss(0, 0.01)  # 1% noise
        
        price = base_price * (1 + oscillation + noise)
        volume = random.randint(100000, 5000000)
        
        tick = {
            "symbol": symbol,
            "source": "synthetic",
            "price": round(price, 2),
            "volume": volume,
            "volatility": abs(noise),
            "timestamp": t
        }
        
        return tick
    
    async def fetch_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch a batch of synthetic ticks"""
        tasks = [self.fetch_tick(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
