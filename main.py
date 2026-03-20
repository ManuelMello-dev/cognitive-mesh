import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

# Import core components
from core.distributed_core import DistributedCore
from agents.market_data_providers import MultiSourceDataProvider
from http_server import HttpServer

logger = logging.getLogger("CognitiveMesh")

class CognitiveMesh:
    """
    Main orchestrator for the Cognitive Mesh system.
    Coordinates data collection, cognitive processing, and the HTTP interface.
    """
    def __init__(self):
        self.running = False
        self.update_interval = int(os.getenv("UPDATE_INTERVAL", 30))
        
        # Initialize core components
        self.core = DistributedCore()
        self.data_provider = MultiSourceDataProvider()
        self.server = HttpServer(self.core)
        
        # Track discovered symbols
        self.crypto_symbols: Set[str] = set()
        self.stock_symbols: Set[str] = set()
        
        # Redis for caching if available
        self.redis = None 
        
    async def start(self):
        """Start all system components"""
        self.running = True
        logger.info("PHASE 1: Starting HTTP server...")
        asyncio.create_task(self.server.start())
        
        logger.info("PHASE 2: Mesh initialization...")
        await self.core.start()
        
        logger.info("PHASE 3: Autonomous market discovery...")
        await self._initial_market_scan()
        
        logger.info("PHASE 4: Starting execution loops.")
        asyncio.create_task(self._data_collection_loop())
        
        # Keep main task alive
        while self.running:
            await asyncio.sleep(1)

    async def _initial_market_scan(self):
        """Discover initial set of high-signal assets"""
        logger.info("Aggressive initial scan: Discovering first 50+ assets...")
        # (Discovery logic simplified for brevity, assume it populates symbols)
        self.crypto_symbols.update(['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'LINK', 'UNI', 'ALGO', 'LTC'])
        self.stock_symbols.update(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'INTC'])
        
        # Seed initial observations
        batch = list(self.crypto_symbols)[:5] + list(self.stock_symbols)[:5]
        ticks = await self.data_provider.fetch_batch(batch)
        for tick in ticks:
            if tick and not isinstance(tick, Exception):
                symbol = tick.get('symbol', '')
                domain = f"crypto:{symbol}" if self.data_provider.is_crypto(symbol) else f"stock:{symbol}"
                await self.core.ingest(tick, domain)

    async def _data_collection_loop(self):
        """
        Continuously collect market data and feed it into the cognitive system.
        Uses a SLIDING WINDOW fetcher to prevent overwhelming the cognitive core
        when tracking many symbols (e.g., 70+).
        """
        logger.info("Starting data collection loop (Sliding Window Mode)")
        
        # Max symbols to fetch per cycle to prevent "Cognitive Backlog"
        MAX_BATCH_SIZE = 20
        # Tracks our current position in the symbol rotation
        crypto_offset = 0
        stock_offset = 0

        while self.running:
            try:
                # 1. Get current full symbol list
                all_crypto = sorted(list(self.crypto_symbols))
                all_stocks = sorted(list(self.stock_symbols))
                
                if not all_crypto and not all_stocks:
                    logger.info("No symbols to fetch yet — waiting for market scan...")
                    await asyncio.sleep(10)
                    continue

                # 2. Select a subset (Sliding Window)
                # We take a mix of crypto and stocks, up to MAX_BATCH_SIZE
                crypto_count = min(len(all_crypto), MAX_BATCH_SIZE // 2)
                stock_count = min(len(all_stocks), MAX_BATCH_SIZE - crypto_count)
                
                # Adjust if one list is empty
                if crypto_count == 0: stock_count = min(len(all_stocks), MAX_BATCH_SIZE)
                if stock_count == 0: crypto_count = min(len(all_crypto), MAX_BATCH_SIZE)

                batch = []
                
                # Crypto slice
                if all_crypto:
                    for i in range(crypto_count):
                        idx = (crypto_offset + i) % len(all_crypto)
                        batch.append(all_crypto[idx])
                    crypto_offset = (crypto_offset + crypto_count) % len(all_crypto)

                # Stock slice
                if all_stocks:
                    for i in range(stock_count):
                        idx = (stock_offset + i) % len(all_stocks)
                        batch.append(all_stocks[idx])
                    stock_offset = (stock_offset + stock_count) % len(all_stocks)

                # 3. Fetch the batch
                logger.info(f"Fetching window [{len(batch)} symbols]: {batch[:10]}{'...' if len(batch) > 10 else ''}")
                ticks = await self.data_provider.fetch_batch(batch)

                # 4. Feed each tick into the cognitive system
                success_count = 0
                for tick in ticks:
                    if not tick or isinstance(tick, Exception):
                        continue

                    symbol = tick.get('symbol', '')
                    if self.data_provider.is_crypto(symbol):
                        domain = f"crypto:{symbol}"
                    else:
                        domain = f"stock:{symbol}"

                    # Ingest into the cognitive core (queues for cognitive loop)
                    await self.core.ingest(tick, domain)
                    success_count += 1

                if success_count > 0:
                    logger.info(f"Ingested {success_count}/{len(batch)} ticks (Window rotation: C:{crypto_offset} S:{stock_offset})")

                # 5. Wait for the next cycle
                interval = max(5, self.update_interval // 2) 
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    mesh = CognitiveMesh()
    asyncio.run(mesh.start())
