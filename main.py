"""
Cognitive Mesh — Main Entry Point
==================================
Wires the CognitiveIntelligentSystem (7 engines) to real-time market data
providers and exposes the cognitive state via HTTP/GPT I/O.

Architecture:
  HTTP Server (port 8080)
    └─ Dashboard, GPT I/O, API endpoints
  CognitiveIntelligentSystem
    ├─ AbstractionEngine       — concept formation
    ├─ ReasoningEngine         — logical inference
    ├─ CrossDomainEngine       — knowledge transfer
    ├─ OpenEndedGoalSystem     — autonomous goals
    ├─ ContinuousLearningEngine — online learning
    ├─ SelfEvolvingSystem      — code evolution
    └─ AlwaysOnOrchestrator    — fault tolerance
  DistributedCognitiveCore
    └─ Async wrapper with DB persistence + cognitive loop thread
  MarketScanner
    └─ Autonomous discovery of trending tickers from free APIs
  MultiSourceDataProvider
    └─ 13 providers with circuit breakers (stocks + crypto)
"""

import os
import sys
import asyncio
import logging
import time
import signal
from typing import List, Dict, Any, Set

# ──────────────────────────────────────────────
# Configure logging to STDOUT (Railway classifies stderr as errors)
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("CognitiveMesh")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from core.distributed_core import DistributedCognitiveCore
from agents.market_data_providers import MultiSourceDataProvider
from agents.market_scanner import MarketScanner
from http_server import start_http_server

# Optional stores
try:
    from storage.postgres_store import PostgresStore
except ImportError:
    PostgresStore = None
try:
    from storage.milvus_store import MilvusStore
except ImportError:
    MilvusStore = None
try:
    from storage.redis_cache import RedisCache
except ImportError:
    RedisCache = None

# Optional networking
try:
    from shared.network_zeromq import ZMQNode, ZMQPubSub
except ImportError:
    ZMQNode = None
    ZMQPubSub = None


class CognitiveMeshOrchestrator:
    """
    Main orchestrator — wires market data into the CognitiveIntelligentSystem
    and runs the cognitive loop alongside the HTTP server.
    """

    def __init__(self):
        self.node_id = Config.NODE_ID
        self.crypto_symbols: Set[str] = set()
        self.stock_symbols: Set[str] = set()
        self.update_interval = Config.UPDATE_INTERVAL

        # Market scanner — autonomous asset discovery
        self.scanner = MarketScanner()

        # Data provider (13 sources, circuit breakers)
        self.data_provider = MultiSourceDataProvider()

        # Core cognitive system (wraps CognitiveIntelligentSystem)
        self.core = DistributedCognitiveCore(node_id=self.node_id)

        # Optional networking
        self.network = None
        self.pubsub = None
        if ZMQNode and ZMQPubSub:
            try:
                self.network = ZMQNode(identity=self.node_id)
                self.pubsub = ZMQPubSub(identity=self.node_id)
            except Exception as e:
                logger.warning(f"ZeroMQ init failed (non-critical): {e}")

        # Optional persistence
        self.postgres = None
        self.milvus = None
        self.redis = None

        self.running = False

    # ──────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────

    async def initialize(self):
        """Initialize all system components"""
        logger.info(f"Initializing Cognitive Mesh: {self.node_id}")

        # Start networking (optional)
        if self.network and self.pubsub:
            try:
                await self.network.start()
                await self.pubsub.start_publisher()
                await self.pubsub.start_subscriber(
                    ["concept", "rule", "transfer", "metrics", "goal"]
                )
            except Exception as e:
                logger.warning(f"Networking init error (non-critical): {e}")

        # Connect to databases if configured
        await self._init_databases()

        # Link databases to core
        self.core.postgres = self.postgres
        self.core.milvus = self.milvus
        self.core.redis = self.redis

        # Start the cognitive loop (background thread)
        self.core.start_cognitive_loop()

        logger.info("Cognitive Mesh initialized successfully")

    async def _init_databases(self):
        """Initialize database connections — all non-blocking with timeouts"""
        # PostgreSQL
        if Config.POSTGRES_URL and PostgresStore:
            try:
                self.postgres = PostgresStore(Config.POSTGRES_URL)
                await asyncio.wait_for(self.postgres.connect(), timeout=5.0)
                logger.info("Connected to PostgreSQL")
            except asyncio.TimeoutError:
                logger.warning("PostgreSQL connection timed out — skipping")
                self.postgres = None
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e} — skipping")
                self.postgres = None

        # Milvus — deferred to background
        if Config.MILVUS_HOST and MilvusStore:
            logger.info("Milvus configured — connecting in background...")
            asyncio.create_task(self._connect_milvus_background())
        else:
            logger.info("Milvus not configured — running without vector store")

        # Redis
        if Config.REDIS_URL and RedisCache:
            try:
                self.redis = RedisCache(Config.REDIS_URL)
                await asyncio.wait_for(self.redis.connect(), timeout=5.0)
                logger.info("Connected to Redis")
            except asyncio.TimeoutError:
                logger.warning("Redis connection timed out — skipping")
                self.redis = None
            except Exception as e:
                logger.warning(f"Redis connection failed: {e} — skipping")
                self.redis = None

    async def _connect_milvus_background(self):
        """Connect to Milvus in background — never blocks startup"""
        try:
            self.milvus = MilvusStore(Config.MILVUS_HOST)
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self._milvus_sync_connect),
                timeout=10.0,
            )
            if self.milvus and getattr(self.milvus, 'connected', False):
                self.core.milvus = self.milvus
                logger.info("Connected to Milvus (background)")
            else:
                logger.warning("Milvus connection failed — running without vector store")
                self.milvus = None
        except Exception as e:
            logger.warning(f"Milvus background connect error: {e}")
            self.milvus = None

    def _milvus_sync_connect(self):
        """Synchronous Milvus connection wrapper"""
        try:
            from pymilvus import connections
            connections.connect(
                "default",
                host=self.milvus.host,
                port=self.milvus.port,
                timeout=5,
            )
            self.milvus.connected = True
        except Exception as e:
            logger.warning(f"Milvus sync connect error: {e}")
            self.milvus.connected = False

    # ──────────────────────────────────────────
    # Main Run Loop
    # ──────────────────────────────────────────

    async def run(self):
        """Run the Cognitive Mesh with phased startup"""
        self.running = True

        try:
            # PHASE 1: Start HTTP server FIRST for Railway health checks
            logger.info("PHASE 1: Starting HTTP server...")
            http_task = asyncio.create_task(
                start_http_server(self.core, self.data_provider)
            )
            await asyncio.sleep(2)
            if http_task.done():
                try:
                    http_task.result()
                except Exception as e:
                    logger.error(f"HTTP server failed to start: {e}")
            else:
                logger.info("HTTP server task is running.")

            # PHASE 2: Initialize mesh (DB, networking, cognitive loop)
            logger.info("PHASE 2: Mesh initialization...")
            await self.initialize()

            # PHASE 3: Initial market scan — discover assets autonomously
            logger.info("PHASE 3: Autonomous market discovery...")
            await self._initial_market_scan()

            # PHASE 4: Start data collection and auxiliary loops
            logger.info("PHASE 4: Starting execution loops.")
            logger.info("=" * 60)
            logger.info("  COGNITIVE MESH IS LIVE")
            logger.info("  Engines: Abstraction, Reasoning, CrossDomain,")
            logger.info("           Goals, Learning, Evolution, Orchestrator")
            logger.info("  Data:    13 providers (stocks + crypto)")
            logger.info(f"  Crypto:  {len(self.crypto_symbols)} symbols discovered")
            logger.info(f"  Stocks:  {len(self.stock_symbols)} symbols discovered")
            logger.info("  GPT I/O: /api/chat, /api/ingest, /api/state")
            logger.info("=" * 60)

            loops = [
                self._data_collection_loop(),
                self._market_rescan_loop(),
                self._metrics_reporter_loop(),
                http_task,
            ]

            # Add optional networking loops
            if self.network:
                loops.append(self._gossip_loop())
                loops.append(self._network_listener_loop())
            if self.pubsub:
                loops.append(self._pubsub_listener_loop())

            await asyncio.gather(*loops, return_exceptions=True)

        except KeyboardInterrupt:
            logger.info("Shutting down due to user interrupt...")
        except Exception as e:
            logger.fatal(f"Unexpected error in main run loop: {e}", exc_info=True)
        finally:
            await self.shutdown()

    # ──────────────────────────────────────────
    # Market Discovery
    # ──────────────────────────────────────────

    async def _initial_market_scan(self):
        """Perform initial market scan to discover assets"""
        try:
            discovered = await self.scanner.scan()
            self.crypto_symbols.update(discovered.get("crypto", set()))
            self.stock_symbols.update(discovered.get("stocks", set()))

            logger.info(
                f"Initial scan complete: "
                f"{len(self.crypto_symbols)} crypto, "
                f"{len(self.stock_symbols)} stocks"
            )
        except Exception as e:
            logger.error(f"Initial market scan failed: {e}")

    async def _market_rescan_loop(self):
        """Periodically rescan for new trending assets"""
        while self.running:
            try:
                # Rescan every 5 minutes
                await asyncio.sleep(300)

                discovered = await self.scanner.scan()
                new_crypto = discovered.get("crypto", set())
                new_stocks = discovered.get("stocks", set())

                if new_crypto:
                    self.crypto_symbols.update(new_crypto)
                    logger.info(f"Rescan: added {len(new_crypto)} new crypto symbols")
                if new_stocks:
                    self.stock_symbols.update(new_stocks)
                    logger.info(f"Rescan: added {len(new_stocks)} new stock symbols")

            except Exception as e:
                logger.error(f"Market rescan error: {e}")
                await asyncio.sleep(60)

    # ──────────────────────────────────────────
    # Data Collection Loop
    # ──────────────────────────────────────────

    async def _data_collection_loop(self):
        """
        Continuously collect market data and feed it into the cognitive system.
        Each tick becomes an observation processed through all 7 engines.
        Fetches ALL discovered symbols every cycle for maximum data density.
        """
        logger.info("Starting data collection loop")

        while self.running:
            try:
                # Fetch ALL discovered symbols every cycle
                batch = sorted(self.crypto_symbols) + sorted(self.stock_symbols)

                if not batch:
                    logger.info("No symbols to fetch yet — waiting for market scan...")
                    await asyncio.sleep(10)
                    continue

                logger.info(f"Fetching batch [{len(batch)} symbols]: {batch[:10]}{'...' if len(batch) > 10 else ''}")
                ticks = await self.data_provider.fetch_batch(batch)

                # Feed each tick into the cognitive system
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

                    # Cache in Redis if available
                    if self.redis:
                        try:
                            await self.redis.cache_tick(symbol, tick)
                        except Exception:
                            pass

                if success_count > 0:
                    logger.info(f"Ingested {success_count}/{len(batch)} ticks into cognitive system")

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(10)

    # ──────────────────────────────────────────
    # Auxiliary Loops
    # ──────────────────────────────────────────

    async def _gossip_loop(self):
        """Periodically broadcast gossip state"""
        while self.running:
            try:
                state = self.core.get_state_summary()
                if self.network:
                    await self.network.broadcast_gossip(state)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
                await asyncio.sleep(15)

    async def _metrics_reporter_loop(self):
        """Periodically report system-wide metrics"""
        while self.running:
            try:
                metrics = self.core.get_metrics()
                if self.pubsub:
                    await self.pubsub.publish("metrics", metrics)

                # Log a summary periodically
                logger.info(
                    f"METRICS | PHI={metrics['global_coherence_phi']:.3f} "
                    f"SIGMA={metrics['noise_level_sigma']:.3f} "
                    f"Concepts={metrics['total_concepts']} "
                    f"Rules={metrics['total_rules']} "
                    f"Goals={metrics['total_goals']} "
                    f"Obs={metrics['total_observations']} "
                    f"Transfers={metrics['knowledge_transfers']} "
                    f"Crypto={len(self.crypto_symbols)} "
                    f"Stocks={len(self.stock_symbols)}"
                )

                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(10)

    async def _network_listener_loop(self):
        """Listen for incoming network messages"""
        while self.running:
            try:
                if self.network:
                    msg = await self.network.receive()
                    if msg:
                        await self.core.process_network_message(msg)
                    else:
                        await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in network listener: {e}")
                await asyncio.sleep(1)

    async def _pubsub_listener_loop(self):
        """Listen for pubsub messages"""
        while self.running:
            try:
                if self.pubsub:
                    msg = await self.pubsub.receive()
                    if msg:
                        await self.core.process_pubsub_message(msg)
                    else:
                        await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in pubsub listener: {e}")
                await asyncio.sleep(1)

    # ──────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────

    async def shutdown(self):
        """Shutdown all components gracefully"""
        self.running = False
        logger.info("Shutting down Cognitive Mesh...")

        # Stop cognitive loop
        self.core.stop_cognitive_loop()

        # Stop scanner
        try:
            await self.scanner.close()
        except Exception:
            pass

        # Stop networking
        tasks = []
        if self.network:
            tasks.append(self.network.stop())
        if self.pubsub:
            tasks.append(self.pubsub.stop())

        # Disconnect databases
        if self.postgres:
            tasks.append(self.postgres.disconnect())
        if self.milvus:
            tasks.append(self.milvus.disconnect())
        if self.redis:
            tasks.append(self.redis.disconnect())

        # Close data provider sessions
        try:
            await self.data_provider.close()
        except Exception:
            pass

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Cognitive Mesh shutdown complete")


async def main():
    """Main entry point"""
    orchestrator = CognitiveMeshOrchestrator()

    # Handle SIGTERM gracefully (Railway sends SIGTERM before stopping)
    loop = asyncio.get_event_loop()

    def handle_sigterm():
        logger.info("Received SIGTERM — initiating graceful shutdown...")
        orchestrator.running = False

    try:
        loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
        loop.add_signal_handler(signal.SIGINT, handle_sigterm)
    except NotImplementedError:
        pass

    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
