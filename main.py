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
import threading
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

# ── Per-module log level overrides ────────────────────────────────────────────
# Railway hard-caps at 500 log lines/sec. These internal modules are very noisy
# at INFO level (they log on every observation, every concept formed, every
# ZeroMQ publish). Raising them to WARNING keeps the important signal visible
# while eliminating the flood. CognitiveMesh (top-level) stays at INFO.
_QUIET_MODULES = [
    "abstraction_engine",       # logs every concept formation
    "reasoning_engine",         # logs every rule inference
    "continuous_learning_engine",
    "cognitive_intelligent_system",
    "cross_domain_engine",
    "always_on_orchestrator",
    "ZeroMQNetwork",            # logs every ZMQ send/publish
    "gossip",
    "gossip_amfg",
    "MarketDataProviders",      # keep warnings (circuit breaker trips) but not debug
    "DistributedCore",          # very verbose at INFO during state save/load
    "self_writing_engine",
    "prediction_validation_engine",
    "goal_formation_system",
]
for _mod in _QUIET_MODULES:
    logging.getLogger(_mod).setLevel(logging.WARNING)
# MarketDataProviders: keep WARNING (circuit breaker trips, rate limits) but silence debug
logging.getLogger("MarketDataProviders").setLevel(logging.WARNING)
# ──────────────────────────────────────────────────────────────────────────────

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
        self.core.data_provider = self.data_provider

        # Load cognitive state from persistence
        await self.core.load_state()

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

    def _start_http_thread(self):
        """
        Run the aiohttp HTTP server in a DEDICATED THREAD with its own event loop.

        ROOT CAUSE OF 499 TIMEOUTS:
        The data collection loop makes 10 concurrent aiohttp requests per cycle
        (12s timeout each, BATCH_CONCURRENCY=10). When these run in the SAME
        event loop as the HTTP server, the event loop is saturated and cannot
        service incoming dashboard requests — causing 2-9s response times and
        Railway 499 client-abort timeouts.

        THE FIX:
        By giving the HTTP server its own dedicated event loop in a separate
        thread, it is completely isolated from data collection I/O. The HTTP
        server can always respond in <50ms regardless of what the data
        collection loop is doing.
        """
        http_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(http_loop)
        try:
            logger.info("HTTP server thread: starting aiohttp on dedicated event loop")
            http_loop.run_until_complete(
                start_http_server(self.core, self.data_provider)
            )
        except Exception as e:
            logger.error(f"HTTP server thread error: {e}", exc_info=True)
        finally:
            try:
                http_loop.close()
            except Exception:
                pass

    async def run(self):
        """Run the Cognitive Mesh with phased startup"""
        self.running = True

        try:
            # PHASE 1: Start HTTP server in its OWN DEDICATED THREAD
            # This isolates the HTTP event loop from the data collection event loop,
            # eliminating 499 timeouts caused by event loop saturation from concurrent
            # aiohttp fetch requests (BATCH_CONCURRENCY=10, timeout=12s each).
            logger.info("PHASE 1: Starting HTTP server (dedicated thread)...")
            http_thread = threading.Thread(
                target=self._start_http_thread,
                name="http-server",
                daemon=True,  # Dies automatically when main process exits
            )
            http_thread.start()
            await asyncio.sleep(2)  # Allow aiohttp to bind to port
            if http_thread.is_alive():
                logger.info("HTTP server task is running.")
            else:
                logger.error("HTTP server thread died unexpectedly")

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
        """Perform aggressive initial market scan to seed the mesh immediately"""
        try:
            logger.info("Aggressive initial scan: Discovering first 50+ assets...")
            discovered = await self.scanner.scan(min_price=Config.MIN_SCAN_PRICE, max_price=Config.MAX_SCAN_PRICE)
            self.crypto_symbols.update(discovered.get("crypto", set()))
            self.stock_symbols.update(discovered.get("stocks", set()))

            # Immediate first data collection pass to trigger consciousness
            if self.crypto_symbols or self.stock_symbols:
                logger.info("Seeding mesh with first observations...")
                await self._collect_all_data()

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

                discovered = await self.scanner.scan(min_price=Config.MIN_SCAN_PRICE, max_price=Config.MAX_SCAN_PRICE)
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

    async def _collect_all_data(self):
        """One-time collection of all discovered symbols"""
        batch = sorted(self.crypto_symbols) + sorted(self.stock_symbols)
        if not batch:
            return
            
        logger.info(f"Seeding batch [{len(batch)} symbols]...")
        ticks = await self.data_provider.fetch_batch(batch)
        
        success_count = 0
        for tick in ticks:
            if not tick or isinstance(tick, Exception):
                continue
            symbol = tick.get('symbol', '')
            domain = f"crypto:{symbol}" if self.data_provider.is_crypto(symbol) else f"stock:{symbol}"
            await self.core.ingest(tick, domain)
            success_count += 1
            
        if success_count > 0:
            logger.info(f"Seeding complete: Ingested {success_count} observations")

    async def _data_collection_loop(self):
        """
        Continuously collect market data and feed it into the cognitive system.
        Uses a SLIDING WINDOW fetcher to prevent overwhelming the cognitive core
        when tracking many symbols (e.g., 70+).
        """
        logger.info("Starting data collection loop (Sliding Window Mode)")
        # Max symbols to fetch per cycle to prevent cognitive backlog
        MAX_BATCH_SIZE = 20
        crypto_offset = 0
        stock_offset = 0
        while self.running:
            try:
                # Get current full symbol list
                all_crypto = sorted(list(self.crypto_symbols))
                all_stocks = sorted(list(self.stock_symbols))
                if not all_crypto and not all_stocks:
                    logger.info("No symbols to fetch yet — waiting for market scan...")
                    await asyncio.sleep(10)
                    continue
                # Select a sliding window subset
                crypto_count = min(len(all_crypto), MAX_BATCH_SIZE // 2)
                stock_count = min(len(all_stocks), MAX_BATCH_SIZE - crypto_count)
                if crypto_count == 0: stock_count = min(len(all_stocks), MAX_BATCH_SIZE)
                if stock_count == 0: crypto_count = min(len(all_crypto), MAX_BATCH_SIZE)
                batch = []
                if all_crypto:
                    for i in range(crypto_count):
                        idx = (crypto_offset + i) % len(all_crypto)
                        batch.append(all_crypto[idx])
                    crypto_offset = (crypto_offset + crypto_count) % len(all_crypto)
                if all_stocks:
                    for i in range(stock_count):
                        idx = (stock_offset + i) % len(all_stocks)
                        batch.append(all_stocks[idx])
                    stock_offset = (stock_offset + stock_count) % len(all_stocks)
                logger.info(f"Fetching window [{len(batch)} symbols]: {batch[:10]}{'...' if len(batch) > 10 else ''}")
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
                    logger.info(f"Ingested {success_count}/{len(batch)} ticks (Window rotation: C:{crypto_offset} S:{stock_offset})")

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(10)

    # ──────────────────────────────────────────
    # Auxiliary Loops
    # ──────────────────────────────────────────

    async def _gossip_loop(self):
        """Periodically broadcast gossip state — reads from cache to avoid lock contention"""
        while self.running:
            try:
                # Use cache to avoid acquiring self._lock in the aiohttp event loop
                cached = self.core.get_cached_state()
                state = {
                    "node_id": cached.get('node_id', self.core.node_id),
                    "phi": cached.get('metrics', {}).get('global_coherence_phi', 0),
                    "sigma": cached.get('metrics', {}).get('noise_level_sigma', 0),
                    "metrics": cached.get('metrics', {}),
                    "timestamp": time.time()
                }
                if self.network:
                    await self.network.broadcast_gossip(state)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
                await asyncio.sleep(15)

    async def _metrics_reporter_loop(self):
        """Periodically report system-wide metrics — reads from cache to avoid lock contention"""
        while self.running:
            try:
                # Use cache to avoid acquiring self._lock in the aiohttp event loop
                cached = self.core.get_cached_state()
                metrics = cached.get('metrics', {})
                if self.pubsub and metrics:
                    await self.pubsub.publish("metrics", metrics)

                # Log a summary periodically
                logger.info(
                    f"METRICS | PHI={metrics.get('global_coherence_phi', 0):.3f} "
                    f"SIGMA={metrics.get('noise_level_sigma', 0):.3f} "
                    f"Concepts={metrics.get('total_concepts', 0)} "
                    f"Rules={metrics.get('total_rules', 0)} "
                    f"Goals={metrics.get('total_goals', 0)} "
                    f"Obs={metrics.get('total_observations', 0)} "
                    f"Transfers={metrics.get('knowledge_transfers', 0)} "
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

        # Save cognitive state to persistence
        logger.info("Saving cognitive state to persistence...")
        try:
            await self.core.save_state()
        except Exception as e:
            logger.error(f"Failed to save cognitive state: {e}")

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
