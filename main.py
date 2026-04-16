"""
Cognitive Mesh — Main Entry Point
==================================
Wires the CognitiveIntelligentSystem (7 engines) to real-time data
sources and exposes the cognitive state via HTTP/GPT I/O.

Architecture:
  HTTP Server (port 8080)
    └─ Dashboard, GPT I/O, API endpoints
  CognitiveIntelligentSystem
    ├─ AbstractionEngine       — concept formation
    ├─ ReasoningEngine         — logical inference
    ├─ CrossDomainEngine       — knowledge transfer
    ├─ OpenEndedGoalSystem     — autonomous goals
    ├─ ContinuousLearningEngine — online learning
    ├─ SelfEvolvingSystem      — code evolution (ACTIVE)
    └─ AlwaysOnOrchestrator    — fault tolerance
  DistributedCognitiveCore
    └─ Async wrapper with DB persistence + cognitive loop thread
  DataPlugin system (domain-agnostic)
    └─ MarketPlugin (financial) — optional, loaded if enabled
    └─ Any other plugin can be added without touching the core

The core is completely domain-agnostic.  All domain-specific logic
lives in DataPlugin subclasses under agents/plugins/.  The orchestrator
simply calls plugin.fetch() each cycle and ingests the returned
observations into the core.
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
# Configure logging to STDOUT
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("CognitiveMesh")

# Per-module log level overrides (noisy internal modules)
# All internal engines are set to WARNING — only errors surface.
# The CognitiveMesh and HttpServer loggers remain at INFO for operational visibility.
_QUIET_MODULES = [
    # Cognitive engines
    "abstraction_engine",
    "reasoning_engine",
    "continuous_learning_engine",
    "cognitive_intelligent_system",
    "cross_domain_engine",
    "always_on_orchestrator",
    "goal_formation_system",
    "prediction_validation_engine",
    "self_writing_engine",
    "EvoEngine",
    # Networking / gossip
    "ZeroMQNetwork",
    "NetworkLayer",
    "gossip",
    "gossip_amfg",
    "AMFGGossip",
    "CognitiveGossip",
    # Storage
    "DistributedCore",
    "PostgresStore",
    "RedisCache",
    "MilvusStore",
    "StorageConnectors",
    # Market / data providers
    "MarketDataProviders",
    "MarketScanner",
    "DataProvider",
    # Agents
    "AutonomousReasoner",
    "NativeInterpreter",
    "PursuitAgent",
]
for _mod in _QUIET_MODULES:
    logging.getLogger(_mod).setLevel(logging.WARNING)

# Rate-limit the top-level CognitiveMesh logger to avoid Railway 500 log/s cap.
# We use a simple dedup filter: identical consecutive messages are suppressed.
class _DedupeFilter(logging.Filter):
    """Suppress consecutive identical log messages to prevent log floods."""
    def __init__(self):
        super().__init__()
        self._last_msg: str = ""
        self._repeat_count: int = 0

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg == self._last_msg:
            self._repeat_count += 1
            # Allow through every 100th repeat so we know it's still happening
            return self._repeat_count % 100 == 0
        self._last_msg = msg
        self._repeat_count = 0
        return True

logging.getLogger("CognitiveMesh").addFilter(_DedupeFilter())
logging.getLogger("HttpServer").addFilter(_DedupeFilter())

# ──────────────────────────────────────────────────────────────────────────────

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from core.distributed_core import DistributedCognitiveCore
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

# Optional fault-tolerance watchdog
try:
    from always_on_orchestrator import AlwaysOnOrchestrator
except ImportError:
    AlwaysOnOrchestrator = None


# ──────────────────────────────────────────────────────────────────────────────
# DataPlugin base class
# ──────────────────────────────────────────────────────────────────────────────

class DataPlugin:
    """
    Base class for all data source plugins.

    A plugin is responsible for:
      1. Discovering entities to observe (optional, can be static)
      2. Fetching observations each cycle
      3. Returning a list of (observation_dict, domain_string) tuples

    The observation_dict MUST contain at least:
      - 'value'     : float  — the primary observable
      - 'entity_id' : str    — unique identifier for this stream

    Optional fields:
      - 'secondary_value' : float  — e.g. volume, confidence, intensity
      - 'timestamp'       : float  — unix timestamp (defaults to now)
      - any domain-specific fields (will be stored as metadata)
    """

    name: str = "base"

    async def initialize(self) -> None:
        """Called once at startup."""
        pass

    async def fetch(self) -> List[tuple]:
        """
        Fetch a batch of observations.
        Returns: list of (observation_dict, domain_string)
        """
        return []

    async def close(self) -> None:
        """Called at shutdown."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
# MarketPlugin — financial data (optional, backward-compatible)
# ──────────────────────────────────────────────────────────────────────────────

class MarketPlugin(DataPlugin):
    """
    Financial market data plugin.
    Wraps the existing MultiSourceDataProvider and MarketScanner.
    Translates financial ticks into the generic observation schema.
    """

    name = "market"
    STATE_CACHE_KEY = "market_plugin_state"

    def __init__(self):
        self._provider = None
        self._scanner = None
        self._crypto_symbols: Set[str] = set()
        self._stock_symbols: Set[str] = set()
        self._crypto_offset = 0
        self._stock_offset = 0
        self._last_discovery_ts = 0.0
        self._last_fetch_ts = 0.0
        self._last_successful_fetch_ts = 0.0
        self._total_observations_emitted = 0
        self.MAX_BATCH_SIZE = 5  # reduced from 20 — free APIs rate-limit at ~10 req/min

    async def initialize(self) -> None:
        try:
            from agents.market_data_providers import MultiSourceDataProvider
            from agents.market_scanner import MarketScanner
            self._provider = MultiSourceDataProvider()
            self._scanner = MarketScanner()
            logger.info("MarketPlugin: initialized (MultiSourceDataProvider + MarketScanner)")
        except ImportError as e:
            logger.warning(f"MarketPlugin: could not import market modules ({e}) — plugin disabled")
            self._provider = None

    def _normalize_symbols(self, symbols: Any) -> Set[str]:
        return {
            str(symbol).strip().upper()
            for symbol in (symbols or [])
            if str(symbol).strip()
        }

    def _register_crypto_symbols(self, symbols: Set[str]) -> None:
        if symbols and self._provider and hasattr(self._provider, "register_crypto_symbols"):
            self._provider.register_crypto_symbols(symbols)

    def export_runtime_state(self) -> Dict[str, Any]:
        return {
            "crypto_symbols": sorted(self._crypto_symbols),
            "stock_symbols": sorted(self._stock_symbols),
            "crypto_offset": self._crypto_offset,
            "stock_offset": self._stock_offset,
            "last_discovery_ts": self._last_discovery_ts,
            "last_fetch_ts": self._last_fetch_ts,
            "last_successful_fetch_ts": self._last_successful_fetch_ts,
            "total_observations_emitted": self._total_observations_emitted,
        }

    async def persist_runtime_state(self, postgres) -> None:
        if not postgres:
            return
        try:
            await postgres.save_caches({self.STATE_CACHE_KEY: self.export_runtime_state()})
        except Exception as e:
            logger.error(f"MarketPlugin state persist error: {e}")

    def _derive_runtime_state_from_core(self, core) -> Dict[str, Any]:
        derived_crypto: Set[str] = set()
        derived_stocks: Set[str] = set()

        prediction_symbols = getattr(getattr(core, "prediction_engine", None), "symbols", {}) or {}
        for key, history in prediction_symbols.items():
            symbol = str(getattr(history, "symbol", key)).strip().upper()
            if not symbol:
                continue
            domain = str(getattr(history, "domain", "") or "").lower()
            if domain.startswith("crypto:"):
                derived_crypto.add(symbol)
            elif domain.startswith("stock:"):
                derived_stocks.add(symbol)

        price_history = getattr(getattr(core, "cognitive_system", None), "_price_history", {}) or {}
        for symbol in price_history.keys():
            normalized = str(symbol).strip().upper()
            if not normalized:
                continue
            if normalized in derived_crypto or normalized in derived_stocks:
                continue
            if self._provider and self._provider.is_crypto(normalized):
                derived_crypto.add(normalized)
            else:
                derived_stocks.add(normalized)

        meta_domains = getattr(getattr(core, "cognitive_system", None), "_meta_domains", {}) or {}
        for domain_key in meta_domains.keys():
            if not isinstance(domain_key, str) or ":" not in domain_key:
                continue
            prefix, symbol = domain_key.split(":", 1)
            normalized = symbol.strip().upper()
            if not normalized:
                continue
            if prefix.lower() == "crypto":
                derived_crypto.add(normalized)
            elif prefix.lower() == "stock":
                derived_stocks.add(normalized)

        return {
            "crypto_symbols": sorted(derived_crypto),
            "stock_symbols": sorted(derived_stocks),
            "crypto_offset": 0,
            "stock_offset": 0,
            "last_discovery_ts": 0.0,
            "last_fetch_ts": 0.0,
            "last_successful_fetch_ts": 0.0,
            "total_observations_emitted": 0,
        }

    async def restore_runtime_state(self, core) -> None:
        restored_state: Dict[str, Any] = {}
        try:
            postgres = getattr(core, "postgres", None)
            if postgres:
                caches = await postgres.load_caches()
                restored_state = caches.get(self.STATE_CACHE_KEY, {}) or {}
        except Exception as e:
            logger.warning(f"MarketPlugin state restore cache read error: {e}")

        if not restored_state:
            restored_state = self._derive_runtime_state_from_core(core)

        self._crypto_symbols = self._normalize_symbols(restored_state.get("crypto_symbols", []))
        self._stock_symbols = self._normalize_symbols(restored_state.get("stock_symbols", []))
        self._register_crypto_symbols(self._crypto_symbols)

        crypto_len = max(len(self._crypto_symbols), 1)
        stock_len = max(len(self._stock_symbols), 1)
        self._crypto_offset = int(restored_state.get("crypto_offset", 0) or 0) % crypto_len if self._crypto_symbols else 0
        self._stock_offset = int(restored_state.get("stock_offset", 0) or 0) % stock_len if self._stock_symbols else 0
        self._last_discovery_ts = float(restored_state.get("last_discovery_ts", 0.0) or 0.0)
        self._last_fetch_ts = float(restored_state.get("last_fetch_ts", 0.0) or 0.0)
        self._last_successful_fetch_ts = float(restored_state.get("last_successful_fetch_ts", 0.0) or 0.0)
        self._total_observations_emitted = int(restored_state.get("total_observations_emitted", 0) or 0)

        if self.stream_count > 0:
            logger.info(
                "MarketPlugin: restored %s active symbols from persistence (%s crypto, %s stocks)",
                self.stream_count,
                len(self._crypto_symbols),
                len(self._stock_symbols),
            )
        else:
            logger.warning("MarketPlugin: no persisted symbols were available to restore")

    async def discover(self) -> None:
        """Discover new tradeable assets."""
        if not self._scanner:
            return
        try:
            discovered = await self._scanner.scan(
                min_price=Config.MIN_SCAN_PRICE,
                max_price=Config.MAX_SCAN_PRICE,
            )
            new_crypto = self._normalize_symbols(discovered.get("crypto", set()))
            new_stocks = self._normalize_symbols(discovered.get("stocks", set()))
            if new_crypto:
                self._crypto_symbols.update(new_crypto)
                # Register with the provider so newly-discovered crypto symbols
                # (HYPE, PENGU, BASED, etc.) are never routed through the stock
                # cascade and don't trip ORTEX's circuit breaker.
                self._register_crypto_symbols(new_crypto)
                logger.info(f"MarketPlugin: discovered {len(new_crypto)} crypto symbols")
            if new_stocks:
                self._stock_symbols.update(new_stocks)
                logger.info(f"MarketPlugin: discovered {len(new_stocks)} stock symbols")
            if new_crypto or new_stocks:
                self._last_discovery_ts = time.time()
        except Exception as e:
            logger.error(f"MarketPlugin discovery error: {e}")

    async def fetch(self) -> List[tuple]:
        if not self._provider:
            return []

        all_crypto = sorted(self._crypto_symbols)
        all_stocks = sorted(self._stock_symbols)
        if not all_crypto and not all_stocks:
            logger.warning("MarketPlugin: fetch skipped because no active symbols are loaded")
            return []

        self._last_fetch_ts = time.time()

        # Sliding window selection
        crypto_count = min(len(all_crypto), self.MAX_BATCH_SIZE // 2)
        stock_count = min(len(all_stocks), self.MAX_BATCH_SIZE - crypto_count)
        if crypto_count == 0:
            stock_count = min(len(all_stocks), self.MAX_BATCH_SIZE)
        if stock_count == 0:
            crypto_count = min(len(all_crypto), self.MAX_BATCH_SIZE)

        batch = []
        if all_crypto:
            for i in range(crypto_count):
                idx = (self._crypto_offset + i) % len(all_crypto)
                batch.append(all_crypto[idx])
            self._crypto_offset = (self._crypto_offset + crypto_count) % len(all_crypto)
        if all_stocks:
            for i in range(stock_count):
                idx = (self._stock_offset + i) % len(all_stocks)
                batch.append(all_stocks[idx])
            self._stock_offset = (self._stock_offset + stock_count) % len(all_stocks)

        try:
            ticks = await self._provider.fetch_batch(batch)
        except Exception as e:
            logger.error(f"MarketPlugin fetch error: {e}")
            return []

        results = []
        for tick in ticks:
            if not tick or isinstance(tick, Exception):
                continue
            symbol = str(tick.get('symbol', '')).strip().upper()
            if not symbol:
                continue
            is_crypto = self._provider.is_crypto(symbol)
            domain = f"crypto:{symbol}" if is_crypto else f"stock:{symbol}"

            # Translate to generic observation schema
            observation = {
                "entity_id": symbol,
                "value": tick.get('price'),
                "secondary_value": tick.get('volume', 0),
                "timestamp": tick.get('timestamp', time.time()),
                # Preserve original fields for backward compatibility
                "symbol": symbol,
                "price": tick.get('price'),
                "volume": tick.get('volume', 0),
            }
            if observation["value"] is not None:
                results.append((observation, domain))

        if results:
            self._last_successful_fetch_ts = time.time()
            self._total_observations_emitted += len(results)

        return results

    async def close(self) -> None:
        if self._provider:
            try:
                await self._provider.close()
            except Exception:
                pass
        if self._scanner:
            try:
                await self._scanner.close()
            except Exception:
                pass

    @property
    def stream_count(self) -> int:
        return len(self._crypto_symbols) + len(self._stock_symbols)


# ──────────────────────────────────────────────────────────────────────────────
# CognitiveMeshOrchestrator
# ──────────────────────────────────────────────────────────────────────────────

class CognitiveMeshOrchestrator:
    """
    Domain-agnostic orchestrator.

    Manages a list of DataPlugins and feeds their observations into the
    DistributedCognitiveCore.  Adding a new data domain requires only
    implementing a DataPlugin subclass — the core never needs to change.
    """

    def __init__(self):
        self.node_id = Config.NODE_ID
        self.update_interval = Config.UPDATE_INTERVAL

        # ── Data plugins ──────────────────────────────────────────────────
        self.plugins: List[DataPlugin] = []

        # Market plugin is loaded by default (can be disabled via env var)
        if os.getenv("DISABLE_MARKET_PLUGIN", "").lower() not in ("1", "true", "yes"):
            self.market_plugin = MarketPlugin()
            self.plugins.append(self.market_plugin)
        else:
            self.market_plugin = None
            logger.info("MarketPlugin disabled via DISABLE_MARKET_PLUGIN env var")

        # ── Extended market-context plugins ───────────────────────────────
        # Each plugin is optional — a failure to import or init never blocks startup.
        # Disable any plugin via env var: DISABLE_<PLUGIN_NAME>_PLUGIN=1
        _plugin_specs = [
            ("SENTIMENT",      "agents.plugins.sentiment_plugin",      "SentimentPlugin"),
            ("MACRO",          "agents.plugins.macro_plugin",          "MacroPlugin"),
            ("ONCHAIN",        "agents.plugins.onchain_plugin",        "OnChainPlugin"),
            ("NEWS",           "agents.plugins.news_plugin",           "NewsPlugin"),
            ("DERIVATIVES",    "agents.plugins.derivatives_plugin",    "DerivativesPlugin"),
            ("SOCIAL",         "agents.plugins.social_plugin",         "SocialPlugin"),
            ("MICROSTRUCTURE", "agents.plugins.microstructure_plugin", "MicrostructurePlugin"),
        ]
        for _env_name, _module, _cls in _plugin_specs:
            if os.getenv(f"DISABLE_{_env_name}_PLUGIN", "").lower() in ("1", "true", "yes"):
                logger.info(f"{_cls} disabled via DISABLE_{_env_name}_PLUGIN env var")
                continue
            try:
                import importlib
                _mod = importlib.import_module(_module)
                _plugin_cls = getattr(_mod, _cls)
                self.plugins.append(_plugin_cls())
            except Exception as _e:
                logger.warning(f"{_cls} could not be loaded ({_e}) — skipping")

        # ── Core cognitive system ─────────────────────────────────────────
        self.core = DistributedCognitiveCore(node_id=self.node_id)

        # ── Optional networking ───────────────────────────────────────────
        self.network = None
        self.pubsub = None
        if ZMQNode and ZMQPubSub:
            try:
                self.network = ZMQNode(identity=self.node_id)
                self.pubsub = ZMQPubSub(identity=self.node_id)
            except Exception as e:
                logger.warning(f"ZeroMQ init failed (non-critical): {e}")
        # ── Optional persistence ──────────────────────────────────────────────────
        self.postgres = None
        self.milvus = None
        self.redis = None
        self.running = False
        # ── Fault-tolerance watchdog ──────────────────────────────────────────────
        self.always_on: 'AlwaysOnOrchestrator | None' = (
            AlwaysOnOrchestrator(
                checkpoint_interval=300,
                auto_restart=True,
                max_restarts=20,
                health_check_interval=30,
            )
            if AlwaysOnOrchestrator is not None
            else None
        )

    def register_plugin(self, plugin: DataPlugin) -> None:
        """Register an additional data plugin at runtime."""
        self.plugins.append(plugin)
        logger.info(f"Registered DataPlugin: {plugin.name}")

    # ──────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────

    async def initialize(self):
        """Initialize all system components"""
        logger.info(f"Initializing Cognitive Mesh: {self.node_id}")

        # Initialize all plugins
        for plugin in self.plugins:
            try:
                await plugin.initialize()
            except Exception as e:
                logger.warning(f"Plugin '{plugin.name}' init error: {e}")

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

        # Connect to databases
        await self._init_databases()

        # Link databases to core
        self.core.postgres = self.postgres
        self.core.milvus = self.milvus
        self.core.redis = self.redis

        # Expose the market provider to the core for the HTTP provider status tab
        if self.market_plugin and self.market_plugin._provider:
            self.core.data_provider = self.market_plugin._provider

        # Load cognitive state from persistence
        await self.core.load_state()

        # Rehydrate plugin runtime state after the core has recalled persistent memory.
        # This restores the active polling universe so the system resumes calling
        # providers immediately after restart instead of waiting for rediscovery.
        await self._restore_plugin_runtime_state()

        # Immediately hydrate the state cache from restored memory so the dashboard
        # shows recalled concepts/rules/goals on the very first poll (not zeros).
        # Without this, the first cache update is delayed ~5s into the cognitive loop.
        try:
            self.core._update_state_cache()
            logger.info("State cache pre-hydrated from restored memory.")
        except Exception as e:
            logger.warning(f"State cache pre-hydration failed (non-critical): {e}")

        # Start the cognitive loop (background thread)
        self.core.start_cognitive_loop()
        # Wire AlwaysOnOrchestrator as a watchdog for the cognitive loop thread.
        # It will detect if the thread dies unexpectedly and restart it.
        if self.always_on:
            self.always_on.running = True
            self.always_on.start_time = __import__('datetime').datetime.now()
            logger.info("AlwaysOnOrchestrator watchdog started")
        logger.info("Cognitive Mesh initialized successfully")

    async def _init_databases(self):
        """Initialize database connections — all non-blocking with timeouts"""
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

        if Config.MILVUS_HOST and MilvusStore:
            logger.info("Milvus configured — connecting in background...")
            asyncio.create_task(self._connect_milvus_background())
        else:
            logger.info("Milvus not configured — running without vector store")

        if (Config.REDIS_URL or Config.REDIS_HOST) and RedisCache:
            try:
                self.redis = RedisCache(
                    connection_string=Config.REDIS_URL,
                    host=Config.REDIS_HOST,
                    port=Config.REDIS_PORT,
                )
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
        Run the aiohttp HTTP server in a dedicated thread with its own
        event loop to prevent data-collection I/O from blocking HTTP responses.

        This method is intentionally self-contained so it can be called
        repeatedly by _http_thread_watchdog() if the thread dies.
        The cognitive loop is completely independent of this thread —
        HTTP is a read-only window, not a lifecycle dependency.
        """
        http_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(http_loop)
        try:
            logger.info("HTTP server thread: starting aiohttp on dedicated event loop")
            # Pass the market provider for the providers status tab (if available)
            data_provider = (
                self.market_plugin._provider
                if self.market_plugin and self.market_plugin._provider
                else None
            )
            http_loop.run_until_complete(
                start_http_server(self.core, data_provider)
            )
        except Exception as e:
            logger.error(f"HTTP server thread error: {e}", exc_info=True)
        finally:
            try:
                http_loop.close()
            except Exception:
                pass

    def _spawn_http_thread(self) -> threading.Thread:
        """Spawn a new HTTP server daemon thread and return it."""
        t = threading.Thread(
            target=self._start_http_thread,
            name="http-server",
            daemon=True,
        )
        t.start()
        return t

    async def _http_thread_watchdog(self, http_thread: threading.Thread):
        """Monitor the HTTP server thread and restart it if it dies.

        The cognitive loop MUST NOT depend on the HTTP server being alive.
        This watchdog ensures the monitoring window is always available
        without ever touching the cognitive loop.
        """
        self._http_thread = http_thread
        await asyncio.sleep(10)  # Give the thread time to start
        while self.running:
            try:
                if not self._http_thread.is_alive():
                    logger.error(
                        "HTTP server thread died — restarting it "
                        "(cognitive loop is unaffected)"
                    )
                    self._http_thread = self._spawn_http_thread()
                    await asyncio.sleep(5)  # Give it time to bind the port
                    if self._http_thread.is_alive():
                        logger.info("HTTP server thread restarted successfully")
                    else:
                        logger.error("HTTP server thread failed to restart — will retry in 30s")
            except Exception as e:
                logger.error(f"HTTP thread watchdog error: {e}")
            await asyncio.sleep(30)

    async def _keepalive_loop(self):
        """Emit periodic traffic to reduce the chance of idle sleeping.

        Railway's serverless sleep detection is based on network activity, so a
        localhost-only ping is not the most reliable fallback. Prefer an
        explicit public keepalive URL, or Railway's public domain if present,
        and only fall back to localhost when no public address is available.

        This remains a belt-and-suspenders measure alongside
        `sleepApplication: false` in railway.json.
        """
        import urllib.request
        port = int(os.environ.get("PORT", 8080))
        public_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
        keepalive_url = os.environ.get("KEEPALIVE_URL", "").strip()
        if not keepalive_url and public_domain:
            keepalive_url = f"https://{public_domain}/health"
        if not keepalive_url:
            keepalive_url = f"http://localhost:{port}/health"

        logger.info(f"Keepalive loop targeting {keepalive_url}")

        # Wait for HTTP server to be ready before first ping
        await asyncio.sleep(30)
        while self.running:
            try:
                # Use a synchronous urllib call in a thread to avoid
                # blocking the event loop
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(keepalive_url, timeout=10).read()
                )
                logger.debug("Keepalive ping sent")
            except Exception as e:
                # Non-fatal — HTTP server may be restarting or the public route
                # may not yet be ready.
                logger.debug(f"Keepalive ping failed: {e}")
            # Ping every 8 minutes (Railway sleeps after 10 min of silence)
            await asyncio.sleep(480)

    async def run(self):
        """Run the Cognitive Mesh with phased startup"""
        self.running = True
        self._http_thread = None

        try:
            # PHASE 1: Start HTTP server in dedicated thread
            logger.info("PHASE 1: Starting HTTP server (dedicated thread)...")
            http_thread = self._spawn_http_thread()
            await asyncio.sleep(2)
            if http_thread.is_alive():
                logger.info("HTTP server is running.")
            else:
                logger.error("HTTP server thread died unexpectedly")

            # PHASE 2: Initialize mesh
            logger.info("PHASE 2: Mesh initialization...")
            await self.initialize()

            # PHASE 3: Initial data discovery
            logger.info("PHASE 3: Initial data discovery...")
            await self._initial_discovery()

            # PHASE 4: Start execution loops
            stream_count = sum(
                getattr(p, 'stream_count', 0) for p in self.plugins
            )
            logger.info("=" * 60)
            logger.info("  COGNITIVE MESH IS LIVE")
            logger.info("  Engines: Abstraction, Reasoning, CrossDomain,")
            logger.info("           Goals, Learning, Evolution, Orchestrator")
            logger.info(f"  Plugins: {[p.name for p in self.plugins]}")
            logger.info(f"  Streams: {stream_count} discovered")
            logger.info("  API:     /api/chat, /api/ingest, /api/state")
            logger.info("=" * 60)

            loops = [
                self._data_collection_loop(),
                self._discovery_loop(),
                self._metrics_reporter_loop(),
                self._checkpoint_loop(),
                self._pursuit_loop(),
                self._cognitive_watchdog_loop(),
                self._http_thread_watchdog(http_thread),
                self._keepalive_loop(),
            ]
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
    # Discovery
    # ──────────────────────────────────────────

    async def _initial_discovery(self):
        """Ask all plugins to discover their initial entity sets."""
        for plugin in self.plugins:
            if hasattr(plugin, 'discover'):
                try:
                    await plugin.discover()
                except Exception as e:
                    logger.error(f"Plugin '{plugin.name}' initial discovery error: {e}")

        # Seed the mesh with first observations immediately
        observations = []
        for plugin in self.plugins:
            try:
                obs = await plugin.fetch()
                observations.extend(obs)
            except Exception as e:
                logger.error(f"Plugin '{plugin.name}' initial fetch error: {e}")

        if observations:
            logger.info(f"Seeding mesh with {len(observations)} initial observations...")
            for observation, domain in observations:
                await self.core.ingest(observation, domain)

    async def _discovery_loop(self):
        """Periodically ask plugins to discover new entities."""
        while self.running:
            try:
                await asyncio.sleep(300)  # every 5 minutes
                for plugin in self.plugins:
                    if hasattr(plugin, 'discover'):
                        try:
                            await plugin.discover()
                        except Exception as e:
                            logger.error(f"Plugin '{plugin.name}' discovery error: {e}")
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(60)

    # ──────────────────────────────────────────
    # Data Collection Loop
    # ──────────────────────────────────────────

    async def _data_collection_loop(self):
        """
        Continuously fetch observations from all plugins and feed them
        into the cognitive core.
        """
        logger.info("Starting data collection loop (plugin-based)")
        _last_skip_reset = time.time()
        _SKIP_RESET_INTERVAL = 600  # reset provider skip-list every 10 minutes
        while self.running:
            try:
                total_ingested = 0

                # ── Periodic skip-list reset ─────────────────────────────────
                # After rate-limiting, symbols accumulate in the provider's
                # skip-list and are never retried. Reset every 10 min.
                now = time.time()
                if now - _last_skip_reset >= _SKIP_RESET_INTERVAL:
                    for plugin in self.plugins:
                        provider = getattr(plugin, '_provider', None)
                        if provider is not None and hasattr(provider, 'reset_skip_list'):
                            provider.reset_skip_list()
                            logger.info(f"Plugin '{plugin.name}': skip list reset")
                    _last_skip_reset = now

                # ── Fetch from all plugins ────────────────────────────────────
                for plugin in self.plugins:
                    try:
                        observations = await plugin.fetch()
                        for observation, domain in observations:
                            await self.core.ingest(observation, domain)
                            if self.redis:
                                try:
                                    entity_id = observation.get('entity_id', domain)
                                    await self.redis.cache_tick(entity_id, observation)
                                except Exception:
                                    pass
                            total_ingested += 1
                    except Exception as e:
                        logger.error(f"Plugin '{plugin.name}' fetch error: {e}")

                # ── Heartbeat fallback when all providers are dry ──────────────
                # If no live data arrived, re-ingest a sample of last-known prices
                # so the cognitive loop keeps processing and learning continuously.
                if total_ingested == 0:
                    try:
                        price_hist = self.core.cognitive_system._price_history
                        heartbeat_count = 0
                        for history_key, prices in list(price_hist.items()):
                            if not prices:
                                continue
                            parts = history_key.split(':', 1)
                            domain = parts[0] if len(parts) == 2 else 'market'
                            entity_id = parts[1] if len(parts) == 2 else history_key
                            obs = {
                                'entity_id': entity_id,
                                'value': prices[-1],
                                'secondary_value': 0,
                                'timestamp': time.time(),
                                'symbol': entity_id,
                                'price': prices[-1],
                                'volume': 0,
                                '_heartbeat': True,
                            }
                            await self.core.ingest(obs, domain)
                            total_ingested += 1
                            heartbeat_count += 1
                            if heartbeat_count >= 10:
                                break
                        if heartbeat_count > 0:
                            logger.debug(
                                f"Heartbeat: re-ingested {heartbeat_count} cached prices "
                                f"(all providers dry)"
                            )
                    except Exception as hb_err:
                        logger.debug(f"Heartbeat fallback error: {hb_err}")

                if total_ingested > 0:
                    logger.info(
                        f"Ingested {total_ingested} observations across "
                        f"{len(self.plugins)} plugin(s)"
                    )

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
                cached = self.core.get_cached_state()
                state = {
                    "node_id": cached.get('node_id', self.core.node_id),
                    "phi": cached.get('metrics', {}).get('global_coherence_phi', 0),
                    "sigma": cached.get('metrics', {}).get('noise_level_sigma', 0),
                    "metrics": cached.get('metrics', {}),
                    "timestamp": time.time(),
                }
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
                cached = self.core.get_cached_state()
                metrics = cached.get('metrics', {})
                if self.pubsub and metrics:
                    await self.pubsub.publish("metrics", metrics)

                stream_count = sum(getattr(p, 'stream_count', 0) for p in self.plugins)
                logger.info(
                    f"METRICS | PHI={metrics.get('global_coherence_phi', 0):.3f} "
                    f"SIGMA={metrics.get('noise_level_sigma', 0):.3f} "
                    f"Concepts={metrics.get('total_concepts', 0)} "
                    f"Rules={metrics.get('total_rules', 0)} "
                    f"Goals={metrics.get('total_goals', 0)} "
                    f"Obs={metrics.get('total_observations', 0)} "
                    f"Streams={metrics.get('streams_tracked', stream_count)} "
                    f"Evolution={self.core._evolution_counter}"
                )
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(10)

    async def _pursuit_loop(self):
        """Run the PursuitAgent cycle every 60 seconds to refine rules and execute goals."""
        from agents.pursuit_agent import PursuitAgent
        from agents.autonomous_reasoner import AutonomousReasoner
        reasoner = AutonomousReasoner()
        agent = PursuitAgent(core=self.core, pubsub=self.pubsub, reasoner=reasoner)
        logger.info("PursuitAgent loop started")
        await asyncio.sleep(60)  # Let the system warm up first
        while self.running:
            try:
                await agent.run_pursuit_cycle()
                # Flush pursuit log back to core activity log
                for entry in agent.goal_history[-5:]:
                    self.core._activity_log.append({
                        "ts": entry.get("started", 0) * 1000,
                        "msg": f"Pursuit: {entry.get('goal', '?')} [{entry.get('source', 'agent')}]"
                    })
            except Exception as e:
                logger.error(f"Error in pursuit loop: {e}")
            await asyncio.sleep(60)

    async def _persist_plugin_runtime_state(self):
        """Persist plugin-specific runtime state that lives outside the core."""
        for plugin in self.plugins:
            if hasattr(plugin, "persist_runtime_state"):
                try:
                    await plugin.persist_runtime_state(self.core.postgres)
                except Exception as e:
                    logger.error(f"Plugin '{plugin.name}' runtime state persist error: {e}")

    async def _restore_plugin_runtime_state(self):
        """Restore plugin-specific runtime state from persistent stores and core memory."""
        for plugin in self.plugins:
            if hasattr(plugin, "restore_runtime_state"):
                try:
                    await plugin.restore_runtime_state(self.core)
                except Exception as e:
                    logger.error(f"Plugin '{plugin.name}' runtime state restore error: {e}")

    async def _checkpoint_loop(self):
        """Periodically save all cognitive state to Postgres.

        This is the safety net that ensures the mesh never loses more than
        CHECKPOINT_INTERVAL seconds of learned state, even if Railway sends
        SIGKILL without a SIGTERM grace period (e.g. OOM kill, hard redeploy).
        Default: every 5 minutes.
        """
        checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", 300))
        logger.info(f"Checkpoint loop started — saving state every {checkpoint_interval}s")
        await asyncio.sleep(checkpoint_interval)  # Skip first cycle (startup)
        while self.running:
            try:
                await self._persist_plugin_runtime_state()
                await self.core.save_state()
                logger.info("Periodic checkpoint saved.")
            except Exception as e:
                logger.error(f"Checkpoint save failed: {e}")
            await asyncio.sleep(checkpoint_interval)

    async def _cognitive_watchdog_loop(self):
        """Monitor the cognitive loop thread and restart it if it dies unexpectedly.

        The cognitive loop thread has per-iteration exception handling, so it
        should never die under normal conditions.  This watchdog is a last-resort
        safety net: if the thread is somehow no longer alive (e.g. an unhandled
        exception escaped the inner try/except), we restart it immediately and
        log an error so the issue is visible in Railway logs.
        """
        logger.info("Cognitive watchdog loop started (checking every 30s)")
        await asyncio.sleep(60)  # Give the thread time to start up before first check
        while self.running:
            try:
                thread = getattr(self.core, '_cognitive_thread', None)
                if thread is not None and not thread.is_alive() and self.core._running:
                    logger.error(
                        "WATCHDOG: Cognitive loop thread died unexpectedly — restarting now"
                    )
                    if self.always_on:
                        self.always_on.error_log.append({
                            'task_id': 'cognitive_loop',
                            'task_name': 'CognitiveLoop',
                            'error': 'Thread died unexpectedly',
                            'restart_count': getattr(self, '_watchdog_restarts', 0) + 1,
                            'timestamp': __import__('datetime').datetime.now().isoformat(),
                        })
                    self._watchdog_restarts = getattr(self, '_watchdog_restarts', 0) + 1
                    # Reset the running flag so start_cognitive_loop() accepts the call
                    self.core._running = False
                    await asyncio.sleep(1)
                    self.core.start_cognitive_loop()
                    logger.info(
                        f"WATCHDOG: Cognitive loop restarted "
                        f"(total restarts: {self._watchdog_restarts})"
                    )
                elif thread is not None and thread.is_alive():
                    # Thread is healthy — optionally report status to AlwaysOnOrchestrator
                    if self.always_on:
                        self.always_on.health_history.append({
                            'timestamp': __import__('datetime').datetime.now().isoformat(),
                            'cognitive_thread': 'alive',
                            'running': self.core._running,
                        })
                        # Keep only the last 100 health entries
                        if len(self.always_on.health_history) > 100:
                            self.always_on.health_history = self.always_on.health_history[-100:]
            except Exception as e:
                logger.error(f"Cognitive watchdog error: {e}")
            await asyncio.sleep(30)

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

        # Stop the AlwaysOnOrchestrator watchdog
        if self.always_on:
            try:
                self.always_on.running = False
                logger.info("AlwaysOnOrchestrator watchdog stopped")
            except Exception as e:
                logger.warning(f"AlwaysOnOrchestrator stop error: {e}")

        self.core.stop_cognitive_loop()

        logger.info("Saving cognitive state to persistence...")
        try:
            await self._persist_plugin_runtime_state()
            await self.core.save_state()
        except Exception as e:
            logger.error(f"Failed to save cognitive state: {e}")

        # Shutdown plugins
        for plugin in self.plugins:
            try:
                await plugin.close()
            except Exception:
                pass

        tasks = []
        if self.network:
            tasks.append(self.network.stop())
        if self.pubsub:
            tasks.append(self.pubsub.stop())
        if self.postgres:
            tasks.append(self.postgres.disconnect())
        if self.milvus:
            tasks.append(self.milvus.disconnect())
        if self.redis:
            tasks.append(self.redis.disconnect())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Cognitive Mesh shutdown complete")


async def main():
    """Main entry point"""
    orchestrator = CognitiveMeshOrchestrator()

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
