import os
import sys
import asyncio
import logging
import time
from typing import List, Dict, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CognitiveMesh")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from core.distributed_core import DistributedCognitiveCore
from agents.market_data_providers import MultiSourceDataProvider
from agents.pursuit_agent import PursuitAgent
from agents.autonomous_reasoner import AutonomousReasoner
from shared.network_zeromq import ZMQNode, ZMQPubSub
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

class CognitiveMeshOrchestrator:
    """Main orchestrator for the Cognitive Mesh system"""
    
    def __init__(self):
        self.node_id = Config.NODE_ID
        self.symbols = self._load_symbols()
        self.update_interval = Config.UPDATE_INTERVAL
        
        # Initialize components
        self.data_provider = MultiSourceDataProvider()
        self.core = DistributedCognitiveCore(node_id=self.node_id)
        self.network = ZMQNode(identity=self.node_id)
        self.pubsub = ZMQPubSub(identity=self.node_id)
        
        # Optional persistence
        self.postgres = None
        self.milvus = None
        self.redis = None
        
        # Initialize autonomous reasoning
        self.reasoner = AutonomousReasoner(self.core)
        self.pursuit = PursuitAgent(self.core, self.pubsub, self.reasoner)
        self.running = False

    def _load_symbols(self) -> Set[str]:
        """Load symbols from environment variable only - no hardcoded defaults"""
        symbols = Config.get_symbols()
        if symbols:
            logger.info(f"Loaded {len(symbols)} symbols from SYMBOLS environment variable")
        else:
            logger.info("No seed symbols configured. Mesh will operate on organically discovered or manually injected data only.")
        return symbols

    async def initialize(self):
        """Initialize all system components with robust error handling"""
        logger.info(f"Initializing Cognitive Mesh: {self.node_id}")
        
        # Start networking
        try:
            await self.network.start()
            await self.pubsub.start_publisher()
            await self.pubsub.start_subscriber(["concept", "rule", "transfer", "metrics", "goal"])
        except Exception as e:
            logger.error(f"Networking initialization error: {e}")
        
        # Connect to databases if configured
        await self._init_databases()
        
        # Link databases to core
        self.core.postgres = self.postgres
        self.core.milvus = self.milvus
        self.core.redis = self.redis
            
        logger.info("Cognitive Mesh initialized successfully")

    async def _init_databases(self):
        """Initialize database connections"""
        try:
            if Config.POSTGRES_URL and PostgresStore:
                self.postgres = PostgresStore(Config.POSTGRES_URL)
                await self.postgres.connect()
                logger.info("Connected to PostgreSQL")
                
            if Config.MILVUS_HOST and MilvusStore:
                self.milvus = MilvusStore(Config.MILVUS_HOST)
                await self.milvus.connect()
                logger.info("Connected to Milvus")
                
            if Config.REDIS_URL and RedisCache:
                self.redis = RedisCache(Config.REDIS_URL)
                await self.redis.connect()
                logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    async def run(self):
        """Run the Cognitive Mesh with phased startup"""
        self.running = True
        
        try:
            # PHASE 1: Start HTTP server for health checks
            logger.info("PHASE 1: Starting HTTP server...")
            http_task = asyncio.create_task(start_http_server(self.core, self.data_provider))
            # Wait to see if it fails immediately
            await asyncio.sleep(2)
            if http_task.done():
                try:
                    http_task.result()
                except Exception as e:
                    logger.error(f"HTTP server failed to start: {e}")
                    # Don't exit, but log the error
            else:
                logger.info("HTTP server task is running.")
            
            # PHASE 2: Background mesh initialization
            logger.info("PHASE 2: Mesh initialization...")
            await self.initialize()
            
            # PHASE 3: Start concurrent execution loops
            logger.info("PHASE 3: Starting execution loops.")
            loops = [
                self._data_collection_loop(),
                self._pursuit_loop(),
                self._gossip_loop(),
                self._metrics_reporter_loop(),
                self._network_listener_loop(),
                self._pubsub_listener_loop(),
                http_task
            ]
            
            await asyncio.gather(*loops, return_exceptions=True)
        
        except KeyboardInterrupt:
            logger.info("Shutting down due to user interrupt...")
        except Exception as e:
            logger.fatal(f"Unexpected error in main run loop: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def _data_collection_loop(self):
        """Continuously collect data from multiple sources with Attention Priority"""
        logger.info("Starting prioritized data collection loop")
        
        while self.running:
            try:
                # Organic discovery from core concepts
                self._discover_new_symbols()
                
                # Prepare batch
                current_batch = self._prepare_data_batch()
                
                if not current_batch:
                    # No symbols to fetch - mesh is idle, waiting for manual ingestion
                    await asyncio.sleep(self.update_interval)
                    continue
                
                logger.info(f"Processing Attention Batch: {current_batch}")
                ticks = await self.data_provider.fetch_batch(current_batch)
                
                for tick in ticks:
                    if not tick or isinstance(tick, Exception): 
                        continue
                    # Determine domain based on symbol type
                    symbol = tick.get('symbol')
                    if self.data_provider.is_crypto(symbol):
                        domain = f"crypto:{symbol}"
                    else:
                        domain = f"stock:{symbol}"
                    await self.core.ingest(tick, domain)
                    if self.redis:
                        await self.redis.cache_tick(symbol, tick)
                
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(10)

    def _discover_new_symbols(self):
        """Find new symbols from formed concepts"""
        active_concepts = self.core.get_concepts_snapshot()
        for cid, concept in active_concepts.items():
            domain = concept.get("domain", "")
            if domain.startswith("stock:") or domain.startswith("crypto:"):
                symbol = domain.split(":")[1].upper()
                if symbol not in self.symbols:
                    logger.info(f"Organically discovered new asset: {symbol}")
                    self.symbols.add(symbol)

    def _prepare_data_batch(self) -> List[str]:
        """Prepare a batch of symbols to fetch - returns empty list if no symbols available"""
        if not self.symbols:
            return []  # No symbols to fetch - mesh operates on manual ingestion only
        
        all_symbols = list(self.symbols)
        batch_size = Config.DATA_BATCH_SIZE
        current_batch = all_symbols[:batch_size]
        
        # Rotate symbols for next cycle
        self.symbols = set(all_symbols[batch_size:] + all_symbols[:batch_size])
        return current_batch

    async def _pursuit_loop(self):
        """Periodically run pursuit agent cycles"""
        while self.running:
            try:
                await self.pursuit.run_pursuit_cycle()
                await asyncio.sleep(45)
            except Exception as e:
                logger.error(f"Error in pursuit loop: {e}")
                await asyncio.sleep(15)

    async def _gossip_loop(self):
        """Periodically broadcast gossip state"""
        while self.running:
            try:
                state = self.core.get_state_summary()
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
                await self.pubsub.publish("metrics", metrics)
                await asyncio.sleep(15)
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(10)

    async def _network_listener_loop(self):
        """Listen for incoming network messages"""
        while self.running:
            try:
                msg = await self.network.receive()
                if msg:
                    await self.core.process_network_message(msg)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in network listener: {e}")
                await asyncio.sleep(1)

    async def _pubsub_listener_loop(self):
        """Listen for pubsub messages"""
        while self.running:
            try:
                msg = await self.pubsub.receive()
                if msg:
                    await self.core.process_pubsub_message(msg)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in pubsub listener: {e}")
                await asyncio.sleep(1)

    async def shutdown(self):
        """Shutdown all components gracefully"""
        self.running = False
        logger.info("Shutting down Cognitive Mesh...")
        
        tasks = [
            self.network.stop(),
            self.pubsub.stop()
        ]
        
        if self.postgres: tasks.append(self.postgres.disconnect())
        if self.milvus: tasks.append(self.milvus.disconnect())
        if self.redis: tasks.append(self.redis.disconnect())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Cognitive Mesh shutdown complete")

async def main():
    """Main entry point"""
    orchestrator = CognitiveMeshOrchestrator()
    await orchestrator.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
