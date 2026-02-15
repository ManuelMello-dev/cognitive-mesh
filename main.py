"""
Cognitive Mesh - Main Orchestrator
Coordinates all components: Data Providers, Gossip Protocol, Databases, and Core
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CognitiveMesh")

# Import components with graceful fallbacks for optional dependencies
from agents.multi_source_provider import MultiSourceDataProvider
from core.distributed_core import DistributedCognitiveCore
from shared.gossip_amfg import AMFGProtocol
from shared.network_zeromq import ZMQNode, ZMQAgent, ZMQPubSub

# Optional database components
try:
    from storage.postgres_store import PostgresStore
    POSTGRES_AVAILABLE = True
except ImportError:
    logger.warning("PostgreSQL support not available (asyncpg not installed)")
    PostgresStore = None
    POSTGRES_AVAILABLE = False

try:
    from storage.milvus_store import MilvusStore
    MILVUS_AVAILABLE = True
except ImportError:
    logger.warning("Milvus support not available (pymilvus not installed)")
    MilvusStore = None
    MILVUS_AVAILABLE = False

try:
    from storage.redis_cache import RedisCache
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis support not available (aioredis not installed)")
    RedisCache = None
    REDIS_AVAILABLE = False


class CognitiveMeshOrchestrator:
    """Main orchestrator for the Cognitive Mesh system"""
    
    def __init__(self):
        self.node_id = os.getenv("NODE_ID", "global_mind_01")
        self.symbols = os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL,TSLA,NVDA,BTC,ETH").split(",")
        self.update_interval = int(os.getenv("UPDATE_INTERVAL", "60"))
        
        # Initialize components
        self.data_provider = MultiSourceDataProvider()
        
        # Initialize optional database components
        self.postgres = PostgresStore() if POSTGRES_AVAILABLE else None
        self.milvus = MilvusStore() if MILVUS_AVAILABLE else None
        self.redis = RedisCache() if REDIS_AVAILABLE else None
        
        self.core = DistributedCognitiveCore(self.node_id, self.postgres, self.milvus, self.redis)
        self.gossip = AMFGProtocol(self.node_id)
        self.network = ZMQNode(self.node_id, port=int(os.getenv("LISTEN_PORT", "5555")))
        self.pubsub = ZMQPubSub(self.node_id)
        
        self.running = False
    
    async def initialize(self):
        """Initialize all components"""
        logger.info(f"Initializing Cognitive Mesh: {self.node_id}")
        
        # Connect to databases if available
        if self.postgres:
            logger.info("Connecting to PostgreSQL...")
            try:
                await self.postgres.connect()
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
                self.postgres = None
        
        if self.milvus:
            logger.info("Connecting to Milvus...")
            try:
                await self.milvus.connect()
            except Exception as e:
                logger.warning(f"Milvus connection failed: {e}")
                self.milvus = None
        
        if self.redis:
            logger.info("Connecting to Redis...")
            try:
                await self.redis.connect()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis = None
        
        # Start network components
        logger.info("Starting network components...")
        await self.network.start()
        await self.pubsub.start_publisher()
        await self.pubsub.start_subscriber(topics=["concept", "rule", "transfer"])
        
        # Register message handlers
        self.network.register_handler("observation", self._handle_observation)
        self.pubsub.register_callback("concept", self._handle_new_concept)
        self.pubsub.register_callback("rule", self._handle_new_rule)
        self.pubsub.register_callback("transfer", self._handle_transfer)
        
        logger.info("Cognitive Mesh initialized successfully")
    
    async def _handle_observation(self, sender: str, message: dict):
        """Handle incoming observation from an agent"""
        try:
            observation = message.get("data", {})
            domain = message.get("domain", "unknown")
            
            result = await self.core.ingest(observation, domain)
            
            if result.get("success"):
                # Cache metrics
                if self.redis:
                    await self.redis.increment_counter("observations_processed")
                
                # Broadcast if high-confidence concept formed
                if result.get("concept_id"):
                    await self.pubsub.publish("concept", {
                        "type": "NEW_CONCEPT",
                        "concept_id": result["concept_id"],
                        "domain": domain,
                        "timestamp": result["timestamp"]
                    })
        except Exception as e:
            logger.error(f"Error handling observation: {e}")
    
    async def _handle_new_concept(self, message: dict):
        """Handle new concept event from gossip"""
        logger.info(f"Received new concept: {message.get('concept_id')}")
        await self.gossip.broadcast(message, priority=1.0)
    
    async def _handle_new_rule(self, message: dict):
        """Handle new rule event from gossip"""
        logger.info(f"Received new rule: {message}")
        await self.gossip.broadcast(message, priority=0.8)
    
    async def _handle_transfer(self, message: dict):
        """Handle cross-domain transfer event"""
        logger.info(f"Cross-domain transfer: {message}")
    
    async def _data_collection_loop(self):
        """Continuously collect data from multiple sources"""
        logger.info(f"Starting data collection for symbols: {self.symbols}")
        
        while self.running:
            try:
                # Fetch data for all symbols
                ticks = await self.data_provider.fetch_batch(self.symbols)
                
                # Process each tick
                for tick in ticks:
                    if isinstance(tick, Exception):
                        logger.error(f"Data fetch error: {tick}")
                        continue
                    
                    # Ingest into core
                    domain = f"stock:{tick.get('symbol')}"
                    result = await self.core.ingest(tick, domain)
                    
                    # Cache the tick
                    if self.redis:
                        await self.redis.cache_tick(tick.get('symbol'), tick)
                    
                    logger.debug(f"Processed tick for {tick.get('symbol')}")
                
                # Log metrics periodically
                if self.core.iteration % 10 == 0:
                    metrics = self.core.get_metrics()
                    logger.info(f"Metrics: {metrics}")
                
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(5)
    
    async def _gossip_loop(self):
        """Periodically broadcast gossip state"""
        while self.running:
            try:
                stats = self.gossip.get_stats()
                
                # Broadcast gossip stats
                await self.pubsub.publish("gossip", stats)
                
                # Cache gossip state
                if self.redis:
                    await self.redis.set_gossip_state(stats)
                
                await asyncio.sleep(30)
            
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_reporter_loop(self):
        """Periodically report system metrics"""
        while self.running:
            try:
                metrics = self.core.get_metrics()
                logger.info(f"System Metrics: {metrics}")
                
                # Save to database
                if self.postgres:
                    await self.postgres.save_metrics(metrics)
                
                await asyncio.sleep(60)
            
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(10)
    
    async def _network_listener_loop(self):
        """Listen for incoming network messages"""
        await self.network.listen(self._handle_observation)
    
    async def _pubsub_listener_loop(self):
        """Listen for published messages"""
        await self.pubsub.listen()
    
    async def run(self):
        """Run the Cognitive Mesh"""
        self.running = True
        
        try:
            await self.initialize()
            
            # Run all components concurrently
            await asyncio.gather(
                self._data_collection_loop(),
                self._gossip_loop(),
                self._metrics_reporter_loop(),
                self._network_listener_loop(),
                self._pubsub_listener_loop(),
                return_exceptions=True
            )
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown all components"""
        self.running = False
        logger.info("Shutting down Cognitive Mesh...")
        
        await self.network.stop()
        await self.pubsub.stop()
        
        if self.postgres:
            await self.postgres.disconnect()
        if self.milvus:
            await self.milvus.disconnect()
        if self.redis:
            await self.redis.disconnect()
        
        logger.info("Cognitive Mesh shutdown complete")


async def main():
    """Main entry point"""
    orchestrator = CognitiveMeshOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
