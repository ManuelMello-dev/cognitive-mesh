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

from core.distributed_core import DistributedCognitiveCore
from agents.multi_source_provider import MultiSourceDataProvider
from agents.pursuit_agent import PursuitAgent
from shared.network_zeromq import ZMQNode, ZMQPubSub
# Optional stores - only import if they exist
try:
    from shared.postgres_store import PostgresStore
except ImportError:
    PostgresStore = None
try:
    from shared.milvus_store import MilvusStore
except ImportError:
    MilvusStore = None
try:
    from shared.redis_cache import RedisCache
except ImportError:
    RedisCache = None
from http_server import start_http_server

class CognitiveMeshOrchestrator:
    """Main orchestrator for the Cognitive Mesh system"""
    
    def __init__(self):
        self.node_id = os.getenv("NODE_ID", "global_mind_01")
        # Load expanded seed symbols from file
        try:
            with open("seed_symbols.txt", "r") as f:
                seeds = f.read().strip().split(",")
            self.symbols = set(seeds)
            logger.info(f"Loaded {len(self.symbols)} seed symbols from file")
        except Exception as e:
            logger.warning(f"Could not load seed_symbols.txt: {e}. Falling back to defaults.")
            self.symbols = set(os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL,TSLA,NVDA,BTC,ETH,SOL,BNB,XRP").split(","))
            
        self.update_interval = int(os.getenv("UPDATE_INTERVAL", "60"))
        
        # Initialize components
        self.data_provider = MultiSourceDataProvider()
        self.core = DistributedCognitiveCore(node_id=self.node_id)
        self.network = ZMQNode(identity=self.node_id)
        self.pubsub = ZMQPubSub(identity=self.node_id)
        
        # Optional persistence
        self.postgres = None
        self.milvus = None
        self.redis = None
        
        self.pursuit = PursuitAgent(self.core, self.pubsub)
        self.running = False

    async def initialize(self):
        """Initialize all system components"""
        logger.info(f"Initializing Cognitive Mesh: {self.node_id}")
        
        # Start networking
        try:
            await self.network.start()
            await self.pubsub.start_publisher()
            await self.pubsub.start_subscriber(["concept", "rule", "transfer", "metrics", "goal"])
        except Exception as e:
            logger.error(f"Networking initialization error: {e}")
        
        # Connect to databases if configured and classes are available
        try:
            if os.getenv("POSTGRES_URL") and PostgresStore:
                self.postgres = PostgresStore(os.getenv("POSTGRES_URL"))
                await self.postgres.connect()
                
            if os.getenv("MILVUS_HOST") and MilvusStore:
                self.milvus = MilvusStore(os.getenv("MILVUS_HOST"))
                await self.milvus.connect()
                
            if os.getenv("REDIS_URL") and RedisCache:
                self.redis = RedisCache(os.getenv("REDIS_URL"))
                await self.redis.connect()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            
        logger.info("Cognitive Mesh initialized successfully")

    async def run(self):
        """Run the Cognitive Mesh"""
        self.running = True
        
        try:
            # PHASE 1: Start HTTP server IMMEDIATELY to satisfy Railway health checks
            # This is critical to prevent the container from being killed during startup
            logger.info("PHASE 1: Starting HTTP server for health checks...")
            http_task = asyncio.create_task(start_http_server(self.core))
            
            # Wait a moment for HTTP server to bind
            await asyncio.sleep(2)
            logger.info("HTTP server should be active. Proceeding with mesh initialization.")
            
            # PHASE 2: Initialize mesh in the background to avoid blocking
            logger.info("PHASE 2: Background mesh initialization...")
            await self.initialize()
            
            # PHASE 3: Start concurrent execution loops
            logger.info("PHASE 3: Starting execution loops.")
            await asyncio.gather(
                self._data_collection_loop(),
                self._pursuit_loop(),
                self._gossip_loop(),
                self._metrics_reporter_loop(),
                self._network_listener_loop(),
                self._pubsub_listener_loop(),
                http_task,
                return_exceptions=True
            )
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        finally:
            await self.shutdown()

    async def _data_collection_loop(self):
        """Continuously collect data from multiple sources with Attention Priority"""
        logger.info("Starting prioritized data collection loop")
        
        batch_size = 20
        priority_symbols = ["BTC", "ETH", "SOL", "HYPE", "TRUMP", "PUMP", "AAPL", "NVDA", "TSLA"]
        
        while self.running:
            try:
                # Organic discovery
                active_concepts = self.core.get_concepts_snapshot()
                for cid, concept in active_concepts.items():
                    domain = concept.get("domain", "")
                    if domain.startswith("stock:") or domain.startswith("crypto:"):
                        symbol = domain.split(":")[1].upper()
                        if symbol not in self.symbols:
                            logger.info(f"Organically discovered new asset: {symbol}")
                            self.symbols.add(symbol)
                
                all_symbols = list(self.symbols)
                # Filter priority to those actually in self.symbols
                current_priority = [s for s in priority_symbols if s in self.symbols]
                others = [s for s in all_symbols if s not in current_priority]
                
                # Process priority symbols + a rotating batch of others
                current_batch = current_priority + others[:batch_size]
                # Rotate symbols for next cycle
                self.symbols = set(current_priority + others[batch_size:] + others[:batch_size])
                
                logger.info(f"Processing Attention Batch: {current_batch}")
                ticks = await self.data_provider.fetch_batch(current_batch)
                
                for tick in ticks:
                    if isinstance(tick, Exception): continue
                    domain = f"stock:{tick.get('symbol')}"
                    await self.core.ingest(tick, domain)
                    if self.redis:
                        await self.redis.cache_tick(tick.get('symbol'), tick)
                
                metrics = self.core.get_metrics()
                logger.info(f"System Metrics: {metrics}")
                
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(5)

    async def _pursuit_loop(self):
        """Periodically run pursuit agent cycles"""
        while self.running:
            try:
                await self.pursuit.run_pursuit_cycle()
                await asyncio.sleep(45)
            except Exception as e:
                logger.error(f"Error in pursuit loop: {e}")
                await asyncio.sleep(10)

    async def _gossip_loop(self):
        """Periodically broadcast gossip state"""
        while self.running:
            try:
                state = self.core.get_state_summary()
                await self.network.broadcast_gossip(state)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
                await asyncio.sleep(10)

    async def _metrics_reporter_loop(self):
        """Periodically report system-wide metrics"""
        while self.running:
            try:
                metrics = self.core.get_metrics()
                await self.pubsub.publish("metrics", metrics)
                await asyncio.sleep(15)
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(5)

    async def _network_listener_loop(self):
        """Listen for incoming network messages"""
        while self.running:
            try:
                msg = await self.network.receive()
                if msg:
                    await self.core.process_network_message(msg)
                else:
                    await asyncio.sleep(0.1) # Prevent CPU spinning
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
                    await asyncio.sleep(0.1) # Prevent CPU spinning
            except Exception as e:
                logger.error(f"Error in pubsub listener: {e}")
                await asyncio.sleep(1)

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
