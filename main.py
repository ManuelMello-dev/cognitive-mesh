import asyncio
import logging
import sys
import os

# Add the current directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cognitive_mesh.core.distributed_core import DistributedCognitiveCore
from cognitive_mesh.shared.network import ZMQNode
from cognitive_mesh.agents.provider import DynamicStockProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CognitiveMeshMain")

async def main():
    node_id = "global_mind_01"
    core = DistributedCognitiveCore(node_id)
    network = ZMQNode(node_id, port=5555)
    
    await network.start()
    
    logger.info("Starting Cognitive Mesh...")
    
    # Run the core and the network listener concurrently
    async def network_listener():
        await network.listen(core.process_observation)

    await asyncio.gather(
        core.run(),
        network_listener()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
