import asyncio
import aiohttp
import json
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMesh")

async def test_api():
    url = "http://localhost:8080"
    async with aiohttp.ClientSession() as session:
        # Test Metrics
        try:
            async with session.get(f"{url}/api/metrics") as resp:
                if resp.status == 200:
                    metrics = await resp.json()
                    logger.info(f"Metrics API OK: {metrics}")
                else:
                    logger.error(f"Metrics API failed: {resp.status}")
        except Exception as e:
            logger.error(f"Metrics API error: {e}")

        # Test Ingest
        test_obs = {
            "observation": {
                "symbol": "TEST",
                "price": 100.0,
                "volume": 5000,
                "timestamp": 123456789
            },
            "domain": "test_domain"
        }
        try:
            async with session.post(f"{url}/api/ingest", json=test_obs) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Ingest API OK: {result}")
                else:
                    logger.error(f"Ingest API failed: {resp.status}")
        except Exception as e:
            logger.error(f"Ingest API error: {e}")

        # Test State
        try:
            async with session.get(f"{url}/api/state") as resp:
                if resp.status == 200:
                    state = await resp.json()
                    logger.info(f"State API OK: Found {len(state.get('concepts', {}))} concepts")
                else:
                    logger.error(f"State API failed: {resp.status}")
        except Exception as e:
            logger.error(f"State API error: {e}")

if __name__ == "__main__":
    # This script assumes the mesh is running on port 8080
    asyncio.run(test_api())
