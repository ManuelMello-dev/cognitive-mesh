"""
Redis Cache Layer for Low-Latency Gossip State and Real-Time Data
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger("RedisCache")


class RedisCache:
    """
    Async Redis connector for:
    - Caching gossip state and seen messages
    - Real-time tick data
    - Node discovery and peer lists
    """
    
    def __init__(self, host: str = None, port: int = 6379, db: int = 0):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port
        self.db = db
        self.redis = None
        self.connected = False
    
    async def connect(self):
        """Initialize Redis connection"""
        try:
            import aioredis
            self.redis = await aioredis.create_redis_pool(
                f"redis://{self.host}:{self.port}/{self.db}"
            )
            self.connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except ImportError:
            logger.warning("aioredis not installed. Install with: pip install aioredis")
            self.connected = False
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.connected = False
    
    async def set_seen_message(self, message_id: str, ttl: int = 3600) -> bool:
        """Cache a seen message ID (for deduplication)"""
        if not self.connected:
            return False
        
        try:
            await self.redis.setex(f"seen:{message_id}", ttl, "1")
            return True
        except Exception as e:
            logger.error(f"Error setting seen message: {e}")
            return False
    
    async def is_message_seen(self, message_id: str) -> bool:
        """Check if message has been seen"""
        if not self.connected:
            return False
        
        try:
            result = await self.redis.get(f"seen:{message_id}")
            return result is not None
        except Exception as e:
            logger.error(f"Error checking seen message: {e}")
            return False
    
    async def cache_tick(self, symbol: str, tick: Dict[str, Any], ttl: int = 60) -> bool:
        """Cache the latest tick data"""
        if not self.connected:
            return False
        
        try:
            key = f"tick:{symbol}"
            value = json.dumps(tick)
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Error caching tick: {e}")
            return False
    
    async def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached tick data"""
        if not self.connected:
            return None
        
        try:
            key = f"tick:{symbol}"
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error retrieving tick: {e}")
            return None
    
    async def register_peer(self, node_id: str, address: str, ttl: int = 300) -> bool:
        """Register a peer node for discovery"""
        if not self.connected:
            return False
        
        try:
            key = f"peer:{node_id}"
            await self.redis.setex(key, ttl, address)
            return True
        except Exception as e:
            logger.error(f"Error registering peer: {e}")
            return False
    
    async def get_peers(self) -> List[str]:
        """Get all registered peers"""
        if not self.connected:
            return []
        
        try:
            keys = await self.redis.keys("peer:*")
            peers = []
            for key in keys:
                value = await self.redis.get(key)
                if value:
                    peers.append(value.decode() if isinstance(value, bytes) else value)
            return peers
        except Exception as e:
            logger.error(f"Error getting peers: {e}")
            return []
    
    async def cache_concept(self, concept_id: str, concept: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache a concept for quick retrieval"""
        if not self.connected:
            return False
        
        try:
            key = f"concept:{concept_id}"
            value = json.dumps(concept)
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Error caching concept: {e}")
            return False
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached concept"""
        if not self.connected:
            return None
        
        try:
            key = f"concept:{concept_id}"
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error retrieving concept: {e}")
            return None
    
    async def increment_counter(self, counter_name: str) -> int:
        """Increment a counter (for metrics)"""
        if not self.connected:
            return 0
        
        try:
            key = f"counter:{counter_name}"
            value = await self.redis.incr(key)
            return value
        except Exception as e:
            logger.error(f"Error incrementing counter: {e}")
            return 0
    
    async def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        if not self.connected:
            return 0
        
        try:
            key = f"counter:{counter_name}"
            value = await self.redis.get(key)
            if value:
                return int(value)
            return 0
        except Exception as e:
            logger.error(f"Error getting counter: {e}")
            return 0
    
    async def set_gossip_state(self, state: Dict[str, Any], ttl: int = 300) -> bool:
        """Cache the current gossip state"""
        if not self.connected:
            return False
        
        try:
            key = "gossip:state"
            value = json.dumps(state)
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Error setting gossip state: {e}")
            return False
    
    async def get_gossip_state(self) -> Optional[Dict[str, Any]]:
        """Retrieve cached gossip state"""
        if not self.connected:
            return None
        
        try:
            key = "gossip:state"
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error retrieving gossip state: {e}")
            return None
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            self.connected = False
            logger.info("Disconnected from Redis")
