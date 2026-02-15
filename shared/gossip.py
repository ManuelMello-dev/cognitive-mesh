import asyncio
import time
import random
import logging
import mmh3
import struct
import hashlib
import heapq
import math
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import msgpack
import httpx

logger = logging.getLogger("CognitiveGossip")

@dataclass
class Message:
    id: str
    sender: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10
    priority: float = 0.0

class DecayingBloomFilter:
    def __init__(self, size: int = 512, num_hashes: int = 3, decay_rate: float = 0.99):
        self.size = size
        self.num_hashes = num_hashes
        self.decay_rate = decay_rate
        self.counts = [0.0] * size
        self.last_decay = time.time()
        self.decay_interval = 60.0
    
    def _maybe_decay(self):
        now = time.time()
        if now - self.last_decay > self.decay_interval:
            for i in range(self.size):
                self.counts[i] *= self.decay_rate
                if self.counts[i] < 0.01:
                    self.counts[i] = 0.0
            self.last_decay = now
    
    def add(self, item: str, weight: float = 1.0):
        self._maybe_decay()
        for i in range(self.num_hashes):
            idx = mmh3.hash(item, i) % self.size
            self.counts[idx] += weight
    
    def contains(self, item: str, threshold: float = 0.5) -> bool:
        self._maybe_decay()
        for i in range(self.num_hashes):
            idx = mmh3.hash(item, i) % self.size
            if self.counts[idx] < threshold:
                return False
        return True

class PriorityMessageQueue:
    def __init__(self, max_size: int = 1000):
        self.messages: Dict[str, Message] = {}
        self.priority_heap: List[Tuple[float, str]] = []
        self.max_size = max_size
    
    def add(self, message: Message):
        if message.id in self.messages:
            return False
        if len(self.messages) >= self.max_size:
            self._evict_lowest_priority()
        self.messages[message.id] = message
        heapq.heappush(self.priority_heap, (message.priority, message.id))
        return True
    
    def _evict_lowest_priority(self):
        if not self.priority_heap:
            return
        _, msg_id = heapq.heappop(self.priority_heap)
        self.messages.pop(msg_id, None)

    def get_all(self) -> List[Message]:
        return list(self.messages.values())

class AMFGProtocol:
    """
    Implementation of the user's Production-Hardened AMFG Protocol.
    """
    def __init__(self, node_id: str, peers: List[str] = None):
        self.node_id = node_id
        self.peers = set(peers or [])
        self.queue = PriorityMessageQueue()
        self.bloom_filter = DecayingBloomFilter()
        self.seen_messages = set()
        
    async def broadcast(self, data: Any, priority: float = 1.0):
        msg_id = hashlib.md5(f"{self.node_id}:{time.time()}:{random.random()}".encode()).hexdigest()
        msg = Message(id=msg_id, sender=self.node_id, data=data, priority=priority)
        self.queue.add(msg)
        self.seen_messages.add(msg_id)
        logger.info(f"Node {self.node_id} broadcasting message {msg_id}")
        await self._gossip(msg)

    async def _gossip(self, msg: Message):
        if msg.ttl <= 0:
            return
        
        # Select subset of peers to gossip to (Fan-out)
        fan_out = min(len(self.peers), 3)
        targets = random.sample(list(self.peers), fan_out) if self.peers else []
        
        for target in targets:
            # In a real ZeroMQ/HTTP implementation, we'd send the message here
            logger.debug(f"Gossiping {msg.id} from {self.node_id} to {target}")
            # Placeholder for actual network call
            pass

    def receive_message(self, msg_dict: Dict[str, Any]):
        msg = Message(**msg_dict)
        if msg.id in self.seen_messages:
            return False
        
        self.seen_messages.add(msg.id)
        msg.ttl -= 1
        self.queue.add(msg)
        return True
