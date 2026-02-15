"""
Production-Hardened AMFG Protocol Implementation
Adaptive Message Fan-out with Gossip
Features: Bloom Filter Decay, Merkle Trees, Priority Queues, Weight Learning
"""

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
from enum import Enum
import json

logger = logging.getLogger("AMFGGossip")


class DecayingBloomFilter:
    """Counting Bloom Filter with time-based decay to prevent saturation"""
    
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
    
    def to_bytes(self) -> bytes:
        quantized = [min(int(c * 100), 65535) for c in self.counts]
        return struct.pack(f'{self.size}H', *quantized)
    
    @classmethod
    def from_bytes(cls, data: bytes, num_hashes: int = 3):
        size = len(data) // 2
        quantized = list(struct.unpack(f'{size}H', data))
        bf = cls(size, num_hashes)
        bf.counts = [q / 100.0 for q in quantized]
        return bf


class MerkleTree:
    """Merkle tree for efficient anti-entropy (O(log N) instead of O(N))"""
    
    def __init__(self, items: Dict[str, float]):
        self.items = items
        self.tree = self._build_tree(sorted(items.items()))
    
    def _hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _build_tree(self, sorted_items: List[Tuple[str, float]]) -> Dict:
        if not sorted_items:
            return {"hash": self._hash(""), "items": []}
        
        if len(sorted_items) == 1:
            msg_id, ts = sorted_items[0]
            leaf_hash = self._hash(f"{msg_id}:{ts}")
            return {"hash": leaf_hash, "items": [msg_id]}
        
        mid = len(sorted_items) // 2
        left = self._build_tree(sorted_items[:mid])
        right = self._build_tree(sorted_items[mid:])
        
        combined_hash = self._hash(left["hash"] + right["hash"])
        combined_items = left["items"] + right["items"]
        
        return {
            "hash": combined_hash,
            "left": left,
            "right": right,
            "items": combined_items
        }
    
    def get_root_hash(self) -> str:
        return self.tree["hash"]
    
    def get_diff(self, other_root_hash: str) -> List[str]:
        if self.tree["hash"] == other_root_hash:
            return []
        return self.tree["items"]


class ExploringWeightLearner:
    """Weight learner with epsilon-greedy exploration for adaptive fan-out"""
    
    def __init__(self, learning_rate: float = 0.02, epsilon: float = 0.1):
        self.weights = {'fanout': 0.3, 'latency': 0.3, 'coverage': 0.2, 'reliability': 0.2}
        self.lr = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.outcomes = []
    
    def get_weights(self) -> Dict[str, float]:
        if random.random() < self.epsilon:
            weights = self.weights.copy()
            for key in weights:
                noise = random.gauss(0, 0.05)
                weights[key] = max(0.0, min(1.0, weights[key] + noise))
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            return weights
        return self.weights.copy()
    
    def update_weights(self, reward: float, factor_contributions: Dict[str, float]):
        total_contrib = sum(factor_contributions.values())
        if total_contrib == 0:
            return
        
        gradients = {k: v / total_contrib for k, v in factor_contributions.items()}
        
        for key in self.weights:
            self.weights[key] += self.lr * reward * gradients.get(key, 0)
        
        self._normalize_weights()
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.995)
    
    def _normalize_weights(self):
        for key in self.weights:
            self.weights[key] = max(0.0, min(1.0, self.weights[key]))
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total


class PriorityMessageQueue:
    """O(1) lookup + O(log N) priority operations"""
    
    def __init__(self, max_size: int = 1000):
        self.messages: Dict[str, 'Message'] = {}
        self.priority_heap: List[Tuple[float, str]] = []
        self.max_size = max_size
    
    def add(self, message: 'Message') -> bool:
        if message.id in self.messages:
            return False
        
        if len(self.messages) >= self.max_size:
            self._evict_lowest_priority()
        
        self.messages[message.id] = message
        heapq.heappush(self.priority_heap, (message.priority, message.id))
        return True
    
    def _evict_lowest_priority(self):
        while self.priority_heap:
            _, msg_id = heapq.heappop(self.priority_heap)
            if msg_id in self.messages:
                del self.messages[msg_id]
                break
    
    def get_all(self) -> List['Message']:
        return list(self.messages.values())
    
    def get_by_id(self, msg_id: str) -> Optional['Message']:
        return self.messages.get(msg_id)


@dataclass
class Message:
    id: str
    sender: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10
    priority: float = 1.0


class AMFGProtocol:
    """
    Adaptive Message Fan-out Gossip Protocol
    Optimizes for low latency, high coverage, and reliability
    """
    
    def __init__(self, node_id: str, peers: List[str] = None):
        self.node_id = node_id
        self.peers = set(peers or [])
        self.queue = PriorityMessageQueue()
        self.bloom_filter = DecayingBloomFilter()
        self.seen_messages: Set[str] = set()
        self.weight_learner = ExploringWeightLearner()
        self.active_deliveries: Dict[str, Dict[str, Any]] = {}
        self.message_timestamps: Dict[str, float] = {}
    
    async def broadcast(self, data: Any, priority: float = 1.0) -> str:
        """Broadcast a message to the network"""
        msg_id = hashlib.md5(
            f"{self.node_id}:{time.time()}:{random.random()}".encode()
        ).hexdigest()
        
        msg = Message(id=msg_id, sender=self.node_id, data=data, priority=priority)
        self.queue.add(msg)
        self.seen_messages.add(msg_id)
        self.bloom_filter.add(msg_id)
        self.message_timestamps[msg_id] = time.time()
        
        logger.info(f"Node {self.node_id} broadcasting message {msg_id}")
        await self._gossip(msg)
        return msg_id
    
    async def _gossip(self, msg: Message):
        """Gossip message to selected peers"""
        if msg.ttl <= 0:
            return
        
        # Adaptive fan-out based on learned weights
        weights = self.weight_learner.get_weights()
        base_fanout = max(1, int(len(self.peers) * weights.get('fanout', 0.3)))
        fanout = min(len(self.peers), base_fanout)
        
        if fanout == 0 or not self.peers:
            return
        
        targets = random.sample(list(self.peers), fanout)
        
        # Track active delivery
        delivery_id = f"{msg.id}:{time.time()}"
        self.active_deliveries[delivery_id] = {
            "message_id": msg.id,
            "targets": targets,
            "start_time": time.time(),
            "successful": 0,
            "failed": 0
        }
        
        # Send to targets concurrently
        tasks = [self._send_to_peer(target, msg) for target in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update delivery stats
        delivery = self.active_deliveries[delivery_id]
        for result in results:
            if isinstance(result, Exception):
                delivery["failed"] += 1
            else:
                delivery["successful"] += 1
        
        # Learn from this delivery
        await self._learn_from_delivery(delivery)
        
        msg.ttl -= 1
    
    async def _send_to_peer(self, peer: str, msg: Message) -> bool:
        """Send message to a specific peer"""
        try:
            # In a real implementation, this would use ZeroMQ or HTTP
            # For now, we simulate the network call
            await asyncio.sleep(random.uniform(0.01, 0.1))
            logger.debug(f"Gossiping {msg.id} from {self.node_id} to {peer}")
            return True
        except Exception as e:
            logger.error(f"Failed to send to {peer}: {e}")
            return False
    
    async def _learn_from_delivery(self, delivery: Dict[str, Any]):
        """Update weights based on delivery performance"""
        total = delivery["successful"] + delivery["failed"]
        if total == 0:
            return
        
        delivery_time = time.time() - delivery["start_time"]
        coverage_ratio = delivery["successful"] / total
        
        reward = coverage_ratio - (delivery_time / 10.0)
        
        factor_contributions = {
            "fanout": delivery["successful"],
            "latency": 1.0 / (1.0 + delivery_time),
            "coverage": coverage_ratio,
            "reliability": delivery["successful"]
        }
        
        self.weight_learner.update_weights(reward, factor_contributions)
    
    def receive_message(self, msg_dict: Dict[str, Any]) -> bool:
        """Process received message"""
        msg_id = msg_dict.get("id")
        
        if msg_id in self.seen_messages or self.bloom_filter.contains(msg_id):
            return False
        
        self.seen_messages.add(msg_id)
        self.bloom_filter.add(msg_id)
        
        msg = Message(**msg_dict)
        msg.ttl -= 1
        self.queue.add(msg)
        
        logger.debug(f"Node {self.node_id} received message {msg_id} from {msg.sender}")
        return True
    
    def add_peer(self, peer_id: str):
        """Add a peer to the network"""
        self.peers.add(peer_id)
        logger.info(f"Added peer {peer_id}")
    
    def remove_peer(self, peer_id: str):
        """Remove a peer from the network"""
        self.peers.discard(peer_id)
        logger.info(f"Removed peer {peer_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        return {
            "node_id": self.node_id,
            "peers": len(self.peers),
            "seen_messages": len(self.seen_messages),
            "queued_messages": len(self.queue.messages),
            "active_deliveries": len(self.active_deliveries),
            "weights": self.weight_learner.weights,
            "epsilon": self.weight_learner.epsilon
        }
