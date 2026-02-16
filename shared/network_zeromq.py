"""
ZeroMQ-based Communication Layer for Distributed Cognitive Mesh
Low-latency, high-throughput message passing between nodes
"""

import zmq
import zmq.asyncio
import json
import logging
import asyncio
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ZeroMQNetwork")


class ZMQNode:
    """
    Router node for the cognitive mesh.
    Receives messages from agents and broadcasts to other nodes.
    """
    
    def __init__(self, identity: str, port: int = 5555):
        self.identity = identity
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.running = False
        self.message_handlers: Dict[str, Callable] = {}
    
    async def start(self):
        """Start the router node"""
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.identity)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True
        logger.info(f"ZeroMQ Node {self.identity} listening on port {self.port}")
    
    async def send(self, receiver: str, message: Dict[str, Any]):
        """Send a message to a specific receiver"""
        if not self.socket:
            return
        
        try:
            await self.socket.send_multipart([
                receiver.encode(),
                json.dumps(message).encode()
            ])
            logger.debug(f"Sent message to {receiver}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: Dict[str, Any], receivers: List[str]):
        """Broadcast a message to multiple receivers"""
        tasks = [self.send(receiver, message) for receiver in receivers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def listen(self, callback: Callable):
        """Listen for incoming messages"""
        while self.running:
            try:
                sender, msg_payload = await self.socket.recv_multipart()
                sender_id = sender.decode()
                
                try:
                    message = json.loads(msg_payload.decode())
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {sender_id}")
                    continue
                
                # Call registered handler or default callback
                handler = self.message_handlers.get(message.get("type"), callback)
                await handler(sender_id, message)
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(0.1)
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def stop(self):
        """Stop the node"""
        self.running = False
        if self.socket:
            self.socket.close()
        logger.info(f"ZeroMQ Node {self.identity} stopped")
        
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Non-blocking receive for the orchestrator loop"""
        if not self.socket: return None
        try:
            # Try a non-blocking receive
            frames = await self.socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(frames) >= 2:
                return json.loads(frames[1].decode())
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Receive error: {e}")
        return None
        
    async def broadcast_gossip(self, state: Dict[str, Any]):
        """Broadcast state to the network"""
        # In a real mesh, this would send to known peers
        # For now, we'll just log that gossip is happening
        logger.debug(f"Broadcasting gossip state: {state.get('node_id')}")


class ZMQAgent:
    """
    Agent node that connects to a router.
    Used by data providers to send observations to the central core.
    """
    
    def __init__(self, identity: str, server_addr: str):
        self.identity = identity
        self.server_addr = server_addr
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.connected = False
    
    async def connect(self):
        """Connect to the router"""
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.identity)
        self.socket.connect(self.server_addr)
        self.connected = True
        logger.info(f"Agent {self.identity} connected to {self.server_addr}")
    
    async def send_observation(self, obs: Dict[str, Any]):
        """Send an observation to the router"""
        if not self.socket or not self.connected:
            raise Exception("Agent not connected")
        
        try:
            await self.socket.send_json(obs)
            logger.debug(f"Agent {self.identity} sent observation")
        except Exception as e:
            logger.error(f"Error sending observation: {e}")
            raise
    
    async def send_batch(self, observations: List[Dict[str, Any]]):
        """Send multiple observations"""
        tasks = [self.send_observation(obs) for obs in observations]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def disconnect(self):
        """Disconnect from the router"""
        if self.socket:
            self.socket.close()
        self.connected = False
        logger.info(f"Agent {self.identity} disconnected")


class ZMQPubSub:
    """
    Pub/Sub pattern for broadcasting high-priority events
    (e.g., new concepts, high-confidence rules)
    """
    
    def __init__(self, identity: str, pub_port: int = 5556, sub_port: int = 5557):
        self.identity = identity
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.context = zmq.asyncio.Context()
        self.pub_socket = None
        self.sub_socket = None
        self.subscriptions: Dict[str, Callable] = {}
    
    async def start_publisher(self):
        """Start the publisher socket"""
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{self.pub_port}")
        logger.info(f"PubSub Publisher {self.identity} listening on port {self.pub_port}")
    
    async def start_subscriber(self, topics: List[str] = None):
        """Start the subscriber socket"""
        self.sub_socket = self.context.socket(zmq.SUB)
        
        # Subscribe to topics
        for topic in (topics or ["*"]):
            self.sub_socket.subscribe(topic.encode())
        
        logger.info(f"PubSub Subscriber {self.identity} subscribed to {topics}")
    
    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message on a topic"""
        if not self.pub_socket:
            return
        
        try:
            await self.pub_socket.send_multipart([
                topic.encode(),
                json.dumps(message).encode()
            ])
            logger.debug(f"Published to {topic}")
        except Exception as e:
            logger.error(f"Error publishing: {e}")
    
    async def listen(self):
        """Listen for published messages"""
        if not self.sub_socket:
            return
        
        while True:
            try:
                topic, msg_payload = await self.sub_socket.recv_multipart()
                topic_str = topic.decode()
                
                try:
                    message = json.loads(msg_payload.decode())
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on topic {topic_str}")
                    continue
                
                # Call registered callback
                handler = self.subscriptions.get(topic_str)
                if handler:
                    await handler(message)
            except Exception as e:
                logger.error(f"Error in subscriber: {e}")
                await asyncio.sleep(0.1)
    
    def register_callback(self, topic: str, callback: Callable):
        """Register a callback for a topic"""
        self.subscriptions[topic] = callback
    
    async def stop(self):
        """Stop publisher and subscriber"""
        if self.pub_socket:
            self.pub_socket.close()
        if self.sub_socket:
            self.sub_socket.close()
        logger.info(f"PubSub {self.identity} stopped")
        
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Non-blocking receive for the subscriber"""
        if not self.sub_socket: return None
        try:
            frames = await self.sub_socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(frames) >= 2:
                return json.loads(frames[1].decode())
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"PubSub receive error: {e}")
        return None
