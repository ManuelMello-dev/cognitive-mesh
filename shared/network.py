import zmq
import zmq.asyncio
import json
import logging
from typing import Any, Callable

logger = logging.getLogger("NetworkLayer")

class ZMQNode:
    def __init__(self, identity: str, port: int = 5555):
        self.identity = identity
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt_string(zmq.IDENTITY, identity)
        
    async def start(self):
        self.socket.bind(f"tcp://*:{self.port}")
        logger.info(f"Node {self.identity} listening on port {self.port}")
        
    async def send(self, receiver: str, message: Any):
        await self.socket.send_multipart([
            receiver.encode(),
            json.dumps(message).encode()
        ])

    async def listen(self, callback: Callable):
        while True:
            sender, msg_payload = await self.socket.recv_multipart()
            message = json.loads(msg_payload.decode())
            await callback(sender.decode(), message)

class ZMQAgent:
    def __init__(self, identity: str, server_addr: str):
        self.identity = identity
        self.server_addr = server_addr
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, identity)

    async def connect(self):
        self.socket.connect(self.server_addr)
        logger.info(f"Agent {self.identity} connected to {self.server_addr}")

    async def send_observation(self, obs: Any):
        await self.socket.send_json(obs)
