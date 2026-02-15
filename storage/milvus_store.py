"""
Milvus Vector Store for High-Dimensional Concept Search
Enables efficient similarity search across millions of concepts
"""

import logging
import os
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger("MilvusStore")


class MilvusStore:
    """
    Async Milvus connector for vector-based concept storage and retrieval.
    Stores concept signatures as vectors for fast similarity search.
    """
    
    def __init__(self, host: str = None, port: int = 19530):
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port
        self.collection_name = "concepts"
        self.client = None
        self.connected = False
    
    async def connect(self):
        """Initialize Milvus connection"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            self.connections = connections
            self.Collection = Collection
            self.FieldSchema = FieldSchema
            self.CollectionSchema = CollectionSchema
            self.DataType = DataType
            
            # Connect to Milvus
            self.connections.connect("default", host=self.host, port=self.port)
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
            await self._init_collection()
        except ImportError:
            logger.warning("pymilvus not installed. Install with: pip install pymilvus")
            self.connected = False
        except Exception as e:
            logger.error(f"Milvus connection error: {e}")
            self.connected = False
    
    async def _init_collection(self):
        """Create collection if it doesn't exist"""
        if not self.connected:
            return
        
        try:
            # Check if collection exists
            if self.Collection(self.collection_name).num_entities == 0:
                # Define schema
                fields = [
                    self.FieldSchema(name="id", dtype=self.DataType.VARCHAR, max_length=256, is_primary=True),
                    self.FieldSchema(name="domain", dtype=self.DataType.VARCHAR, max_length=256),
                    self.FieldSchema(name="vector", dtype=self.DataType.FLOAT_VECTOR, dim=128),
                    self.FieldSchema(name="confidence", dtype=self.DataType.FLOAT),
                    self.FieldSchema(name="timestamp", dtype=self.DataType.INT64),
                ]
                
                schema = self.CollectionSchema(fields, "Cognitive concept vectors")
                collection = self.Collection(self.collection_name, schema)
                
                # Create index
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index("vector", index_params)
                logger.info("Milvus collection created")
        except Exception as e:
            logger.error(f"Collection initialization error: {e}")
    
    async def insert_concept(self, concept_id: str, domain: str, signature: Dict[str, float], 
                            confidence: float, timestamp: float) -> bool:
        """Insert a concept vector into Milvus"""
        if not self.connected:
            return False
        
        try:
            # Convert signature dict to vector (128-dim)
            vector = self._signature_to_vector(signature)
            
            collection = self.Collection(self.collection_name)
            collection.insert([
                [concept_id],
                [domain],
                [vector],
                [confidence],
                [int(timestamp)]
            ])
            
            logger.debug(f"Inserted concept {concept_id} into Milvus")
            return True
        except Exception as e:
            logger.error(f"Error inserting concept: {e}")
            return False
    
    async def search_similar_concepts(self, signature: Dict[str, float], 
                                     domain: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar concepts using vector similarity"""
        if not self.connected:
            return []
        
        try:
            vector = self._signature_to_vector(signature)
            
            collection = self.Collection(self.collection_name)
            
            # Build filter if domain is specified
            expr = f"domain == '{domain}'" if domain else None
            
            results = collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                expr=expr
            )
            
            similar_concepts = []
            for hits in results:
                for hit in hits:
                    similar_concepts.append({
                        "id": hit.entity.get("id"),
                        "domain": hit.entity.get("domain"),
                        "confidence": hit.entity.get("confidence"),
                        "distance": hit.distance
                    })
            
            logger.debug(f"Found {len(similar_concepts)} similar concepts")
            return similar_concepts
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []
    
    async def update_concept_confidence(self, concept_id: str, new_confidence: float) -> bool:
        """Update confidence score of a concept"""
        if not self.connected:
            return False
        
        try:
            collection = self.Collection(self.collection_name)
            collection.update([{"id": concept_id, "confidence": new_confidence}])
            logger.debug(f"Updated concept {concept_id} confidence to {new_confidence}")
            return True
        except Exception as e:
            logger.error(f"Error updating concept: {e}")
            return False
    
    async def get_concept_stats(self) -> Dict[str, Any]:
        """Get statistics about stored concepts"""
        if not self.connected:
            return {}
        
        try:
            collection = self.Collection(self.collection_name)
            return {
                "total_concepts": collection.num_entities,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def _signature_to_vector(self, signature: Dict[str, float], dim: int = 128) -> List[float]:
        """Convert signature dict to fixed-dimension vector"""
        # Simple approach: hash keys and create a sparse vector
        vector = [0.0] * dim
        
        for key, value in signature.items():
            # Hash the key to get an index
            hash_val = hash(key) % dim
            vector[hash_val] += value
        
        # Normalize
        magnitude = sum(v**2 for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        
        return vector
    
    async def disconnect(self):
        """Close Milvus connection"""
        if self.connected:
            try:
                self.connections.disconnect("default")
                self.connected = False
                logger.info("Disconnected from Milvus")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
