"""
PostgreSQL/YugabyteDB Persistence Layer for Rules, Concepts, and Metadata
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("PostgresStore")


class PostgresStore:
    """
    Async PostgreSQL connector for persistent storage of:
    - Rules and their confidence scores
    - Concept metadata and temporal metrics
    - System metrics and audit logs
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/cognitive_mesh"
        )
        self.pool = None
        self.connected = False
    
    async def connect(self):
        """Initialize connection pool"""
        try:
            import asyncpg
            self.pool = await asyncpg.create_pool(self.connection_string)
            self.connected = True
            logger.info("Connected to PostgreSQL")
            await self._init_schema()
        except ImportError:
            logger.warning("asyncpg not installed. Install with: pip install asyncpg")
            self.connected = False
    
    async def _init_schema(self):
        """Create tables if they don't exist"""
        if not self.pool:
            return
        
        schema = """
        -- Rules table
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            antecedent TEXT NOT NULL,
            consequent TEXT NOT NULL,
            confidence FLOAT NOT NULL,
            support INT DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            observation_span_hours FLOAT DEFAULT 0.0,
            domain TEXT,
            metadata JSONB
        );
        
        -- Concepts table
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            domain TEXT NOT NULL,
            signature JSONB NOT NULL,
            confidence FLOAT NOT NULL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            observation_count INT DEFAULT 0,
            observation_span_hours FLOAT DEFAULT 0.0,
            distinct_time_windows INT DEFAULT 0,
            metadata JSONB
        );
        
        -- Observations table (time-series)
        CREATE TABLE IF NOT EXISTS observations (
            id SERIAL PRIMARY KEY,
            concept_id TEXT REFERENCES concepts(id),
            symbol TEXT,
            price FLOAT,
            volume FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );
        
        -- System metrics
        CREATE TABLE IF NOT EXISTS metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            concepts_formed INT,
            concepts_decayed INT,
            rules_learned INT,
            transfers_made INT,
            goals_generated INT,
            total_observations INT,
            errors INT,
            uptime_seconds FLOAT,
            metadata JSONB
        );
        
        -- Gossip events (for audit trail)
        CREATE TABLE IF NOT EXISTS gossip_events (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            event_type TEXT,
            data JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ttl INT DEFAULT 10
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_rules_domain ON rules(domain);
        CREATE INDEX IF NOT EXISTS idx_rules_confidence ON rules(confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain);
        CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON concepts(confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON observations(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_observations_symbol ON observations(symbol);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_gossip_timestamp ON gossip_events(timestamp DESC);
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(schema)
            logger.info("Schema initialized")
        except Exception as e:
            logger.error(f"Schema initialization error: {e}")
    
    async def save_rule(self, rule_id: str, rule: Dict[str, Any]):
        """Save a rule to the database"""
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rules (id, antecedent, consequent, confidence, support, domain, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE SET
                        confidence = $4,
                        support = support + 1,
                        last_seen = CURRENT_TIMESTAMP,
                        metadata = $7
                """, 
                rule_id,
                rule.get('antecedent'),
                rule.get('consequent'),
                rule.get('confidence', 0.0),
                rule.get('support', 1),
                rule.get('domain'),
                json.dumps(rule.get('metadata', {}))
                )
        except Exception as e:
            logger.error(f"Error saving rule: {e}")
    
    async def save_concept(self, concept_id: str, concept: Dict[str, Any]):
        """Save a concept to the database"""
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO concepts (id, domain, signature, confidence, first_seen, last_seen, 
                                        observation_count, observation_span_hours, distinct_time_windows, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE SET
                        confidence = $4,
                        last_seen = $6,
                        observation_count = $7,
                        observation_span_hours = $8,
                        distinct_time_windows = $9,
                        metadata = $10
                """,
                concept_id,
                concept.get('domain'),
                json.dumps(concept.get('signature', {})),
                concept.get('confidence', 0.0),
                datetime.fromtimestamp(concept.get('first_seen', 0)),
                datetime.fromtimestamp(concept.get('last_seen', 0)),
                concept.get('observation_count', 0),
                concept.get('observation_span_hours', 0.0),
                concept.get('distinct_time_windows', 0),
                json.dumps(concept.get('metadata', {}))
                )
        except Exception as e:
            logger.error(f"Error saving concept: {e}")
    
    async def save_observation(self, observation: Dict[str, Any]):
        """Save an observation (tick data)"""
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO observations (concept_id, symbol, price, volume, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                observation.get('concept_id'),
                observation.get('symbol'),
                observation.get('price'),
                observation.get('volume'),
                json.dumps(observation.get('metadata', {}))
                )
        except Exception as e:
            logger.error(f"Error saving observation: {e}")
    
    async def save_metrics(self, metrics: Dict[str, Any]):
        """Save system metrics"""
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO metrics (concepts_formed, concepts_decayed, rules_learned, 
                                       transfers_made, goals_generated, total_observations, 
                                       errors, uptime_seconds, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                metrics.get('concepts_formed', 0),
                metrics.get('concepts_decayed', 0),
                metrics.get('rules_learned', 0),
                metrics.get('transfers_made', 0),
                metrics.get('goals_generated', 0),
                metrics.get('total_observations', 0),
                metrics.get('errors', 0),
                metrics.get('uptime_seconds', 0.0),
                json.dumps(metrics.get('metadata', {}))
                )
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    async def save_gossip_event(self, event_id: str, node_id: str, event_type: str, data: Dict[str, Any]):
        """Save a gossip event for audit trail"""
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO gossip_events (id, node_id, event_type, data)
                    VALUES ($1, $2, $3, $4)
                """,
                event_id,
                node_id,
                event_type,
                json.dumps(data)
                )
        except Exception as e:
            logger.error(f"Error saving gossip event: {e}")
    
    async def get_high_confidence_rules(self, threshold: float = 0.7, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve high-confidence rules"""
        if not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, antecedent, consequent, confidence, support, domain
                    FROM rules
                    WHERE confidence >= $1
                    ORDER BY confidence DESC
                    LIMIT $2
                """, threshold, limit)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving rules: {e}")
            return []
    
    async def get_recent_observations(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent observations for a symbol"""
        if not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT symbol, price, volume, timestamp, metadata
                    FROM observations
                    WHERE symbol = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, symbol, limit)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving observations: {e}")
            return []
    
    async def cleanup_old_gossip_events(self, days: int = 7):
        """Clean up old gossip events"""
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM gossip_events
                    WHERE timestamp < NOW() - INTERVAL '%d days'
                """ % days)
        except Exception as e:
            logger.error(f"Error cleaning up gossip events: {e}")
    
    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.connected = False
            logger.info("Disconnected from PostgreSQL")
