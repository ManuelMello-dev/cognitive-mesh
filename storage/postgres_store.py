"""
PostgreSQL/YugabyteDB Persistence Layer for Rules, Concepts, and Metadata
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict, deque

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
        """Initialize connection pool with deployment-friendly defaults."""
        try:
            import asyncpg

            min_size = int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1"))
            max_size = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "3"))
            command_timeout = float(os.getenv("POSTGRES_COMMAND_TIMEOUT", "30"))
            if max_size < min_size:
                max_size = min_size

            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=min_size,
                max_size=max_size,
                command_timeout=command_timeout,
            )
            self.connected = True
            logger.info(
                "Connected to PostgreSQL (pool min=%s max=%s command_timeout=%ss)",
                min_size,
                max_size,
                command_timeout,
            )
            await self._init_schema()
        except ImportError:
            logger.warning("asyncpg not installed. Install with: pip install asyncpg")
            self.connected = False
        except asyncio.CancelledError:
            await self._close_partial_pool()
            self.connected = False
            raise
        except Exception:
            await self._close_partial_pool()
            self.connected = False
            raise

    async def _close_partial_pool(self):
        """Best-effort cleanup when startup is cancelled mid-connection."""
        if not self.pool:
            return
        try:
            await asyncio.wait_for(self.pool.close(), timeout=5.0)
        except Exception:
            try:
                self.pool.terminate()
            except Exception:
                pass
        finally:
            self.pool = None
    
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
        
        -- Goals table
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            description TEXT NOT NULL,
            success_criteria JSONB,
            priority FLOAT,
            status TEXT,
            progress FLOAT,
            value_estimate FLOAT,
            created_at TIMESTAMP,
            deadline TIMESTAMP,
            parent_goal TEXT,
            sub_goals JSONB,
            attempts INT,
            last_attempt TIMESTAMP,
            achieved_at TIMESTAMP
        );

        -- Prediction Engine State table
        CREATE TABLE IF NOT EXISTS prediction_engine_state (
            id TEXT PRIMARY KEY,
            state_data JSONB
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
        CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
        CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals(priority DESC);

        -- Facts table
        CREATE TABLE IF NOT EXISTS facts (
            fact TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Cross-domain mappings table
        CREATE TABLE IF NOT EXISTS cross_domain_mappings (
            mapping_id TEXT PRIMARY KEY,
            source_domain TEXT NOT NULL,
            target_domain TEXT NOT NULL,
            concept_mappings JSONB,
            confidence FLOAT,
            bidirectional BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Observation history table
        CREATE TABLE IF NOT EXISTS observation_history (
            id SERIAL PRIMARY KEY,
            observation JSONB NOT NULL,
            domain TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Price history table
        CREATE TABLE IF NOT EXISTS price_history (
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            price FLOAT NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        );

        -- Meta domains table
        CREATE TABLE IF NOT EXISTS meta_domains (
            meta_domain TEXT PRIMARY KEY,
            domains JSONB NOT NULL
        );

        -- Learning engine state table
        CREATE TABLE IF NOT EXISTS learning_engine_state (
            id TEXT PRIMARY KEY,
            state_data JSONB
        );

        -- Caches table
        CREATE TABLE IF NOT EXISTS caches (
            cache_name TEXT PRIMARY KEY,
            data JSONB
        );

        -- Short-term memory table (learning engine rolling window + observation count)
        CREATE TABLE IF NOT EXISTS short_term_memory (
            id TEXT PRIMARY KEY,
            data JSONB NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Public Z3 organism-state persistence
        CREATE TABLE IF NOT EXISTS z3_baselines (
            baseline_id TEXT PRIMARY KEY,
            version INT NOT NULL,
            state JSONB NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS z3_novelty_events (
            event_id TEXT PRIMARY KEY,
            baseline_version INT,
            source TEXT,
            signal_type TEXT,
            novelty_score FLOAT,
            severity TEXT,
            event JSONB NOT NULL,
            observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS z3_decisions (
            decision_id TEXT PRIMARY KEY,
            baseline_version INT,
            action TEXT,
            linked_event_id TEXT,
            confidence FLOAT,
            decision JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS z3_transitions (
            transition_id TEXT PRIMARY KEY,
            from_baseline_version INT,
            to_baseline_version INT,
            action TEXT,
            decision_id TEXT,
            linked_event_id TEXT,
            transition JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS z3_snapshots (
            id SERIAL PRIMARY KEY,
            baseline_version INT,
            state JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        -- Add indexes for new tables
        CREATE INDEX IF NOT EXISTS idx_facts_created_at ON facts(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_cross_domain_mappings_source_target ON cross_domain_mappings(source_domain, target_domain);
        CREATE INDEX IF NOT EXISTS idx_observation_history_timestamp ON observation_history(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_price_history_symbol ON price_history(symbol);
        CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_z3_baselines_version ON z3_baselines(version DESC);
        CREATE INDEX IF NOT EXISTS idx_z3_novelty_observed ON z3_novelty_events(observed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_z3_decisions_created ON z3_decisions(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_z3_transitions_created ON z3_transitions(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_z3_snapshots_created ON z3_snapshots(created_at DESC);
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(schema)
            logger.info("Schema initialized")
        except Exception as e:
            logger.error(f"Schema initialization error: {e}")
    async def save_rules(self, rules: List[Dict[str, Any]]):
        """Save a list of rules to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE rules") # Clear existing rules
                for rule in rules:
                    await conn.execute("""
                        INSERT INTO rules (id, antecedent, consequent, confidence, support, domain, metadata, created_at, last_seen, observation_span_hours)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, 
                    rule.get("rule_id"),
                    json.dumps(rule.get("antecedents")), # Store as JSONB
                    rule.get("consequent"),
                    rule.get("confidence", 0.0),
                    rule.get("support_count", 0),
                    rule.get("domain"),
                    json.dumps(rule.get("metadata", {})),
                    datetime.fromisoformat(rule.get("created_at")) if rule.get("created_at") else None,
                    datetime.fromisoformat(rule.get("last_seen")) if rule.get("last_seen") else None,
                    rule.get("observation_span_hours", 0.0)
                    )
        except Exception as e:
            logger.error(f"Error saving rules: {e}")

    async def save_concepts(self, concepts: List[Dict[str, Any]]):
        """Save a list of concepts to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE concepts") # Clear existing concepts
                for concept in concepts:
                    await conn.execute("""
                        INSERT INTO concepts (id, domain, signature, confidence, first_seen, last_seen, 
                                        observation_count, observation_span_hours, distinct_time_windows, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    concept.get("concept_id"),
                    concept.get("domain"),
                    json.dumps(concept.get("attributes", {})), # Store attributes as signature
                    concept.get("confidence", 0.0),
                    datetime.fromisoformat(concept.get("created_at")) if concept.get("created_at") else None, # Using created_at as first_seen
                    datetime.fromisoformat(concept.get("created_at")) if concept.get("created_at") else None, # Using created_at as last_seen for now
                    concept.get("example_count", 0),
                    0.0, # observation_span_hours not directly available
                    0, # distinct_time_windows not directly available
                    json.dumps(concept.get("metadata", {}))
                    )
        except Exception as e:
            logger.error(f"Error saving concepts: {e}")

    
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
    
    async def load_rules(self) -> List[Dict[str, Any]]:
        """Load all rules from the database"""
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT id, antecedent, consequent, confidence, support, created_at, last_seen, observation_span_hours, domain, metadata FROM rules")
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            return []

    async def load_concepts(self) -> List[Dict[str, Any]]:
        """Load all concepts from the database"""
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT id, domain, signature, confidence, first_seen, last_seen, observation_count, observation_span_hours, distinct_time_windows, metadata FROM concepts")
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error loading concepts: {e}")
            return []

    async def save_goals(self, goals: List[Dict[str, Any]]):
        """Save a list of goals to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                # Clear existing goals and insert new ones
                await conn.execute("DELETE FROM goals")
                for goal in goals:
                    await conn.execute("""
                        INSERT INTO goals (id, type, description, success_criteria, priority, status, progress, value_estimate, created_at, deadline, parent_goal, sub_goals, attempts, last_attempt, achieved_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    goal.get('goal_id'),
                    goal.get('type'),
                    goal.get('description'),
                    json.dumps(goal.get('success_criteria', {})),
                    goal.get('priority'),
                    goal.get('status'),
                    goal.get('progress'),
                    goal.get('value_estimate'),
                    datetime.fromisoformat(goal.get('created_at')) if goal.get('created_at') else None,
                    datetime.fromisoformat(goal.get('deadline')) if goal.get('deadline') else None,
                    goal.get('parent_goal'),
                    json.dumps(goal.get('sub_goals', [])),
                    goal.get('attempts'),
                    datetime.fromisoformat(goal.get('last_attempt')) if goal.get('last_attempt') else None,
                    datetime.fromisoformat(goal.get('achieved_at')) if goal.get('achieved_at') else None
                    )
        except Exception as e:
            logger.error(f"Error saving goals: {e}")

    async def load_goals(self) -> List[Dict[str, Any]]:
        """Load all goals from the database"""
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT id, type, description, success_criteria, priority, status, progress, value_estimate, created_at, deadline, parent_goal, sub_goals, attempts, last_attempt, achieved_at FROM goals")
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error loading goals: {e}")
            return []

    async def save_prediction_engine_state(self, state: Dict[str, Any]):
        """Save the state of the prediction engine"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO prediction_engine_state (id, state_data)
                    VALUES ($1, $2)
                    ON CONFLICT (id) DO UPDATE SET
                        state_data = $2
                """, "current_state", json.dumps(state))
        except Exception as e:
            logger.error(f"Error saving prediction engine state: {e}")

    async def load_prediction_engine_state(self) -> Optional[Dict[str, Any]]:
        """Load the state of the prediction engine"""
        if not self.pool:
            return None
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT state_data FROM prediction_engine_state WHERE id = $1", "current_state")
                if row:
                    return json.loads(row['state_data'])
                return None
        except Exception as e:
            logger.error(f"Error loading prediction engine state: {e}")
            return None

    async def save_facts(self, facts: Set[str]):
        """Save a set of facts to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE facts") # Clear existing facts
                for fact in facts:
                    await conn.execute("INSERT INTO facts (fact) VALUES ($1) ON CONFLICT (fact) DO NOTHING", fact)
        except Exception as e:
            logger.error(f"Error saving facts: {e}")

    async def load_facts(self) -> Set[str]:
        """Load all facts from the database"""
        if not self.pool:
            return set()
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT fact FROM facts")
                return {row["fact"] for row in rows}
        except Exception as e:
            logger.error(f"Error loading facts: {e}")
            return set()

    async def save_cross_domain_mappings(self, mappings: List[Dict[str, Any]]):
        """Save cross-domain mappings to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE cross_domain_mappings")
                for mapping in mappings:
                    await conn.execute("""
                        INSERT INTO cross_domain_mappings (mapping_id, source_domain, target_domain, concept_mappings, confidence, bidirectional)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    mapping.get("mapping_id"),
                    mapping.get("source_domain"),
                    mapping.get("target_domain"),
                    json.dumps(mapping.get("concept_mappings", {})),
                    mapping.get("confidence"),
                    mapping.get("bidirectional", False)
                    )
        except Exception as e:
            logger.error(f"Error saving cross-domain mappings: {e}")

    async def load_cross_domain_mappings(self) -> List[Dict[str, Any]]:
        """Load cross-domain mappings from the database"""
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT mapping_id, source_domain, target_domain, concept_mappings, confidence, bidirectional FROM cross_domain_mappings")
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error loading cross-domain mappings: {e}")
            return []

    async def save_observation_history(self, history: List[Tuple[Dict[str, Any], str]]):
        """Save observation history to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE observation_history")
                for obs, domain in history:
                    await conn.execute("INSERT INTO observation_history (observation, domain) VALUES ($1, $2)", json.dumps(obs), domain)
        except Exception as e:
            logger.error(f"Error saving observation history: {e}")

    async def load_observation_history(self) -> List[Tuple[Dict[str, Any], str]]:
        """Load observation history from the database"""
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT observation, domain FROM observation_history ORDER BY timestamp ASC")
                return [(json.loads(row["observation"]), row["domain"]) for row in rows]
        except Exception as e:
            logger.error(f"Error loading observation history: {e}")
            return []

    async def save_price_history(self, price_history: Dict[str, List[float]]):
        """Save price history to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE price_history")
                for symbol, prices in price_history.items():
                    for price in prices:
                        await conn.execute("INSERT INTO price_history (symbol, price) VALUES ($1, $2)", symbol, price)
        except Exception as e:
            logger.error(f"Error saving price history: {e}")

    async def load_price_history(self) -> Dict[str, List[float]]:
        """Load price history from the database"""
        if not self.pool:
            return defaultdict(list)
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT symbol, price FROM price_history ORDER BY timestamp ASC")
                history = defaultdict(list)
                for row in rows:
                    history[row["symbol"]].append(row["price"])
                return history
        except Exception as e:
            logger.error(f"Error loading price history: {e}")
            return defaultdict(list)

    async def save_meta_domains(self, meta_domains: Dict[str, Set[str]]):
        """Save meta domains to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE meta_domains")
                for meta_domain, domains in meta_domains.items():
                    await conn.execute("INSERT INTO meta_domains (meta_domain, domains) VALUES ($1, $2)", meta_domain, json.dumps(list(domains)))
        except Exception as e:
            logger.error(f"Error saving meta domains: {e}")

    async def load_meta_domains(self) -> Dict[str, Set[str]]:
        """Load meta domains from the database"""
        if not self.pool:
            return defaultdict(set)
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT meta_domain, domains FROM meta_domains")
                meta_domains = defaultdict(set)
                for row in rows:
                    meta_domains[row["meta_domain"]] = set(json.loads(row["domains"]))
                return meta_domains
        except Exception as e:
            logger.error(f"Error loading meta domains: {e}")
            return defaultdict(set)

    async def save_learning_engine_state(self, state: Dict[str, Any]):
        """Save the state of the learning engine"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO learning_engine_state (id, state_data)
                    VALUES ($1, $2)
                    ON CONFLICT (id) DO UPDATE SET
                        state_data = $2
                """, "current_state", json.dumps(state))
        except Exception as e:
            logger.error(f"Error saving learning engine state: {e}")

    async def load_learning_engine_state(self) -> Optional[Dict[str, Any]]:
        """Load the state of the learning engine"""
        if not self.pool:
            return None
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT state_data FROM learning_engine_state WHERE id = $1", "current_state")
                if row:
                    return json.loads(row["state_data"])
                return None
        except Exception as e:
            logger.error(f"Error loading learning engine state: {e}")
            return None

    async def save_caches(self, caches: Dict[str, Any]):
        """Save various caches to the database"""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                for cache_name, data in caches.items():
                    # Handle both list/deque and dict
                    if isinstance(data, (list, deque)):
                        serializable_data = list(data)
                    else:
                        serializable_data = data
                    
                    await conn.execute("""
                        INSERT INTO caches (cache_name, data)
                        VALUES ($1, $2)
                        ON CONFLICT (cache_name) DO UPDATE SET
                            data = $2
                    """, cache_name, json.dumps(serializable_data))
        except Exception as e:
            logger.error(f"Error saving caches: {e}")

    async def load_caches(self) -> Dict[str, Any]:
        """Load various caches from the database"""
        if not self.pool:
            return {}
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT cache_name, data FROM caches")
                loaded_caches = {}
                for row in rows:
                    loaded_caches[row["cache_name"]] = json.loads(row["data"])
                return loaded_caches
        except Exception as e:
            logger.error(f"Error loading caches: {e}")
            return {}

    async def save_short_term_memory(self, data: Dict[str, Any]):
        """Save the learning engine's short-term memory and observation count."""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO short_term_memory (id, data, updated_at)
                    VALUES ('singleton', $1, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO UPDATE SET
                        data = $1,
                        updated_at = CURRENT_TIMESTAMP
                """, json.dumps(data))
        except Exception as e:
            logger.error(f"Error saving short-term memory: {e}")

    async def load_short_term_memory(self) -> Optional[Dict[str, Any]]:
        """Load the learning engine's short-term memory and observation count."""
        if not self.pool:
            return None
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT data FROM short_term_memory WHERE id = 'singleton'")
                if row:
                    return json.loads(row["data"])
                return None
        except Exception as e:
            logger.error(f"Error loading short-term memory: {e}")
            return None

    async def save_z3_state(self, z3_state: Dict[str, Any]):
        """Persist public Z3 baseline, novelty feed, decisions, transitions, and snapshot."""
        if not self.pool or not z3_state:
            return
        try:
            baseline = z3_state.get("baseline") or {}
            baseline_id = baseline.get("baseline_id", "z3-baseline-unknown")
            baseline_version = int(baseline.get("version", 0) or 0)
            last_decision = z3_state.get("last_decision") or {}
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO z3_baselines (baseline_id, version, state, updated_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                    ON CONFLICT (baseline_id) DO UPDATE SET
                        version = $2,
                        state = $3,
                        updated_at = CURRENT_TIMESTAMP
                """, baseline_id, baseline_version, json.dumps(baseline))

                for event in z3_state.get("novelty_events", []) or []:
                    if not isinstance(event, dict):
                        continue
                    await conn.execute("""
                        INSERT INTO z3_novelty_events
                            (event_id, baseline_version, source, signal_type, novelty_score, severity, event, observed_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, to_timestamp($8))
                        ON CONFLICT (event_id) DO UPDATE SET event = $7
                    """,
                    event.get("event_id"),
                    int(event.get("baseline_version", baseline_version) or baseline_version),
                    event.get("source"),
                    event.get("signal_type"),
                    float(event.get("novelty_score", 0.0) or 0.0),
                    event.get("severity"),
                    json.dumps(event),
                    float(event.get("observed_at", 0.0) or 0.0))

                if last_decision:
                    await conn.execute("""
                        INSERT INTO z3_decisions
                            (decision_id, baseline_version, action, linked_event_id, confidence, decision, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, to_timestamp($7))
                        ON CONFLICT (decision_id) DO UPDATE SET decision = $6
                    """,
                    last_decision.get("decision_id"),
                    int(last_decision.get("baseline_version", baseline_version) or baseline_version),
                    last_decision.get("action"),
                    last_decision.get("linked_event_id"),
                    float(last_decision.get("confidence", 0.0) or 0.0),
                    json.dumps(last_decision),
                    float(last_decision.get("created_at", 0.0) or 0.0))

                for transition in z3_state.get("transitions", []) or []:
                    if not isinstance(transition, dict):
                        continue
                    await conn.execute("""
                        INSERT INTO z3_transitions
                            (transition_id, from_baseline_version, to_baseline_version, action, decision_id, linked_event_id, transition, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, to_timestamp($8))
                        ON CONFLICT (transition_id) DO UPDATE SET transition = $7
                    """,
                    transition.get("transition_id"),
                    int(transition.get("from_baseline_version", baseline_version) or baseline_version),
                    int(transition.get("to_baseline_version", baseline_version) or baseline_version),
                    transition.get("action"),
                    transition.get("decision_id"),
                    transition.get("linked_event_id"),
                    json.dumps(transition),
                    float(transition.get("created_at", 0.0) or 0.0))

                await conn.execute("""
                    INSERT INTO z3_snapshots (baseline_version, state)
                    VALUES ($1, $2)
                """, baseline_version, json.dumps(z3_state))
        except Exception as e:
            logger.error(f"Error saving Z3 state: {e}")

    async def load_latest_z3_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the latest public Z3 snapshot."""
        if not self.pool:
            return None
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT state FROM z3_snapshots
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                if row:
                    return json.loads(row["state"])
                return None
        except Exception as e:
            logger.error(f"Error loading latest Z3 snapshot: {e}")
            return None

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
        """Close connection pool without hanging shutdown indefinitely."""
        if self.pool:
            try:
                await asyncio.wait_for(self.pool.close(), timeout=10.0)
            except Exception as e:
                logger.warning(f"PostgreSQL pool close timed out/failed ({e}); terminating pool")
                try:
                    self.pool.terminate()
                except Exception:
                    pass
            finally:
                self.pool = None
                self.connected = False
                logger.info("Disconnected from PostgreSQL")
