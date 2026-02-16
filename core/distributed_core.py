"""
Distributed Cognitive Core
==========================
Async wrapper around the original CognitiveIntelligentSystem that:
- Bridges async HTTP/network layer with the synchronous cognitive engines
- Maintains database persistence (PostgreSQL, Milvus, Redis)
- Exposes the real cognitive state (concepts, rules, goals, cross-domain mappings)
- Runs the cognitive loop in a background thread
"""

import asyncio
import logging
import time
import os
import threading
import math
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from collections import deque

from config.config import Config

# Import the REAL cognitive engines
from cognitive_intelligent_system import CognitiveIntelligentSystem
from abstraction_engine import AbstractionEngine
from reasoning_engine import ReasoningEngine, RuleType
from cross_domain_engine import CrossDomainEngine
from goal_formation_system import OpenEndedGoalSystem, GoalType

logger = logging.getLogger("DistributedCognitiveCore")


class DistributedCognitiveCore:
    """
    Production-grade async wrapper around CognitiveIntelligentSystem.

    The original CognitiveIntelligentSystem is synchronous and runs its own
    cognitive loop. This wrapper:
    1. Creates the CognitiveIntelligentSystem with all 7 engines
    2. Exposes an async `ingest()` method for the HTTP/network layer
    3. Runs the cognitive loop in a background thread
    4. Persists state to PostgreSQL, Milvus, Redis when available
    5. Exposes the real cognitive state for the dashboard and GPT I/O
    """

    def __init__(self, node_id: str, postgres=None, milvus=None, redis=None):
        self.node_id = node_id
        self.postgres = postgres
        self.milvus = milvus
        self.redis = redis

        # Create the REAL cognitive system
        self.cognitive_system = CognitiveIntelligentSystem(
            system_id=node_id,
            feature_dim=50,
            learning_rate=0.01,
            enable_all_features=True
        )

        # Thread lock for safe access from async context
        self._lock = threading.Lock()

        # Track observations for the data collection loop
        self._pending_observations: deque = deque(maxlen=5000)
        self._observation_count = 0

        # Metrics overlay (combines cognitive system metrics with infra metrics)
        self._start_time = time.time()
        self._last_observation_time: Optional[float] = None
        self._errors = 0

        # Background cognitive loop control
        self._cognitive_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"DistributedCognitiveCore initialized with CognitiveIntelligentSystem: {node_id}")

    def start_cognitive_loop(self):
        """Start the cognitive processing loop in a background thread"""
        if self._cognitive_thread and self._cognitive_thread.is_alive():
            logger.warning("Cognitive loop already running")
            return

        self._running = True

        def _loop():
            logger.info("Cognitive loop started (background thread)")
            iteration = 0

            while self._running:
                try:
                    iteration += 1

                    # Process any pending observations
                    observations_processed = 0
                    while self._pending_observations and observations_processed < 50:
                        try:
                            obs, domain = self._pending_observations.popleft()
                        except IndexError:
                            break

                        with self._lock:
                            try:
                                outcome = obs.pop('outcome', None)
                                result = self.cognitive_system.process_observation(
                                    obs, domain, outcome
                                )
                                observations_processed += 1

                                if result.get('concept'):
                                    logger.info(
                                        f"Concept formed: {result['concept']} "
                                        f"in domain {domain}"
                                    )

                                if result.get('inferred_facts'):
                                    logger.info(
                                        f"Inferred {len(result['inferred_facts'])} new facts"
                                    )

                            except Exception as e:
                                logger.error(f"Error processing observation: {e}")
                                self._errors += 1

                    # Cognitive thinking every 10 iterations
                    if iteration % 10 == 0:
                        with self._lock:
                            try:
                                thoughts = self.cognitive_system.think({})
                                if thoughts:
                                    thought_count = sum(
                                        len(v) if isinstance(v, (list, dict)) else 1
                                        for v in thoughts.values()
                                    )
                                    if thought_count > 0:
                                        logger.info(
                                            f"Cognitive thoughts: {thought_count} insights "
                                            f"(analogies, explanations, goals)"
                                        )
                            except Exception as e:
                                logger.error(f"Error in cognitive thinking: {e}")

                    # Generate autonomous goals every 50 iterations
                    if iteration % 50 == 0:
                        with self._lock:
                            try:
                                new_goals = self.cognitive_system.generate_autonomous_goals()
                                if new_goals:
                                    logger.info(
                                        f"Generated {len(new_goals)} autonomous goals"
                                    )
                            except Exception as e:
                                logger.error(f"Error generating goals: {e}")

                    # Pursue goals every 25 iterations
                    if iteration % 25 == 0:
                        with self._lock:
                            try:
                                self.cognitive_system.pursue_goals()
                            except Exception as e:
                                logger.error(f"Error pursuing goals: {e}")

                    # Introspect every 100 iterations
                    if iteration % 100 == 0:
                        with self._lock:
                            try:
                                introspection = self.cognitive_system.introspect()
                                logger.info("=" * 60)
                                logger.info("INTROSPECTION")
                                logger.info(f"  Concepts: {introspection['abstraction']['total_concepts']}")
                                logger.info(f"  Rules: {introspection['reasoning']['total_rules']}")
                                logger.info(f"  Goals: {introspection['goals']['total_goals']}")
                                logger.info(f"  Transfers: {introspection['cognitive_metrics']['knowledge_transfers']}")
                                logger.info(f"  Domains: {introspection['active_domains']}")
                                logger.info("=" * 60)
                            except Exception as e:
                                logger.error(f"Error in introspection: {e}")

                    time.sleep(1.0)  # 1 second per iteration

                except Exception as e:
                    logger.error(f"Error in cognitive loop iteration {iteration}: {e}")
                    time.sleep(5)

            logger.info("Cognitive loop stopped")

        self._cognitive_thread = threading.Thread(target=_loop, daemon=True, name="CognitiveLoop")
        self._cognitive_thread.start()
        logger.info("Cognitive loop thread started")

    def stop_cognitive_loop(self):
        """Stop the cognitive loop"""
        self._running = False
        if self._cognitive_thread:
            self._cognitive_thread.join(timeout=10)
            logger.info("Cognitive loop thread stopped")

    async def ingest(self, observation: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Async ingestion endpoint.
        Queues the observation for the cognitive loop to process.
        """
        self._observation_count += 1
        self._last_observation_time = time.time()

        # Register domain if new
        with self._lock:
            if domain not in self.cognitive_system.cross_domain.domains:
                self.cognitive_system.cross_domain.register_domain(domain, domain)
                logger.info(f"Registered new domain: {domain}")

        # Queue for cognitive processing
        self._pending_observations.append((observation.copy(), domain))

        # Async persistence
        await self._persist_observation(observation, domain)

        return {
            "success": True,
            "queued": True,
            "observation_count": self._observation_count,
            "pending": len(self._pending_observations),
            "concept_count": len(self.cognitive_system.abstraction.concepts),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _persist_observation(self, observation: Dict[str, Any], domain: str):
        """Persist observation to databases"""
        try:
            if self.postgres:
                await self.postgres.save_observation({
                    "symbol": observation.get("symbol"),
                    "price": observation.get("price"),
                    "volume": observation.get("volume"),
                    "domain": domain,
                    "metadata": {"source": observation.get("source", "unknown")}
                })
        except Exception as e:
            logger.error(f"Persistence error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined cognitive + infrastructure metrics"""
        with self._lock:
            cognitive_metrics = self.cognitive_system.cognitive_metrics.copy()
            abstraction_insights = self.cognitive_system.abstraction.get_insights()
            reasoning_insights = self.cognitive_system.reasoning.get_insights()
            cross_domain_insights = self.cognitive_system.cross_domain.get_insights()
            goal_insights = self.cognitive_system.goals.get_insights()
            learning_insights = self.cognitive_system.learning_engine.get_insights()

        # Calculate PHI and SIGMA from the real cognitive state
        phi = self._calculate_phi(reasoning_insights, abstraction_insights)
        sigma = self._calculate_sigma(abstraction_insights, learning_insights)

        return {
            # Core consciousness metrics
            "global_coherence_phi": round(phi, 4),
            "noise_level_sigma": round(sigma, 4),
            "attention_density": round(
                abstraction_insights.get('total_concepts', 0) / 1000.0, 4
            ),

            # Cognitive metrics from the real engines
            "concepts_formed": cognitive_metrics.get('concepts_formed', 0),
            "rules_learned": cognitive_metrics.get('rules_learned', 0),
            "analogies_found": cognitive_metrics.get('analogies_found', 0),
            "goals_achieved": cognitive_metrics.get('goals_achieved', 0),
            "knowledge_transfers": cognitive_metrics.get('knowledge_transfers', 0),

            # Engine-specific state
            "total_concepts": abstraction_insights.get('total_concepts', 0),
            "total_rules": reasoning_insights.get('total_rules', 0),
            "total_facts": reasoning_insights.get('total_facts', 0),
            "total_domains": cross_domain_insights.get('total_domains', 0),
            "total_goals": goal_insights.get('total_goals', 0),
            "active_goals": goal_insights.get('active_goals', 0),
            "achieved_goals": goal_insights.get('achieved_goals', 0),
            "total_mappings": cross_domain_insights.get('total_mappings', 0),

            # Learning metrics
            "learning_accuracy": learning_insights.get('metrics', {}).get('accuracy', 0),
            "patterns_discovered": learning_insights.get('total_patterns', 0),
            "samples_processed": learning_insights.get('metrics', {}).get('samples_processed', 0),

            # Infrastructure metrics
            "total_observations": self._observation_count,
            "pending_observations": len(self._pending_observations),
            "errors": self._errors,
            "uptime_seconds": time.time() - self._start_time,
            "last_observation_time": self._last_observation_time,
            "cognitive_loop_running": self._running,

            # Transfers count (legacy field for dashboard compatibility)
            "transfers_made": cognitive_metrics.get('knowledge_transfers', 0),
        }

    def _calculate_phi(self, reasoning: Dict, abstraction: Dict) -> float:
        """
        PHI (Global Coherence) — derived from:
        - Rule confidence (how stable are learned rules)
        - Concept confidence (how well-formed are concepts)
        - Goal achievement rate
        """
        rule_confidence = 0.5
        total_rules = reasoning.get('total_rules', 0)
        if total_rules > 0:
            # Average rule confidence from the reasoning engine
            rule_confidence = min(1.0, 0.5 + (total_rules * 0.02))

        concept_confidence = 0.5
        total_concepts = abstraction.get('total_concepts', 0)
        if total_concepts > 0:
            avg_confidence = abstraction.get('average_confidence', 0.5)
            concept_confidence = avg_confidence

        phi = (rule_confidence * 0.5 + concept_confidence * 0.5)
        return max(0.0, min(1.0, phi))

    def _calculate_sigma(self, abstraction: Dict, learning: Dict) -> float:
        """
        SIGMA (Noise Level) — derived from:
        - Concept volatility (how much concepts are changing)
        - Learning adaptation rate (how much drift is detected)
        - Inverse of pattern stability
        """
        total_concepts = abstraction.get('total_concepts', 0)
        total_patterns = learning.get('total_patterns', 0)
        adaptations = learning.get('metrics', {}).get('adaptations', 0)

        # Base noise from concept instability
        if total_concepts == 0:
            concept_noise = 0.5
        else:
            concept_noise = max(0.1, 1.0 - (total_concepts * 0.02))

        # Adaptation noise (more adaptations = more drift = more noise)
        adaptation_noise = min(0.5, adaptations * 0.05)

        sigma = concept_noise * 0.7 + adaptation_noise * 0.3
        return max(0.0, min(1.0, sigma))

    def get_concepts_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all concepts from the AbstractionEngine"""
        with self._lock:
            concepts = {}

            # Build a reverse map: concept_id -> domain from cross-domain engine
            concept_domain_map = {}
            for domain_id, domain in self.cognitive_system.cross_domain.domains.items():
                for concept_id in getattr(domain, 'concepts', []):
                    concept_domain_map[concept_id] = domain_id

            for cid, concept in self.cognitive_system.abstraction.concepts.items():
                examples = getattr(concept, 'examples', [])
                recent_examples = []
                for ex in examples[-5:]:
                    # Convert numpy types to native Python types for JSON
                    if isinstance(ex, dict):
                        clean = {}
                        for k, v in ex.items():
                            try:
                                clean[k] = float(v) if hasattr(v, '__float__') and not isinstance(v, str) else v
                            except (ValueError, TypeError):
                                clean[k] = str(v)
                        recent_examples.append(clean)
                    else:
                        recent_examples.append(ex)

                # Extract symbol from examples if available
                symbol = None
                for ex in examples[-3:]:
                    if isinstance(ex, dict) and 'symbol' in ex:
                        symbol = ex['symbol']
                        break

                concepts[cid] = {
                    "id": cid,
                    "name": getattr(concept, 'name', cid),
                    "domain": concept_domain_map.get(cid, 'unknown'),
                    "symbol": symbol,
                    "confidence": round(getattr(concept, 'confidence', 0), 4),
                    "level": getattr(concept, 'level', 0),
                    "observation_count": len(examples),
                    "examples": recent_examples,
                    "created_at": getattr(concept, 'created_at', None),
                    "parent_concepts": list(getattr(concept, 'parent_concepts', set())),
                    "child_concepts": list(getattr(concept, 'child_concepts', set())),
                }

            return concepts

    def get_rules_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all rules from the ReasoningEngine"""
        with self._lock:
            rules = {}
            for rid, rule in self.cognitive_system.reasoning.rules.items():
                rules[rid] = {
                    "id": rid,
                    "rule_type": getattr(rule, 'rule_type', RuleType.IF_THEN).value
                        if hasattr(getattr(rule, 'rule_type', None), 'value')
                        else str(getattr(rule, 'rule_type', 'unknown')),
                    "confidence": getattr(rule, 'confidence', 0),
                    "support": getattr(rule, 'support', 0),
                    "antecedents": list(getattr(rule, 'antecedents', [])),
                    "consequents": list(getattr(rule, 'consequents', [])),
                }
            return rules

    def get_goals_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all goals"""
        with self._lock:
            goals = {}
            for gid, goal in self.cognitive_system.goals.goals.items():
                goals[gid] = {
                    "id": gid,
                    "description": getattr(goal, 'description', ''),
                    "goal_type": getattr(goal, 'goal_type', GoalType.EXPLORATION).value
                        if hasattr(getattr(goal, 'goal_type', None), 'value')
                        else str(getattr(goal, 'goal_type', 'unknown')),
                    "status": str(getattr(goal, 'status', 'unknown')),
                    "priority": getattr(goal, 'priority', 0),
                    "progress": getattr(goal, 'progress', 0),
                }
            return goals

    def get_cross_domain_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of cross-domain mappings"""
        with self._lock:
            mappings = {}
            for mid, mapping in self.cognitive_system.cross_domain.mappings.items():
                mappings[mid] = {
                    "id": mid,
                    "source_domain": getattr(mapping, 'source_domain', ''),
                    "target_domain": getattr(mapping, 'target_domain', ''),
                    "confidence": getattr(mapping, 'confidence', 0),
                    "concept_pairs": getattr(mapping, 'concept_pairs', []),
                }
            return mappings

    def get_learning_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of learning engine state"""
        with self._lock:
            return self.cognitive_system.learning_engine.get_insights()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get complete state summary for gossip/network"""
        metrics = self.get_metrics()
        return {
            "node_id": self.node_id,
            "phi": metrics["global_coherence_phi"],
            "sigma": metrics["noise_level_sigma"],
            "concepts_count": metrics["total_concepts"],
            "rules_count": metrics["total_rules"],
            "goals_count": metrics["total_goals"],
            "domains_count": metrics["total_domains"],
            "metrics": metrics,
            "timestamp": time.time()
        }

    def get_introspection(self) -> Dict[str, Any]:
        """Full system introspection"""
        with self._lock:
            return self.cognitive_system.introspect()

    async def process_network_message(self, msg: Any):
        """Process incoming network messages"""
        logger.debug(f"Processing network message: {msg}")

    async def process_pubsub_message(self, msg: Any):
        """Process incoming pubsub messages"""
        logger.debug(f"Processing pubsub message: {msg}")
