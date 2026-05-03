"""
Distributed Cognitive Core — The Orchestrator
==============================================
Manages multiple cognitive engines, prediction engine, and infrastructure.
Enables cross-domain coherence, rule feedback, and concept convergence.

This core is fully domain-agnostic.  It accepts observations of ANY shape
and routes them through all 7 cognitive engines.  Financial market data,
IoT sensor readings, weather observations, and any other continuous stream
are all treated identically.  Domain-specific adapters (e.g. MarketPlugin)
live outside this core and translate their native schemas into the generic
observation format before ingestion.
"""

import logging
import time
import threading
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
import numpy as np

from abstraction_engine import Concept
from reasoning_engine import Rule, RuleType
from cross_domain_engine import DomainMapping
from goal_formation_system import Goal, GoalType, GoalStatus
from prediction_validation_engine import Prediction, Direction, StreamHistory, SymbolHistory, RulePerformance
from continuous_learning_engine import Pattern, LearningMetrics

from cognitive_intelligent_system import CognitiveIntelligentSystem
from prediction_validation_engine import PredictionValidationEngine
from world_model import OnlineWorldModel
from config.config import Config
# Mesh architecture — Coordinator and contracts
from coordinator import MeshCoordinator
from contracts import EEGOutput

logger = logging.getLogger("DistributedCore")


def _serialize_enum(obj):
    if hasattr(obj, 'value'):
        return obj.value
    return str(obj)


class DistributedCognitiveCore:
    """
    Orchestrates the high-level cognitive processes across the mesh.
    - Manages the CognitiveIntelligentSystem (7 engines)
    - Manages the PredictionValidationEngine
    - Runs the background cognitive loop
    - Performs concept convergence and rule feedback
    """

    def __init__(
        self,
        node_id: str = "core-0",
        postgres=None,
        milvus=None,
        redis=None,
        network=None,
        pubsub=None
    ):
        self.node_id = node_id
        self.postgres = postgres
        self.milvus = milvus
        self.redis = redis
        self.network = network
        self.pubsub = pubsub

        # Core Engines
        self.cognitive_system = CognitiveIntelligentSystem(system_id=node_id)
        self.prediction_engine = PredictionValidationEngine()
        self.world_model = OnlineWorldModel()
        # Mesh Coordinator — the Z³ anchor (Principle 3 & 4)
        self.coordinator = MeshCoordinator()

        # Self-evolution engine (reactivated)
        from self_writing_engine import SelfEvolvingSystem
        self.code_evolver = SelfEvolvingSystem(
            max_gen=5,
            pop_size=10,
            mutation_rate=0.3,
            crossover_rate=0.6
        )
        self._evolution_counter = 0

        # State
        self._running = False
        self._cognitive_thread = None
        self._start_time = time.time()
        self._observation_count = 0
        self._last_observation_time = 0
        self._errors = 0
        self._lock = threading.Lock()

        # Convergence tracking
        self._convergence_counter = 0
        self._concepts_merged = 0
        self._concepts_pruned = 0

        # Self-evolution toggle (on by default)
        self._toggles_extra: Dict[str, Any] = {"self_evolution": True}

        # Buffer for async ingestion
        self._pending_observations = deque(maxlen=500)

        # Pre-computed state cache — updated by the cognitive loop, read by HTTP handlers
        self._state_cache: Dict[str, Any] = {}
        self._state_cache_lock = threading.Lock()
        self._state_cache_ready = False

        # Data provider reference — set externally after init so providers tab works
        self.data_provider = None

        # Self-reflection baseline — tracks historical introspection snapshots
        # so the system can compare current state against its own past and act.
        self._introspection_baseline: Dict[str, Any] = {}
        self._introspection_history: deque = deque(maxlen=50)

        # Activity log — rolling window of cognitive events shown in dashboard
        self._activity_log: deque = deque(maxlen=200)

        # Recursive integration state — compact shared state that feeds module
        # outputs back into future module behavior. This turns the mesh from a
        # set of reporters into a closed observation → state → control loop.
        self._recursive_state: Dict[str, Any] = {
            "iteration": 0,
            "coherence_loss": 0.5,
            "loss_delta": 0.0,
            "last_entity_id": None,
            "last_domain": None,
            "phi": 0.5,
            "sigma": 0.5,
            "drift": 0.0,
            "prediction_accuracy": 0.0,
            "memory_reconstruction_confidence": 0.0,
            "world_model_loss": 0.5,
            "world_model_prediction_loss": 0.5,
            "world_model_reconstruction_loss": 0.5,
            "world_model_memory_loss": 1.0,
            "active_goals": 0,
            "control": {},
        }
        self._recursive_feedback_log: deque = deque(maxlen=200)

        # Toggles — runtime feature flags
        self._toggles: Dict[str, Any] = {
            "cognitive_loop": True,
            "concept_convergence": True,
            "rule_feedback": True,
            "deep_introspection": True,
            "goal_formation": True,
            "knowledge_transfer": True,
            "prediction_engine": True,
            "self_evolution": True,
        }

    async def ingest_historical_data(self, historical_observations: List[Tuple[Dict[str, Any], str]]):
        """Ingest historical observations, replaying them through the cognitive system."""
        logger.info(f"Ingesting {len(historical_observations)} historical observations...")
        for i, (obs, domain) in enumerate(historical_observations):
            try:
                # Replay historical observation through the standard ingestion pipeline
                # This ensures it hits all 7 engines and lands in all the right places
                await self.ingest(obs, domain)

                if i % 100 == 0:
                    logger.info(f"Queued {i+1}/{len(historical_observations)} historical observations for replay.")

            except Exception as e:
                logger.error(f"Error queuing historical observation {i}: {e}")
        logger.info("Historical data ingestion complete.")


    async def save_state(self):
        """Save the entire cognitive state to persistent storage."""
        logger.info("Saving cognitive state...")
        if self.postgres:
            # Save Abstraction Engine state
            concepts_to_save = [c.to_dict() for c in self.cognitive_system.abstraction.concepts.values()]
            await self.postgres.save_concepts(concepts_to_save)
            logger.debug(f"Saved {len(concepts_to_save)} concepts to Postgres.")

            # Save Reasoning Engine state
            rules_to_save = [r.to_dict() for r in self.cognitive_system.reasoning.rules.values()]
            await self.postgres.save_rules(rules_to_save)
            await self.postgres.save_facts(self.cognitive_system.reasoning.facts)
            logger.debug(f"Saved {len(rules_to_save)} rules and {len(self.cognitive_system.reasoning.facts)} facts to Postgres.")

            # Save Cross-Domain Engine state (mappings)
            mappings_to_save = [m.to_dict() for m in self.cognitive_system.cross_domain.mappings.values()]
            await self.postgres.save_cross_domain_mappings(mappings_to_save)
            logger.debug(f"Saved {len(mappings_to_save)} cross-domain mappings to Postgres.")

            # Save Goal Formation System state
            goals_to_save = [g.to_dict() for g in self.cognitive_system.goals.goals.values()]
            await self.postgres.save_goals(goals_to_save)
            logger.debug(f"Saved {len(goals_to_save)} goals to Postgres.")

            # Save Prediction Validation Engine state
            # The prediction engine has a `save` method that takes postgres as an argument
            # I need to ensure that the prediction engine's save method is implemented to use the postgres client
            # For now, I will save its internal state directly if possible, or adapt its save method later.
            # Assuming prediction_engine has a serializable state for now.
            prediction_engine_state = {
                "symbols": {s_id: s.__dict__ for s_id, s in self.prediction_engine.symbols.items()},
                "all_predictions": [p.__dict__ for p in self.prediction_engine.all_predictions],
                "prediction_counter": self.prediction_engine.prediction_counter,
                "rule_performance": {r_id: r.__dict__ for r_id, r in self.prediction_engine.rule_performance.items()},
                "accuracy_history": list(self.prediction_engine.accuracy_history),
                "phi_history": list(self.prediction_engine.phi_history),
                "sigma_history": list(self.prediction_engine.sigma_history),
                "total_predictions": self.prediction_engine.total_predictions,
                "total_correct": self.prediction_engine.total_correct,
                "total_validated": self.prediction_engine.total_validated,
            }
            await self.postgres.save_prediction_engine_state(prediction_engine_state)
            logger.debug("Saved Prediction Validation Engine state to Postgres.")

            # Save other RAM-only states from CognitiveIntelligentSystem
            # _observation_history
            observation_history_to_save = list(self.cognitive_system._observation_history)
            await self.postgres.save_observation_history(observation_history_to_save)
            logger.debug(f"Saved {len(observation_history_to_save)} observations to Postgres.")

            # _price_history
            price_history_to_save = {k: list(v) for k, v in self.cognitive_system._price_history.items()}
            await self.postgres.save_price_history(price_history_to_save)
            logger.debug(f"Saved price history for {len(price_history_to_save)} symbols to Postgres.")

            # _meta_domains
            meta_domains_to_save = {k: list(v) for k, v in self.cognitive_system._meta_domains.items()}
            await self.postgres.save_meta_domains(meta_domains_to_save)
            logger.debug(f"Saved {len(meta_domains_to_save)} meta domains to Postgres.")

            # learning_engine (weights, bias, feature_names, feature_index, long_term_patterns, metrics, prediction_history, feature_means, feature_vars, sample_count, drift_events, _current_lr_history)
            learning_engine_state = {
                "weights": self.cognitive_system.learning_engine.weights.tolist(),
                "bias": self.cognitive_system.learning_engine.bias,
                "feature_names": self.cognitive_system.learning_engine.feature_names,
                "feature_index": self.cognitive_system.learning_engine.feature_index,
                "long_term_patterns": {p_id: p.to_dict() for p_id, p in self.cognitive_system.learning_engine.long_term_patterns.items()},
                "metrics": self.cognitive_system.learning_engine.metrics.to_dict(),
                "prediction_history": list(self.cognitive_system.learning_engine.prediction_history),
                "feature_means": dict(self.cognitive_system.learning_engine.feature_means),
                "feature_vars": dict(self.cognitive_system.learning_engine.feature_vars),
                "sample_count": self.cognitive_system.learning_engine.sample_count,
                "drift_events": list(self.cognitive_system.learning_engine.drift_events),
                "_current_lr_history": list(self.cognitive_system.learning_engine._current_lr_history),
            }
            await self.postgres.save_learning_engine_state(learning_engine_state)
            logger.debug("Saved Learning Engine state to Postgres.")

            # Caches from CognitiveIntelligentSystem
            caches_to_save = {
                "_recent_analogies": list(self.cognitive_system._recent_analogies),
                "_recent_explanations": list(self.cognitive_system._recent_explanations),
                "_recent_plans": list(self.cognitive_system._recent_plans),
                "_causal_discovery_log": list(self.cognitive_system._causal_discovery_log),
                "_transfer_suggestions_cache": self.cognitive_system._transfer_suggestions_cache,
                "_pursuit_log": list(self.cognitive_system._pursuit_log),
                "resonant_memory_state": self.cognitive_system.resonant_memory.export_state(),
                "world_model_state": self.world_model.save_state(),
            }
            await self.postgres.save_caches(caches_to_save)
            logger.debug("Saved CognitiveIntelligentSystem caches to Postgres.")

            # Save short-term memory (the learning engine's rolling observation window).
            # Without this, the mesh cannot detect patterns or mine rules until it
            # re-accumulates enough data after every restart.
            # Also save _observation_count so the dashboard counter is continuous.
            short_term_to_save = {
                "short_term_memory": list(self.cognitive_system.learning_engine.short_term_memory),
                "observation_count": self._observation_count,
            }
            await self.postgres.save_short_term_memory(short_term_to_save)
            logger.debug(f"Saved {len(short_term_to_save['short_term_memory'])} short-term memory entries and observation_count={self._observation_count} to Postgres.")

        if self.milvus:
            # Sync concept vectors to Milvus so vector-similarity search stays current.
            # Concepts are already saved to Postgres above; here we push their signatures
            # as 128-dim vectors so the abstraction engine can do cross-domain recall.
            try:
                concepts_for_milvus = [
                    {
                        'concept_id': c.concept_id,
                        'domain': list(c.parent_concepts)[0] if c.parent_concepts else 'general',
                        'signature': {
                            k: v.get('mean', 0.5) if isinstance(v, dict) else float(v)
                            for k, v in c.attributes.items()
                            if isinstance(v, (int, float, dict))
                        },
                        'confidence': c.confidence,
                    }
                    for c in self.cognitive_system.abstraction.concepts.values()
                ]
                stored = await self.milvus.bulk_store_concepts(concepts_for_milvus)
                logger.debug(f"Synced {stored}/{len(concepts_for_milvus)} concept vectors to Milvus.")
            except Exception as e:
                logger.warning(f"Milvus concept sync skipped: {e}")

        if self.redis:
            # Push a lightweight state snapshot to Redis for fast cross-process reads.
            # TTL = 10 minutes — refreshed on every checkpoint so it never goes stale.
            try:
                snapshot = {
                    'node_id': self.node_id,
                    'observation_count': self._observation_count,
                    'total_concepts': len(self.cognitive_system.abstraction.concepts),
                    'total_rules': len(self.cognitive_system.reasoning.rules),
                    'total_goals': len(self.cognitive_system.goals.goals),
                    'saved_at': time.time(),
                }
                await self.redis.set_gossip_state(snapshot, ttl=600)
                logger.debug("State snapshot pushed to Redis.")
            except Exception as e:
                logger.warning(f"Redis state snapshot skipped: {e}")

        logger.info("Cognitive state saved.")

    async def load_state(self):
        """Load the entire cognitive state from persistent storage."""
        logger.info("Loading cognitive state...")
        if self.postgres:
            # Load Abstraction Engine state
            loaded_concepts = await self.postgres.load_concepts()
            for c_data in loaded_concepts:
                # Reconstruct Concept objects. Handle datetime conversion.
                c_data["created_at"] = datetime.fromisoformat(c_data["created_at"]) if c_data.get("created_at") else datetime.now()
                c_data["examples"] = [] # Examples are not stored in Postgres for now, only count
                c_data["parent_concepts"] = set(c_data.get("parents", []))
                c_data["child_concepts"] = set(c_data.get("children", []))
                concept = Concept(concept_id=c_data["id"], name=c_data["name"], level=c_data["level"], attributes=c_data["signature"], examples=[], confidence=c_data["confidence"], created_at=c_data["created_at"], parent_concepts=c_data["parent_concepts"], child_concepts=c_data["child_concepts"])
                self.cognitive_system.abstraction.concepts[concept.concept_id] = concept
            logger.debug(f"Loaded {len(loaded_concepts)} concepts from Postgres.")

            # Load Reasoning Engine state
            loaded_rules = await self.postgres.load_rules()
            for r_data in loaded_rules:
                r_data["created_at"] = datetime.fromisoformat(r_data["created_at"]) if r_data.get("created_at") else datetime.now()
                rule = Rule(rule_id=r_data["id"], rule_type=RuleType(r_data["type"]), antecedents=r_data["antecedent"], consequent=r_data["consequent"], confidence=r_data["confidence"], support_count=r_data["support"], created_at=r_data["created_at"])
                self.cognitive_system.reasoning.rules[rule.rule_id] = rule
            self.cognitive_system.reasoning.facts = await self.postgres.load_facts()
            logger.debug(f"Loaded {len(loaded_rules)} rules and {len(self.cognitive_system.reasoning.facts)} facts from Postgres.")

            # Load Cross-Domain Engine state (mappings)
            loaded_mappings = await self.postgres.load_cross_domain_mappings()
            for m_data in loaded_mappings:
                mapping = DomainMapping(mapping_id=m_data["mapping_id"], source_domain=m_data["source_domain"], target_domain=m_data["target_domain"], concept_mappings=m_data["concept_mappings"], confidence=m_data["confidence"], bidirectional=m_data["bidirectional"])
                self.cognitive_system.cross_domain.mappings[mapping.mapping_id] = mapping
            logger.debug(f"Loaded {len(loaded_mappings)} cross-domain mappings from Postgres.")

            # Load Goal Formation System state
            loaded_goals = await self.postgres.load_goals()
            for g_data in loaded_goals:
                g_data["created_at"] = datetime.fromisoformat(g_data["created_at"]) if g_data.get("created_at") else datetime.now()
                g_data["deadline"] = datetime.fromisoformat(g_data["deadline"]) if g_data.get("deadline") else None
                g_data["last_attempt"] = datetime.fromisoformat(g_data["last_attempt"]) if g_data.get("last_attempt") else None
                g_data["achieved_at"] = datetime.fromisoformat(g_data["achieved_at"]) if g_data.get("achieved_at") else None
                goal = Goal(goal_id=g_data["id"], goal_type=GoalType(g_data["type"]), description=g_data["description"], success_criteria=g_data["success_criteria"], priority=g_data["priority"], status=GoalStatus(g_data["status"]), progress=g_data["progress"], value_estimate=g_data["value_estimate"], created_at=g_data["created_at"], deadline=g_data["deadline"], parent_goal=g_data["parent_goal"], sub_goals=g_data["sub_goals"], attempts=g_data["attempts"], last_attempt=g_data["last_attempt"], achieved_at=g_data["achieved_at"])
                self.cognitive_system.goals.goals[goal.goal_id] = goal
            logger.debug(f"Loaded {len(loaded_goals)} goals from Postgres.")

            # Load Prediction Validation Engine state
            prediction_engine_state = await self.postgres.load_prediction_engine_state()
            if prediction_engine_state:
                for s_id, s_data in prediction_engine_state["symbols"].items():
                    history = SymbolHistory(symbol=s_data["symbol"], domain=s_data["domain"])
                    history.prices = deque(s_data["prices"], maxlen=200)
                    history.volumes = deque(s_data["volumes"], maxlen=200)
                    history.timestamps = deque(s_data["timestamps"], maxlen=200)
                    if s_data["pending_prediction"]:
                        pred_data = s_data["pending_prediction"]
                        history.pending_prediction = Prediction(prediction_id=pred_data["prediction_id"], symbol=pred_data["symbol"], domain=pred_data["domain"], direction=Direction(pred_data["direction"]), confidence=pred_data["confidence"], basis=pred_data["basis"], predicted_at=pred_data["predicted_at"], predicted_price=pred_data["predicted_price"], horizon=pred_data["horizon"], ticks_remaining=pred_data["ticks_remaining"], dead_zone_pct=pred_data["dead_zone_pct"], target_price=pred_data["target_price"], actual_price=pred_data["actual_price"], actual_direction=Direction(pred_data["actual_direction"]) if pred_data["actual_direction"] else None, validated=pred_data["validated"], max_post_signal_move_pct=pred_data["max_post_signal_move_pct"], correct=pred_data["correct"], validation_time=pred_data["validation_time"])
                    history.total_predictions = s_data["total_predictions"]
                    history.correct_predictions = s_data["correct_predictions"]
                    history.recent_accuracy = deque(s_data["recent_accuracy"], maxlen=50)
                    self.prediction_engine.symbols[s_id] = history
                self.prediction_engine.all_predictions = deque([Prediction(prediction_id=p["prediction_id"], symbol=p["symbol"], domain=p["domain"], direction=Direction(p["direction"]), confidence=p["confidence"], basis=p["basis"], predicted_at=p["predicted_at"], predicted_price=p["predicted_price"], horizon=p["horizon"], ticks_remaining=p["ticks_remaining"], dead_zone_pct=p["dead_zone_pct"], target_price=p["target_price"], actual_price=p["actual_price"], actual_direction=Direction(p["actual_direction"]) if p["actual_direction"] else None, validated=p["validated"], max_post_signal_move_pct=p["max_post_signal_move_pct"], correct=p["correct"], validation_time=p["validation_time"]) for p in prediction_engine_state["all_predictions"]], maxlen=5000)
                self.prediction_engine.prediction_counter = prediction_engine_state["prediction_counter"]
                for r_id, r_data in prediction_engine_state["rule_performance"].items():
                    rule_perf = RulePerformance(rule_id=r_data["rule_id"])
                    rule_perf.predictions_made = r_data["predictions_made"]
                    rule_perf.correct_predictions = r_data["correct_predictions"]
                    rule_perf.recent_accuracy = deque(r_data["recent_accuracy"], maxlen=30)
                    self.prediction_engine.rule_performance[r_id] = rule_perf
                self.prediction_engine.accuracy_history = deque(prediction_engine_state["accuracy_history"], maxlen=500)
                self.prediction_engine.phi_history = deque(prediction_engine_state["phi_history"], maxlen=500)
                self.prediction_engine.sigma_history = deque(prediction_engine_state["sigma_history"], maxlen=500)
                self.prediction_engine.total_predictions = prediction_engine_state["total_predictions"]
                self.prediction_engine.total_correct = prediction_engine_state["total_correct"]
                self.prediction_engine.total_validated = prediction_engine_state["total_validated"]
            logger.debug("Loaded Prediction Validation Engine state from Postgres.")

            # Load other RAM-only states from CognitiveIntelligentSystem
            # _observation_history
            loaded_observation_history = await self.postgres.load_observation_history()
            self.cognitive_system._observation_history = deque(loaded_observation_history, maxlen=self.cognitive_system._max_observation_history)
            logger.debug(f"Loaded {len(loaded_observation_history)} observations into history.")

            # _price_history
            loaded_price_history = await self.postgres.load_price_history()
            for symbol, prices in loaded_price_history.items():
                self.cognitive_system._price_history[symbol] = deque(prices, maxlen=self.cognitive_system._max_price_history)
            logger.debug(f"Loaded price history for {len(loaded_price_history)} symbols.")

            # _meta_domains
            loaded_meta_domains = await self.postgres.load_meta_domains()
            for meta_domain, domains in loaded_meta_domains.items():
                self.cognitive_system._meta_domains[meta_domain] = domains
            logger.debug(f"Loaded {len(loaded_meta_domains)} meta domains.")

            # learning_engine
            learning_engine_state = await self.postgres.load_learning_engine_state()
            if learning_engine_state:
                self.cognitive_system.learning_engine.weights = np.array(learning_engine_state["weights"])
                self.cognitive_system.learning_engine.bias = learning_engine_state["bias"]
                self.cognitive_system.learning_engine.feature_names = learning_engine_state["feature_names"]
                self.cognitive_system.learning_engine.feature_index = learning_engine_state["feature_index"]
                self.cognitive_system.learning_engine.long_term_patterns = {p_id: Pattern(pattern_id=p["pattern_id"], centroid=np.array(p["centroid"]), examples=[], confidence=p["confidence"], created_at=datetime.fromisoformat(p["created_at"]), hit_count=p["hit_count"]) for p_id, p in learning_engine_state["long_term_patterns"].items()}
                self.cognitive_system.learning_engine.metrics = LearningMetrics(accuracy=learning_engine_state["metrics"]["accuracy"], samples_processed=learning_engine_state["metrics"]["samples_processed"], patterns_discovered=learning_engine_state["metrics"]["patterns_discovered"], adaptations=learning_engine_state["metrics"]["adaptations"], last_update=datetime.fromisoformat(learning_engine_state["metrics"]["last_update"]))
                self.cognitive_system.learning_engine.prediction_history = deque(learning_engine_state["prediction_history"], maxlen=100)
                self.cognitive_system.learning_engine.feature_means = defaultdict(float, learning_engine_state["feature_means"])
                self.cognitive_system.learning_engine.feature_vars = defaultdict(lambda: 1.0, learning_engine_state["feature_vars"])
                self.cognitive_system.learning_engine.sample_count = learning_engine_state["sample_count"]
                self.cognitive_system.learning_engine.drift_events = deque(learning_engine_state["drift_events"], maxlen=200)
                self.cognitive_system.learning_engine._current_lr_history = deque(learning_engine_state["_current_lr_history"], maxlen=200)
            logger.debug("Loaded Learning Engine state from Postgres.")

            # Caches from CognitiveIntelligentSystem
            loaded_caches = await self.postgres.load_caches()
            self.cognitive_system._recent_analogies = deque(loaded_caches.get("_recent_analogies", []), maxlen=100)
            self.cognitive_system._recent_explanations = deque(loaded_caches.get("_recent_explanations", []), maxlen=100)
            self.cognitive_system._recent_plans = deque(loaded_caches.get("_recent_plans", []), maxlen=100)
            self.cognitive_system._causal_discovery_log = deque(loaded_caches.get("_causal_discovery_log", []), maxlen=100)
            self.cognitive_system._transfer_suggestions_cache = loaded_caches.get("_transfer_suggestions_cache", {})
            self.cognitive_system._pursuit_log = deque(loaded_caches.get("_pursuit_log", []), maxlen=100)
            self.cognitive_system.resonant_memory.load_state(loaded_caches.get("resonant_memory_state", {}))
            self.world_model.load_state(loaded_caches.get("world_model_state", {}))
            logger.debug("Loaded CognitiveIntelligentSystem caches and world model from Postgres.")

            # Load short-term memory and restore observation_count
            short_term_data = await self.postgres.load_short_term_memory()
            if short_term_data:
                stm_entries = short_term_data.get("short_term_memory", [])
                self.cognitive_system.learning_engine.short_term_memory = deque(
                    stm_entries,
                    maxlen=self.cognitive_system.learning_engine.short_term_memory.maxlen
                )
                # Restore the cumulative observation counter so the dashboard is continuous
                saved_count = short_term_data.get("observation_count", 0)
                if saved_count > 0:
                    self._observation_count = saved_count
                logger.debug(f"Loaded {len(stm_entries)} short-term memory entries; observation_count restored to {self._observation_count}.")

        if self.milvus:
            # Re-populate Milvus with concept vectors loaded from Postgres.
            # This ensures vector-similarity search works immediately after restart
            # without waiting for the abstraction engine to re-encounter every concept.
            try:
                concepts_for_milvus = [
                    {
                        'concept_id': c.concept_id,
                        'domain': list(c.parent_concepts)[0] if c.parent_concepts else 'general',
                        'signature': {
                            k: v.get('mean', 0.5) if isinstance(v, dict) else float(v)
                            for k, v in c.attributes.items()
                            if isinstance(v, (int, float, dict))
                        },
                        'confidence': c.confidence,
                    }
                    for c in self.cognitive_system.abstraction.concepts.values()
                ]
                if concepts_for_milvus:
                    restored = await self.milvus.bulk_store_concepts(concepts_for_milvus)
                    logger.info(f"Restored {restored}/{len(concepts_for_milvus)} concept vectors to Milvus from Postgres.")
                    # Wire Milvus into the abstraction engine so future concepts are stored automatically
                    self.cognitive_system.abstraction.milvus = self.milvus
                    logger.debug("Milvus wired into AbstractionEngine for live concept storage.")
            except Exception as e:
                logger.warning(f"Milvus concept restore skipped: {e}")

        # Always wire Milvus into the abstraction engine when available so that
        # concepts formed after startup are automatically stored as vectors.
        if self.milvus:
            self.cognitive_system.abstraction.milvus = self.milvus
            logger.debug("Milvus wired into AbstractionEngine.")

        logger.info("Cognitive state loaded.")

    # ──────────────────────────────────────────
    # Cognitive Loop
    # ──────────────────────────────────────────

    def _update_state_cache(self):
        """Pre-compute the full state snapshot outside of HTTP request paths.
        CRITICAL: self._lock is held for the ABSOLUTE MINIMUM time possible.
        Phase 1 only snapshots raw object references and scalar values.
        All dict-building, formatting, and sorting happens OUTSIDE the lock.
        """
        try:
            # ── PHASE 1: Snapshot ONLY raw references and scalars under lock ──
            # This is intentionally minimal — no dict building, no sorting, no formatting.
            with self._lock:
                metrics_raw = self.cognitive_system.cognitive_metrics.copy()
                # Snapshot raw engine insight dicts (these are lightweight .copy() calls)
                try:
                    abstraction_insights = self.cognitive_system.abstraction.get_insights()
                except Exception:
                    abstraction_insights = {"total_concepts": len(self.cognitive_system.abstraction.concepts)}
                try:
                    reasoning_insights = self.cognitive_system.reasoning.get_insights()
                except Exception:
                    reasoning_insights = {"total_rules": 0, "total_facts": 0}
                try:
                    cross_domain_insights = self.cognitive_system.cross_domain.get_insights()
                except Exception:
                    cross_domain_insights = {"total_domains": 0, "total_mappings": 0}
                try:
                    goal_insights = self.cognitive_system.goals.get_insights()
                except Exception:
                    goal_insights = {"total_goals": 0, "active_goals": 0, "achieved_goals": 0}
                try:
                    learning_insights = self.cognitive_system.learning_engine.get_insights()
                except Exception:
                    learning_insights = {"metrics": {"accuracy": 0, "samples_processed": 0}, "total_patterns": 0}
                try:
                    prediction_insights = self.prediction_engine.get_insights()
                except Exception:
                    prediction_insights = {"phi": 0.5, "sigma": 0.5, "global_accuracy": 0, "total_validated": 0, "total_correct": 0, "symbols_tracked": 0}

                # Snapshot raw object collections as shallow lists (no dict-building here)
                _raw_concept_items = list(self.cognitive_system.abstraction.concepts.items())
                _raw_rule_items = list(self.cognitive_system.reasoning.rules.items())[-200:]
                _raw_facts = list(self.cognitive_system.reasoning.facts)[:50]
                _raw_goal_items = list(self.cognitive_system.goals.goals.items())
                _raw_mapping_items = list(self.cognitive_system.cross_domain.mappings.items())[-100:]
                _raw_domain_items = list(self.cognitive_system.cross_domain.domains.items())
                _raw_analogies = list(self.cognitive_system._recent_analogies)[-20:]
                # Snapshot scalars
                _concepts_merged = self._concepts_merged
                _concepts_pruned = self._concepts_pruned
                _observation_count = self._observation_count
                _pending_len = len(self._pending_observations)
                _errors = self._errors
                _start_time = self._start_time
                _last_obs_time = self._last_observation_time
                _running = self._running
                _toggles = dict(self._toggles)
                _node_id = self.node_id
            # ── PHASE 2: Build ALL state dicts OUTSIDE the lock ──
            # Causal graph (has its own internal state, safe to call outside lock)
            try:
                causal_graph = self.cognitive_system.get_causal_graph_snapshot()
            except Exception:
                causal_graph = {}

            # Build concept_domain_map from raw domain items
            concept_domain_map = {}
            for domain_id, domain in _raw_domain_items:
                for concept_id in getattr(domain, 'concepts', []):
                    concept_domain_map[concept_id] = domain_id

            # Build concepts dict
            _raw_concept_items.sort(key=lambda x: getattr(x[1], 'created_at', datetime.now()), reverse=True)
            concepts = {}
            for cid, concept in _raw_concept_items[:100]:
                concepts[cid] = {
                    "id": cid,
                    "symbol": getattr(concept, 'symbol', None),
                    "domain": concept_domain_map.get(cid, getattr(concept, 'domain', 'unknown')),
                    "confidence": round(getattr(concept, 'confidence', 0), 4),
                    "observation_count": getattr(concept, 'observation_count', 0),
                    "created_at": getattr(concept, 'created_at', datetime.now()).isoformat() if hasattr(getattr(concept, 'created_at', None), 'isoformat') else str(getattr(concept, 'created_at', '')),
                }

            # Build rules dict
            rules = {}
            for rid, rule in _raw_rule_items:
                rules[rid] = {
                    "id": rid,
                    "antecedents": list(getattr(rule, 'antecedents', []))[:10],
                    "consequents": list(getattr(rule, 'consequents', []))[:10],
                    "confidence": round(getattr(rule, 'confidence', 0), 4),
                    "support": getattr(rule, 'support', 0),
                }

            # Build facts list
            facts = []
            for fact in _raw_facts:
                if isinstance(fact, dict):
                    facts.append(fact)
                elif hasattr(fact, '__dict__'):
                    facts.append(fact.__dict__)
                elif isinstance(fact, str):
                    parts = fact.split('_', 2)
                    if len(parts) == 3:
                        facts.append({"subject": parts[0], "predicate": parts[1], "object": parts[2]})
                    else:
                        facts.append({"subject": fact, "predicate": "is", "object": "true"})
                else:
                    facts.append({"subject": str(fact), "predicate": "is", "object": "true"})

            # Build goals dict
            goals = {}
            for gid, goal in _raw_goal_items:
                goals[gid] = {
                    "id": gid,
                    "description": getattr(goal, 'description', None),
                    "goal_type": getattr(goal, 'goal_type', {}).value if hasattr(getattr(goal, 'goal_type', None), 'value') else str(getattr(goal, 'goal_type', '')),
                    "status": getattr(goal, 'status', {}).value if hasattr(getattr(goal, 'status', None), 'value') else str(getattr(goal, 'status', '')),
                    "progress": round(getattr(goal, 'progress', 0), 4),
                }

            # Build cross_domain dict
            cross_domain = {}
            for mid, mapping in _raw_mapping_items:
                cross_domain[mid] = {
                    "id": mid,
                    "source_domain": getattr(mapping, 'source_domain', ''),
                    "target_domain": getattr(mapping, 'target_domain', ''),
                    "confidence": round(getattr(mapping, 'confidence', 0), 4),
                }

            # Build analogies list
            analogies = []
            for a in _raw_analogies:
                if isinstance(a, dict):
                    analogies.append(a)
                elif hasattr(a, '__dict__'):
                    analogies.append(a.__dict__)

            # ── PHASE 3: Build metrics and final state dict (all outside lock) ──
            # Prediction insights (has its own internal lock)
            recent_preds = prediction_insights.get('recent_predictions', [])
            predictions = []
            for i, p in enumerate(recent_preds):
                predictions.append({
                    "id": f"pred_{i}",
                    "symbol": p.get('symbol', '?'),
                    "type": p.get('direction', '?'),
                    "confidence": p.get('confidence', 0),
                    "horizon": p.get('horizon', '?'),
                    "outcome": 'correct' if p.get('correct') else ('pending' if not p.get('validated') else 'wrong'),
                })

            # PHI/SIGMA from prediction engine
            phi = prediction_insights.get('phi', 0.5)
            sigma = prediction_insights.get('sigma', 0.5)

            # Build metrics dict
            metrics = {
                "global_coherence_phi": round(phi, 4),
                "noise_level_sigma": round(sigma, 4),
                "prediction_accuracy": prediction_insights.get('global_accuracy', 0),
                "predictions_validated": prediction_insights.get('total_validated', 0),
                "predictions_correct": prediction_insights.get('total_correct', 0),
                "streams_tracked": prediction_insights.get('streams_tracked', 0),
                "symbols_tracked": prediction_insights.get('symbols_tracked', 0),  # backward-compat
                "rules_learned": metrics_raw.get('rules_learned', 0),
                "analogies_found": metrics_raw.get('analogies_found', 0),
                "goals_achieved": metrics_raw.get('goals_achieved', 0),
                "knowledge_transfers": metrics_raw.get('knowledge_transfers', 0),
                "causal_links_discovered": metrics_raw.get('causal_links_discovered', 0),
                "total_concepts": abstraction_insights.get('total_concepts', 0),
                "total_rules": reasoning_insights.get('total_rules', 0),
                "total_facts": reasoning_insights.get('total_facts', 0),
                "total_domains": cross_domain_insights.get('total_domains', 0),
                "total_goals": goal_insights.get('total_goals', 0),
                "active_goals": goal_insights.get('active_goals', 0),
                "achieved_goals": goal_insights.get('achieved_goals', 0),
                "total_mappings": cross_domain_insights.get('total_mappings', 0),
                "learning_accuracy": learning_insights.get('metrics', {}).get('accuracy', 0),
                "patterns_discovered": learning_insights.get('total_patterns', 0),
                "samples_processed": learning_insights.get('metrics', {}).get('samples_processed', 0),
                "concepts_merged": _concepts_merged,
                "concepts_pruned": _concepts_pruned,
                "total_observations": _observation_count,
                "pending_observations": _pending_len,
                "errors": _errors,
                "uptime_seconds": time.time() - _start_time,
                "last_observation_time": _last_obs_time,
                "cognitive_loop_running": _running,
                "attention_density": min(1.0, prediction_insights.get('streams_tracked', prediction_insights.get('symbols_tracked', 0)) / max(abstraction_insights.get('total_concepts', 1), 1)),
            }

            # Providers
            providers = {}
            if self.data_provider:
                try:
                    raw_status = self.data_provider.get_provider_status()
                    # Collect all provider categories (financial, IoT, or any other plugin)
                    all_providers = []
                    for category_providers in raw_status.values():
                        if isinstance(category_providers, list):
                            all_providers.extend(category_providers)
                    for p in all_providers:
                        name = p.get('name', '?')
                        providers[name] = {
                            "name": name,
                            "status": p.get('ui_status', p.get('state', 'unknown')),
                            "breaker_state": p.get('state', 'unknown'),
                            "enabled": p.get('enabled', True),
                            "available": p.get('available', True),
                            "assets_tracked": p.get('assets_tracked', 0),
                            "latency_ms": p.get('latency_ms', 0),
                            "request_count": p.get('request_count', 0),
                            "success_count": p.get('success_count', 0),
                            "last_symbol": p.get('last_symbol'),
                            "last_error": p.get('last_error'),
                        }
                except Exception as e:
                    logger.debug(f"Provider status error in cache: {e}")

            # Build all missing fields that endpoints and dashboard expect
            try:
                concept_hierarchy = self.cognitive_system.get_concept_hierarchy_snapshot()
            except Exception:
                concept_hierarchy = {"nodes": [], "edges": [], "total_levels": 0}

            try:
                drift_events = self.cognitive_system.get_drift_events()
            except Exception:
                drift_events = []

            try:
                explanations = self.cognitive_system.get_explanations_snapshot()
            except Exception:
                explanations = []

            try:
                feature_importances = self.cognitive_system.get_feature_importances()
            except Exception:
                feature_importances = {}

            try:
                plans = self.cognitive_system.get_plans_snapshot()
            except Exception:
                plans = []

            try:
                pursuits = self.cognitive_system.get_pursuit_log()
            except Exception:
                pursuits = []

            try:
                transfer_suggestions = self.cognitive_system.get_transfer_suggestions_snapshot()
            except Exception:
                transfer_suggestions = []

            try:
                strategy_performance = self.cognitive_system.get_strategy_performance()
            except Exception:
                strategy_performance = {}

            # Activity log for dashboard live feed
            activity_log = list(self._activity_log)

            # Prediction snapshot (structured for the predictions tab)
            prediction_snapshot = {
                "predictions": predictions,
                "metrics": {
                    "accuracy": prediction_insights.get('global_accuracy', 0),
                    "total_validated": prediction_insights.get('total_validated', 0),
                    "total_correct": prediction_insights.get('total_correct', 0),
                    "phi": round(phi, 4),
                    "sigma": round(sigma, 4),
                },
            }

            # Learning snapshot
            learning = {
                "metrics": metrics,
                "feature_importances": feature_importances,
                "drift_events": drift_events,
                "strategy_performance": strategy_performance,
            }

            # Orchestrator status
            orchestrator_status = {
                "status": "running" if self._running else "stopped",
                "node_id": _node_id,
                "uptime_seconds": round(time.time() - _start_time, 1),
                "iteration": _observation_count,
                "errors": _errors,
                "toggles": _toggles,
            }

            # Introspection snapshot (cached from last self-reflection run)
            introspection = self.get_introspection()

            # Add examples to concepts (last 3 per concept for hypothesis generation)
            for cid, concept in _raw_concept_items[:100]:
                if cid in concepts:
                    examples = getattr(concept, 'examples', [])
                    concepts[cid]["examples"] = examples[-3:] if examples else []

            state = {
                "metrics": metrics,
                "concepts": concepts,
                "rules": rules,
                "facts": facts,
                "goals": goals,
                "cross_domain": cross_domain,
                "analogies": analogies,
                "causal_graph": causal_graph,
                "predictions": predictions,
                "providers": providers,
                "toggles": _toggles,
                "node_id": _node_id,
                # Previously missing fields — now fully wired
                "concept_hierarchy": concept_hierarchy,
                "drift_events": drift_events,
                "explanations": explanations,
                "feature_importances": feature_importances,
                "plans": plans,
                "pursuits": pursuits,
                "transfer_suggestions": transfer_suggestions,
                "strategy_performance": strategy_performance,
                "log": activity_log,
                "prediction_snapshot": prediction_snapshot,
                "learning": learning,
                "orchestrator_status": orchestrator_status,
                "introspection": introspection,
                "_cache_warming": False,
            }
            with self._state_cache_lock:
                self._state_cache = state
                self._state_cache_ready = True
        except Exception as e:
            logger.error(f"Error updating state cache: {e}", exc_info=True)

    def _update_coordinator_cache(self):
        """Rebuild the state cache from the Coordinator's Z³ state dict.
        Merges coordinator state with runtime scalars and provider status.
        Called by the cognitive loop every 5s — never called from HTTP handlers.
        """
        try:
            # Get the coordinator's structured state dict
            coord_state = self.coordinator.get_state_dict()

            # coordinator.get_state_dict() returns phi/sigma as top-level keys.
            # The dashboard reads state.metrics.global_coherence_phi — map everything here.
            if "metrics" not in coord_state:
                coord_state["metrics"] = {}
            m = coord_state["metrics"]

            # ── Phi / Sigma from coordinator top-level keys ────────────────────
            m["global_coherence_phi"] = round(float(coord_state.get("phi", 0.5)), 4)
            m["noise_level_sigma"]    = round(float(coord_state.get("sigma", 0.5)), 4)

            # ── Cognitive metrics from subsystems ─────────────────────────────
            cs = self.cognitive_system
            # Prediction insights come from self.prediction_engine (on DistributedCognitiveCore)
            try:
                pred_i = self.prediction_engine.get_insights()
            except Exception:
                pred_i = {}
            try:
                abstr_i = cs.abstraction.get_insights() if hasattr(cs, 'abstraction') else {}
            except Exception:
                abstr_i = {}
            try:
                reason_i = cs.reasoning.get_insights() if hasattr(cs, 'reasoning') else {}
            except Exception:
                reason_i = {}
            try:
                # goals attribute on CognitiveIntelligentSystem is cs.goals (OpenEndedGoalSystem)
                goal_i = cs.goals.get_insights() if hasattr(cs, 'goals') else {}
            except Exception:
                goal_i = {}
            try:
                # learning engine is cs.learning_engine (ContinuousLearningEngine)
                learn_i = cs.learning_engine.get_insights() if hasattr(cs, 'learning_engine') else {}
            except Exception:
                learn_i = {}
            try:
                xd_i = cs.cross_domain.get_insights() if hasattr(cs, 'cross_domain') else {}
            except Exception:
                xd_i = {}
            try:
                cog_m = cs.cognitive_metrics if hasattr(cs, 'cognitive_metrics') else {}
            except Exception:
                cog_m = {}

            streams = pred_i.get('streams_tracked', pred_i.get('symbols_tracked', 0))
            total_concepts = max(abstr_i.get('total_concepts', 1), 1)

            m["prediction_accuracy"]     = pred_i.get('global_accuracy', 0)
            m["predictions_validated"]   = pred_i.get('total_validated', 0)
            m["predictions_correct"]     = pred_i.get('total_correct', 0)
            m["streams_tracked"]         = streams
            m["symbols_tracked"]         = pred_i.get('symbols_tracked', streams)
            m["attention_density"]       = min(1.0, streams / total_concepts)
            m["total_concepts"]          = abstr_i.get('total_concepts', 0)
            m["total_rules"]             = reason_i.get('total_rules', len(getattr(getattr(cs, 'reasoning', None), 'rules', {})))
            m["total_facts"]             = reason_i.get('total_facts', 0)
            m["total_goals"]             = goal_i.get('total_goals', 0)
            m["active_goals"]            = goal_i.get('active_goals', 0)
            m["achieved_goals"]          = goal_i.get('achieved_goals', 0)
            m["learning_accuracy"]       = learn_i.get('metrics', {}).get('accuracy', 0)
            m["patterns_discovered"]     = learn_i.get('total_patterns', 0)
            m["samples_processed"]       = learn_i.get('metrics', {}).get('samples_processed', 0)
            m["total_domains"]           = xd_i.get('total_domains', 0)
            m["total_mappings"]          = xd_i.get('total_mappings', 0)
            m["rules_learned"]           = cog_m.get('rules_learned', m.get('total_rules', 0))
            m["analogies_found"]         = cog_m.get('analogies_found', 0)
            m["goals_achieved"]          = cog_m.get('goals_achieved', 0)
            m["knowledge_transfers"]     = cog_m.get('knowledge_transfers', 0)
            m["causal_links_discovered"] = cog_m.get('causal_links_discovered', 0)
            resonance_i = cs.get_resonant_memory_snapshot() if hasattr(cs, 'get_resonant_memory_snapshot') else {}
            resonance_m = resonance_i.get('metrics', {})
            m["resonant_memory_rings"] = resonance_m.get('rings', 0)
            m["phi_access_window"] = resonance_m.get('phi_access_window', 0)
            m["average_resonance"] = resonance_m.get('average_resonance', 0)
            m["memory_reconstruction_confidence"] = resonance_m.get('last_reconstruction_confidence', 0)

            # ── Runtime scalars ───────────────────────────────────────────────
            m["total_observations"]      = self._observation_count
            m["pending_observations"]    = len(self._pending_observations)
            m["errors"]                  = self._errors
            m["uptime_seconds"]          = round(time.time() - self._start_time, 1)
            m["last_observation_time"]   = self._last_observation_time
            m["cognitive_loop_running"]  = self._running
            m["concepts_merged"]         = self._concepts_merged
            m["concepts_pruned"]         = self._concepts_pruned
            m["recursive_loss"]          = self._recursive_state.get("coherence_loss", 0.5)
            m["recursive_loss_delta"]    = self._recursive_state.get("loss_delta", 0.0)
            m["world_model_loss"]        = self._recursive_state.get("world_model_loss", 0.5)
            m["world_model_prediction_loss"] = self._recursive_state.get("world_model_prediction_loss", 0.5)
            m["world_model_reconstruction_loss"] = self._recursive_state.get("world_model_reconstruction_loss", 0.5)
            m["world_model_memory_loss"] = self._recursive_state.get("world_model_memory_loss", 1.0)

            # ── Normalize coordinator-backed cache to the dashboard schema ─────
            try:
                concepts = self.get_concepts_snapshot()
            except Exception:
                concepts = {}
            try:
                rules = self.get_rules_snapshot()
            except Exception:
                rules = {}
            try:
                goals = self.get_goals_snapshot()
            except Exception:
                goals = {}
            try:
                cross_domain = self.get_cross_domain_snapshot()
            except Exception:
                cross_domain = {}
            try:
                analogies = self.get_analogies()
            except Exception:
                analogies = []
            try:
                causal_graph = self.get_causal_graph()
            except Exception:
                causal_graph = {}
            try:
                concept_hierarchy = self.get_concept_hierarchy()
            except Exception:
                concept_hierarchy = {}
            try:
                explanations = self.get_explanations()
            except Exception:
                explanations = []
            try:
                plans = self.get_plans()
            except Exception:
                plans = []
            try:
                pursuits = self.get_pursuit_log()
            except Exception:
                pursuits = []
            try:
                transfer_suggestions = self.get_transfer_suggestions()
            except Exception:
                transfer_suggestions = []
            try:
                strategy_performance = self.get_strategy_performance()
            except Exception:
                strategy_performance = {}
            try:
                feature_importances = self.get_feature_importances()
            except Exception:
                feature_importances = {}
            try:
                drift_events = self.get_drift_events()
            except Exception:
                drift_events = []
            try:
                orchestrator_status = self.get_orchestrator_status()
            except Exception:
                orchestrator_status = {
                    "status": "running" if self._running else "stopped",
                    "node_id": self.node_id,
                }

            facts = []
            try:
                with self._lock:
                    raw_facts = list(self.cognitive_system.reasoning.facts)[:50]
                for fact in raw_facts:
                    if isinstance(fact, dict):
                        facts.append(fact)
                    elif hasattr(fact, '__dict__'):
                        facts.append(fact.__dict__)
                    elif isinstance(fact, str):
                        parts = fact.split('_', 2)
                        if len(parts) == 3:
                            facts.append({"subject": parts[0], "predicate": parts[1], "object": parts[2]})
                        else:
                            facts.append({"subject": fact, "predicate": "is", "object": "true"})
                    else:
                        facts.append({"subject": str(fact), "predicate": "is", "object": "true"})
            except Exception:
                facts = []

            recent_preds = pred_i.get('recent_predictions', []) if isinstance(pred_i, dict) else []
            predictions = []
            for i, p in enumerate(recent_preds):
                if isinstance(p, dict):
                    predictions.append({
                        "id": p.get('id', f"pred_{i}"),
                        "symbol": p.get('symbol', '?'),
                        "type": p.get('direction', p.get('type', '?')),
                        "confidence": p.get('confidence', 0),
                        "horizon": p.get('horizon', '?'),
                        "outcome": 'correct' if p.get('correct') else ('pending' if not p.get('validated') else 'wrong'),
                    })

            prediction_snapshot = {
                "predictions": predictions,
                "metrics": {
                    "accuracy": pred_i.get('global_accuracy', 0) if isinstance(pred_i, dict) else 0,
                    "total_validated": pred_i.get('total_validated', 0) if isinstance(pred_i, dict) else 0,
                    "total_correct": pred_i.get('total_correct', 0) if isinstance(pred_i, dict) else 0,
                    "phi": round(float(coord_state.get("phi", 0.5)), 4),
                    "sigma": round(float(coord_state.get("sigma", 0.5)), 4),
                },
            }

            learning = {
                "metrics": dict(m),
                "feature_importances": feature_importances,
                "drift_events": drift_events,
                "strategy_performance": strategy_performance,
            }

            coord_state["concepts"] = concepts
            coord_state["rules"] = rules
            coord_state["facts"] = facts
            coord_state["goals"] = goals
            coord_state["cross_domain"] = cross_domain
            coord_state["analogies"] = analogies
            coord_state["causal_graph"] = causal_graph
            coord_state["predictions"] = predictions
            coord_state["concept_hierarchy"] = concept_hierarchy
            coord_state["explanations"] = explanations
            coord_state["plans"] = plans
            coord_state["pursuits"] = pursuits
            coord_state["transfer_suggestions"] = transfer_suggestions
            coord_state["strategy_performance"] = strategy_performance
            coord_state["feature_importances"] = feature_importances
            coord_state["drift_events"] = drift_events
            coord_state["prediction_snapshot"] = prediction_snapshot
            coord_state["learning"] = learning
            coord_state["orchestrator_status"] = orchestrator_status
            coord_state["log"] = list(self._activity_log)
            coord_state["recursive_state"] = dict(self._recursive_state)
            coord_state["recursive_feedback"] = list(self._recursive_feedback_log)[-50:]
            coord_state["world_model"] = self.world_model.get_state()
            coord_state["_cache_warming"] = False

            # Provider status (data plane, not cognitive plane)
            providers = {}
            if self.data_provider:
                try:
                    raw_status = self.data_provider.get_provider_status()
                    all_providers = raw_status.get('stock', []) + raw_status.get('crypto', [])
                    for p in all_providers:
                        name = p.get('name', '?')
                        providers[name] = {
                            "name": name,
                            "status": p.get('ui_status', p.get('state', 'unknown')),
                            "breaker_state": p.get('state', 'unknown'),
                            "enabled": p.get('enabled', True),
                            "available": p.get('available', True),
                            "assets_tracked": p.get('assets_tracked', 0),
                            "latency_ms": p.get('latency_ms', 0),
                            "request_count": p.get('request_count', 0),
                            "success_count": p.get('success_count', 0),
                            "last_symbol": p.get('last_symbol'),
                            "last_error": p.get('last_error'),
                        }
                except Exception as e:
                    logger.debug(f"Provider status error in cache: {e}")
            coord_state["providers"] = providers
            coord_state["resonant_memory"] = resonance_i
            coord_state["toggles"] = dict(self._toggles)
            coord_state["node_id"] = self.node_id

            with self._state_cache_lock:
                self._state_cache = coord_state
                self._state_cache_ready = True
        except Exception as e:
            logger.error(f"Error updating coordinator cache: {e}", exc_info=True)
            # Fall back to old _update_state_cache if coordinator fails
            self._update_state_cache()

    def get_cached_state(self) -> Dict[str, Any]:
        """Return the pre-computed state cache.
        NEVER acquires self._lock — always returns immediately.
        Returns empty defaults while cache is warming (first ~5s after boot).
        """
        with self._state_cache_lock:
            if self._state_cache_ready:
                return dict(self._state_cache)
        # Cache not ready yet — return empty defaults immediately (NO lock acquisition)
        return {
            "metrics": {
                "global_coherence_phi": 0.5,
                "noise_level_sigma": 0.5,
                "prediction_accuracy": 0,
                "predictions_validated": 0,
                "predictions_correct": 0,
                "symbols_tracked": 0,
                "rules_learned": 0,
                "analogies_found": 0,
                "goals_achieved": 0,
                "knowledge_transfers": 0,
                "causal_links_discovered": 0,
                "total_concepts": 0,
                "total_rules": 0,
                "total_facts": 0,
                "total_domains": 0,
                "total_goals": 0,
                "active_goals": 0,
                "achieved_goals": 0,
                "total_mappings": 0,
                "learning_accuracy": 0,
                "patterns_discovered": 0,
                "samples_processed": 0,
                "concepts_merged": 0,
                "concepts_pruned": 0,
                "total_observations": self._observation_count,
                "pending_observations": len(self._pending_observations),
                "errors": self._errors,
                "uptime_seconds": time.time() - self._start_time,
                "last_observation_time": self._last_observation_time,
                "cognitive_loop_running": self._running,
                "attention_density": 0,
                "resonant_memory_rings": 0,
                "phi_access_window": 0,
                "average_resonance": 0,
                "memory_reconstruction_confidence": 0,
            },
            "concepts": {},
            "rules": {},
            "facts": [],
            "goals": {},
            "cross_domain": {},
            "analogies": [],
            "causal_graph": {},
            "predictions": [],
            "providers": {},
            "resonant_memory": {"metrics": {"rings": 0, "phi_access_window": 0, "average_resonance": 0, "last_reconstruction_confidence": 0}, "recent_rings": [], "top_matches": []},
            "toggles": dict(self._toggles),
            "recursive_state": dict(self._recursive_state),
            "recursive_feedback": list(self._recursive_feedback_log)[-50:],
            "world_model": self.world_model.get_state(),
            "node_id": self.node_id,
            "_cache_warming": True,
        }

    def start_cognitive_loop(self):
        """Start the background cognitive loop with dynamic batching"""
        if self._running:
            return

        self._running = True

        def _loop():
            logger.info("Cognitive loop thread started (Dynamic Batching Mode)")
            iteration = 0
            last_cache_update = 0
            
            while self._running:
                try:
                    iteration += 1
                    start_time = time.time()
                    
                    # 1. Process pending observations with DYNAMIC BATCHING
                    # If the queue is backed up, we process more items per loop
                    queue_size = len(self._pending_observations)
                    batch_size = 10
                    if queue_size > 50: batch_size = 20
                    if queue_size > 200: batch_size = 50
                    
                    batch = []
                    while self._pending_observations and len(batch) < batch_size:
                        batch.append(self._pending_observations.popleft())

                    if batch:
                        for obs, domain in batch:
                            try:
                                # ── Unified Mesh Pipeline ─────────────────────────────
                                # Route every queued observation through the full
                                # CognitiveIntelligentSystem pipeline first. This is the
                                # path that enriches observations, updates constitutional
                                # physics, extracts facts, runs inference, stores history,
                                # and writes resonant memory.
                                cs = self.cognitive_system
                                pipeline_result = {}
                                try:
                                    pipeline_result = cs.process_observation(obs, domain)
                                except Exception as e:
                                    logger.error(f"Unified cognitive pipeline error: {e}", exc_info=True)
                                    pipeline_result = {}

                                abstractions = []
                                concept_out = pipeline_result.get('concept') if isinstance(pipeline_result, dict) else None
                                if concept_out:
                                    abstractions = [concept_out]

                                # Prediction is owned by DistributedCognitiveCore, so it
                                # remains adjacent to the unified cognitive-system pass.
                                predictions_out = []
                                validation_result = None
                                try:
                                    validation_result = self.prediction_engine.record_observation(obs, domain)
                                    predictions_out = self.prediction_engine.get_active_predictions_as_outputs()
                                except Exception as e:
                                    logger.debug(f"Prediction error: {e}")

                                rules_out = []
                                try:
                                    rules_out = cs.reasoning.get_rules_as_outputs() if hasattr(cs.reasoning, 'get_rules_as_outputs') else []
                                except Exception as e:
                                    logger.debug(f"Reasoning output error: {e}")

                                eeg_out = None
                                if self._observation_count % 10 == 0:
                                    try:
                                        raw_eeg = cs.eeg_analyzer.get_eeg_data() if hasattr(cs, 'eeg_analyzer') else {}
                                        if raw_eeg:
                                            eeg_out = EEGOutput(
                                                phi=raw_eeg.get('phi', 0.5),
                                                sigma=raw_eeg.get('sigma', 0.5),
                                                dominant_frequency=raw_eeg.get('dominant_frequency', 'gamma'),
                                                band_power=raw_eeg.get('band_power', {}),
                                                phase_lock_pairs=raw_eeg.get('phase_lock_pairs', []),
                                                attention_score=raw_eeg.get('attention_score', 0.5),
                                                coherence_map=raw_eeg.get('coherence_map', {}),
                                            )
                                    except Exception as e:
                                        logger.debug(f"EEG output error: {e}")

                                transfers_out = []
                                try:
                                    transfers_out = cs.cross_domain.get_transfers_as_outputs() if hasattr(cs.cross_domain, 'get_transfers_as_outputs') else []
                                except Exception as e:
                                    logger.debug(f"Cross-domain output error: {e}")

                                goals_out = []
                                try:
                                    goals_out = cs.goals.get_active_goals_as_outputs() if hasattr(cs.goals, 'get_active_goals_as_outputs') else []
                                except Exception as e:
                                    logger.debug(f"Goals output error: {e}")

                                learning_out = []
                                try:
                                    learning_out = cs.learning_engine.get_patterns_as_outputs() if hasattr(cs.learning_engine, 'get_patterns_as_outputs') else []
                                except Exception as e:
                                    logger.debug(f"Learning output error: {e}")

                                # ── Coordinator: collect all outputs, update Z³ state ──
                                self.coordinator.coordinate(
                                    constitutional=cs.get_constitutional_output() if hasattr(cs, 'get_constitutional_output') else None,
                                    abstractions=abstractions,
                                    rules=rules_out,
                                    predictions=predictions_out,
                                    eeg=eeg_out,
                                    cross_domain_transfers=transfers_out,
                                    active_goals=goals_out,
                                    learning_patterns=learning_out,
                                )

                                world_model_output = self.world_model.observe(
                                    obs,
                                    domain=domain,
                                    coordinator_state=self.coordinator.state,
                                )
                                self._apply_recursive_feedback(
                                    obs=obs,
                                    domain=domain,
                                    pipeline_result=pipeline_result,
                                    validation_result=validation_result,
                                    world_model_output=world_model_output.to_dict(),
                                )

                            except Exception as e:
                                self._errors += 1
                                logger.error(f"Error processing observation: {e}")

                    # 1b. Update state cache after a processed batch, every 5s,
                    # or when the queue is large. This keeps recursive/world-model
                    # metrics visible to API readers without waiting for the next
                    # maintenance interval.
                    now = time.time()
                    if batch or (now - last_cache_update > 5.0) or (queue_size > 100 and iteration % 20 == 0):
                        self._update_coordinator_cache()
                        last_cache_update = now

                    # 2. Perform periodic maintenance (every ~10s)
                    if iteration % 100 == 0:
                        with self._lock:
                            # Feed prediction accuracy back to rules
                            self._feed_rule_confidence_back()

                            # Mine association rules from accumulated observations
                            obs_history = list(self.cognitive_system._observation_history)
                            if len(obs_history) >= 3:
                                try:
                                    before = len(self.cognitive_system.reasoning.rules)
                                    self.cognitive_system.reasoning.learn_rules_from_observations(
                                        obs_history, min_support=3
                                    )
                                    after = len(self.cognitive_system.reasoning.rules)
                                    if after > before:
                                        self.cognitive_system.cognitive_metrics['rules_learned'] = after
                                        logger.info(f"Rule mining: {after - before} new rules (total={after})")
                                        self._activity_log.append({"ts": time.time() * 1000, "msg": f"Rule mining: {after - before} new rules learned (total={after})"})
                                except Exception as e:
                                    logger.error(f"Rule learning error: {e}")

                            # Converge concepts (merge/prune)
                            self._run_concept_convergence()

                    # 3. Self-Reflection + Goal Formation (every ~30s)
                    if iteration % 300 == 0 and self._toggles.get('deep_introspection', True):
                        with self._lock:
                            try:
                                # ── A. Trigger goal formation with real observations ──
                                ctx = self._build_goal_context()
                                new_goals = self.cognitive_system.goals.generate_goals(ctx)
                                if new_goals:
                                    logger.info(f"Goal formation: {len(new_goals)} new goals generated")
                                    for g in new_goals[:3]:
                                        self._activity_log.append({"ts": time.time() * 1000, "msg": f"Goal formed: {getattr(g, 'description', str(g))[:80]}"})

                                # ── B. Genuine self-reflection: observe, compare, act ──
                                self._run_self_reflection()
                            except Exception as e:
                                logger.error(f"Error in self-reflection: {e}")

                    # 4. Self-evolution (every ~5 min)
                    if iteration % 3000 == 0 and self._toggles.get('self_evolution', True):
                        try:
                            self._run_self_evolution()
                            self._activity_log.append({"ts": time.time() * 1000, "msg": "Self-evolution cycle complete"})
                        except Exception as e:
                            logger.error(f"Error in self-evolution: {e}")

                    # Performance monitoring
                    elapsed = time.time() - start_time
                    if elapsed > 1.0:
                        logger.warning(f"Cognitive loop slow: {elapsed:.2f}s (Batch: {len(batch)}, Queue: {queue_size})")

                    # Adaptive sleep: sleep less if the queue is backed up
                    sleep_time = 0.1
                    if queue_size > 100: sleep_time = 0.01
                    if queue_size > 500: sleep_time = 0.001
                    
                    time.sleep(sleep_time)

                except Exception as e:
                    logger.error(f"Critical error in cognitive loop iteration {iteration}: {e}")
                    time.sleep(1)

            logger.info("Cognitive loop stopped")

        self._cognitive_thread = threading.Thread(target=_loop, daemon=True, name="CognitiveLoop")
        self._cognitive_thread.start()

    def stop_cognitive_loop(self):
        """Stop the cognitive loop"""
        self._running = False
        if self._cognitive_thread:
            self._cognitive_thread.join(timeout=10)
            logger.info("Cognitive loop thread stopped")

    # ──────────────────────────────────────────
    # Recursive Integration Feedback
    # ──────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _apply_recursive_feedback(
        self,
        obs: Dict[str, Any],
        domain: str,
        pipeline_result: Dict[str, Any],
        validation_result: Optional[Dict[str, Any]] = None,
        world_model_output: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Close the loop between module outputs and future module behavior."""
        try:
            cs = self.cognitive_system
            coord = self.coordinator.state
            constitutional = (pipeline_result or {}).get("constitutional", {}) or {}
            resonance = (pipeline_result or {}).get("resonance", {}) or {}

            phi = self._clamp(getattr(coord, "phi", constitutional.get("phi", 0.5)), 0.0, 1.0)
            sigma = self._clamp(getattr(coord, "sigma", constitutional.get("sigma", 0.5)), 0.0, 1.0)
            drift = float(getattr(coord, "drift_vector", constitutional.get("drift", 0.0)) or 0.0)

            try:
                pred_i = self.prediction_engine.get_insights() or {}
            except Exception:
                pred_i = {}
            accuracy = float(pred_i.get("global_accuracy", 0.0) or 0.0)
            validated = int(pred_i.get("total_validated", 0) or 0)
            prediction_error = (1.0 - accuracy) if validated > 0 else 0.5

            memory_conf = float(
                resonance.get("reconstruction_confidence", resonance.get("last_reconstruction_confidence", 0.0)) or 0.0
            )
            wm = world_model_output or {}
            wm_loss = float(wm.get("total_loss", 0.5) or 0.5)
            wm_prediction_loss = float(wm.get("prediction_loss", 0.5) or 0.5)
            wm_reconstruction_loss = float(wm.get("reconstruction_loss", 0.5) or 0.5)
            wm_memory_loss = float(wm.get("memory_loss", 1.0) or 1.0)
            wm_coherence_loss = float(wm.get("coherence_alignment_loss", 0.5) or 0.5)

            loss = self._clamp(
                (0.25 * (1.0 - phi))
                + (0.20 * sigma)
                + (0.15 * abs(drift))
                + (0.10 * prediction_error)
                + (0.05 * (1.0 - self._clamp(memory_conf, 0.0, 1.0)))
                + (0.25 * self._clamp(wm_loss, 0.0, 1.0)),
                0.0,
                1.0,
            )
            prev_loss = float(self._recursive_state.get("coherence_loss", loss) or loss)
            loss_delta = loss - prev_loss

            actions = []

            le = cs.learning_engine
            old_lr = float(getattr(le, "learning_rate", 0.01))
            new_lr = old_lr
            if loss > 0.55 or loss_delta > 0.03 or sigma > 0.60 or wm_prediction_loss > 0.65:
                new_lr = self._clamp((old_lr * 1.03) + 0.0005, 0.001, 0.05)
            elif loss < 0.35 and loss_delta <= 0.0:
                new_lr = self._clamp(old_lr * 0.995, 0.001, 0.05)
            if abs(new_lr - old_lr) > 1e-9:
                le.learning_rate = new_lr
                actions.append(f"learning_rate {old_lr:.5f}->{new_lr:.5f}")

            abst = cs.abstraction
            obs_count = max(len(getattr(cs, "_observation_history", []) or []), 1)
            concept_count = len(getattr(abst, "concepts", {}) or {})
            concept_pressure = concept_count / obs_count
            old_threshold = float(getattr(abst, "similarity_threshold", 0.85))
            new_threshold = old_threshold
            if obs_count >= 10 and concept_pressure > 0.75:
                new_threshold = self._clamp(old_threshold + 0.01, 0.50, 0.95)
            elif obs_count >= 20 and concept_pressure < 0.10:
                new_threshold = self._clamp(old_threshold - 0.01, 0.50, 0.95)
            if abs(new_threshold - old_threshold) > 1e-9:
                abst.similarity_threshold = new_threshold
                actions.append(f"abstraction_threshold {old_threshold:.3f}->{new_threshold:.3f}")

            if validation_result:
                entity = validation_result.get("entity_id") or obs.get("entity_id") or domain
                correct = bool(validation_result.get("correct"))
                fact = f"prediction_{'correct' if correct else 'incorrect'}({entity})"
                try:
                    cs.reasoning.assert_fact(fact)
                    actions.append(f"assert_fact {fact}")
                except Exception:
                    pass
                if not correct:
                    old_explore = float(getattr(cs.goals, "exploration_rate", 0.3))
                    cs.goals.exploration_rate = self._clamp(old_explore + 0.01, 0.05, 0.80)
                    actions.append(f"exploration_rate {old_explore:.3f}->{cs.goals.exploration_rate:.3f}")

            should_goal_cycle = (
                self._observation_count % 25 == 0
                or phi < 0.45
                or wm_loss > 0.70
                or (not cs.goals.active_goals and obs_count >= 10)
            )
            if should_goal_cycle:
                try:
                    before_goals = len(cs.goals.goals)
                    ctx = self._build_goal_context()
                    cs.goals.generate_goals(ctx)
                    cs.goals.select_active_goals()
                    added = len(cs.goals.goals) - before_goals
                    if added > 0:
                        actions.append(f"generated_goals +{added}")
                except Exception as e:
                    logger.debug(f"Recursive goal cycle skipped: {e}")

            goal_metrics = {
                "constitutional_phi": phi,
                "constitutional_sigma": sigma,
                "constitutional_drift": abs(drift),
                "prediction_accuracy": accuracy,
                "total_concepts": concept_count,
                "total_rules": len(getattr(cs.reasoning, "rules", {}) or {}),
                "memory_reconstruction_confidence": memory_conf,
                "recursive_loss": loss,
                "world_model_loss": wm_loss,
                "world_model_prediction_loss": wm_prediction_loss,
                "world_model_reconstruction_loss": wm_reconstruction_loss,
                "world_model_memory_loss": wm_memory_loss,
                "world_model_coherence_loss": wm_coherence_loss,
            }
            for gid in list(getattr(cs.goals, "active_goals", []) or []):
                try:
                    goal = cs.goals.goals.get(gid)
                    before = goal.progress if goal is not None else None
                    cs.goals.update_goal_progress(gid, goal_metrics)
                    goal = cs.goals.goals.get(gid)
                    after = goal.progress if goal is not None else None
                    if before is not None and after is not None and after != before:
                        actions.append(f"goal_progress {gid} {before:.2f}->{after:.2f}")
                except Exception:
                    pass

            self._recursive_state = {
                "iteration": getattr(coord, "iteration", 0),
                "coherence_loss": round(loss, 6),
                "loss_delta": round(loss_delta, 6),
                "last_entity_id": obs.get("entity_id") or obs.get("symbol") or domain,
                "last_domain": domain,
                "phi": round(phi, 6),
                "sigma": round(sigma, 6),
                "drift": round(drift, 6),
                "prediction_accuracy": round(accuracy, 6),
                "memory_reconstruction_confidence": round(memory_conf, 6),
                "world_model_loss": round(wm_loss, 6),
                "world_model_prediction_loss": round(wm_prediction_loss, 6),
                "world_model_reconstruction_loss": round(wm_reconstruction_loss, 6),
                "world_model_memory_loss": round(wm_memory_loss, 6),
                "world_model_coherence_loss": round(wm_coherence_loss, 6),
                "world_model_latent_state": wm.get("latent_state", []),
                "active_goals": len(getattr(cs.goals, "active_goals", []) or []),
                "control": {
                    "learning_rate": round(float(getattr(le, "learning_rate", 0.0)), 8),
                    "abstraction_similarity_threshold": round(float(getattr(abst, "similarity_threshold", 0.0)), 6),
                    "goal_exploration_rate": round(float(getattr(cs.goals, "exploration_rate", 0.0)), 6),
                    "concept_pressure": round(concept_pressure, 6),
                },
            }

            if actions or self._observation_count % 25 == 0:
                event = {
                    "ts": time.time(),
                    "observation_count": self._observation_count,
                    "domain": domain,
                    "loss": round(loss, 6),
                    "phi": round(phi, 6),
                    "sigma": round(sigma, 6),
                    "world_model_loss": round(wm_loss, 6),
                    "actions": actions,
                }
                self._recursive_feedback_log.append(event)
                if actions:
                    self._activity_log.append({
                        "ts": time.time() * 1000,
                        "msg": "Recursive feedback: " + "; ".join(actions[:4]),
                    })
        except Exception as e:
            logger.debug(f"Recursive feedback error: {e}")

    # ──────────────────────────────────────────
    # Rule Confidence Feedback
    # ──────────────────────────────────────────

    def _feed_rule_confidence_back(self):
        """
        Feed prediction accuracy back into rule confidence.
        """
        adjustments = self.prediction_engine.get_rule_confidence_adjustments()
        if not adjustments:
            return

        adjusted_count = 0
        for rule_id, accuracy in adjustments.items():
            for rid, rule in self.cognitive_system.reasoning.rules.items():
                rule_conf = getattr(rule, 'confidence', 0.5)
                new_conf = accuracy * 0.7 + rule_conf * 0.3
                new_conf = max(0.05, min(0.99, new_conf))

                if abs(new_conf - rule_conf) > 0.01:
                    rule.confidence = new_conf
                    adjusted_count += 1

        if adjusted_count > 0:
            logger.info(f"Adjusted confidence for {adjusted_count} rules based on prediction accuracy")

    # ──────────────────────────────────────────
    # Concept Convergence
    # ──────────────────────────────────────────

    def _run_concept_convergence(self):
        """
        Merge similar concepts and prune stale ones.
        """
        self._convergence_counter += 1
        concepts = self.cognitive_system.abstraction.concepts

        if len(concepts) < 5:
            return

        # Group concepts by their primary entity identifier (domain-agnostic)
        entity_concepts: Dict[str, List[str]] = defaultdict(list)
        for cid, concept in concepts.items():
            examples = getattr(concept, 'examples', [])
            for ex in examples[-3:]:
                if isinstance(ex, dict):
                    # Support generic entity_id, legacy symbol, or sensor_id
                    eid = ex.get('entity_id') or ex.get('symbol') or ex.get('sensor_id')
                    if eid:
                        entity_concepts[str(eid)].append(cid)
                        break

        # Backward-compat alias
        symbol_concepts = entity_concepts

        merged = 0
        for symbol, cids in entity_concepts.items():
            if len(cids) <= 1:
                continue

            concept_sizes = [(cid, len(getattr(concepts[cid], 'examples', [])))
                           for cid in cids if cid in concepts]
            if len(concept_sizes) < 2:
                continue

            concept_sizes.sort(key=lambda x: x[1], reverse=True)
            primary_cid = concept_sizes[0][0]
            primary = concepts[primary_cid]

            for cid, size in concept_sizes[1:]:
                if cid not in concepts:
                    continue
                secondary = concepts[cid]
                primary_examples = getattr(primary, 'examples', [])
                secondary_examples = getattr(secondary, 'examples', [])
                for ex in secondary_examples:
                    if len(primary_examples) < 100:
                        primary_examples.append(ex)
                primary.confidence = min(0.99, getattr(primary, 'confidence', 0.5) + 0.05)
                del concepts[cid]
                merged += 1

        if merged > 0:
            self._concepts_merged += merged
            logger.info(f"Concept convergence: merged {merged} duplicate concepts")

        pruned = 0
        stale_ids = []
        for cid, concept in concepts.items():
            examples = getattr(concept, 'examples', [])
            confidence = getattr(concept, 'confidence', 0)
            age = (datetime.now() - getattr(concept, 'created_at', datetime.now())).total_seconds()
            if confidence < 0.15 and len(examples) <= 1 and age > 300:
                stale_ids.append(cid)

        for cid in stale_ids:
            if cid in concepts:
                del concepts[cid]
                pruned += 1

        if pruned > 0:
            self._concepts_pruned += pruned
            logger.info(f"Concept convergence: pruned {pruned} stale concepts")

    # ──────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────

    async def ingest(self, observation: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Async ingestion endpoint.
        """
        self._observation_count += 1
        self._last_observation_time = time.time()

        with self._lock:
            if domain not in self.cognitive_system.cross_domain.domains:
                self.cognitive_system.cross_domain.register_domain(domain, domain)
            # Extract meta-domain: the part before the first ':' (e.g. 'stock', 'crypto', 'iot', 'weather')
            meta = domain.split(':')[0] if ':' in domain else None
            if meta:
                self.cognitive_system._ensure_meta_domain(meta)
                self.cognitive_system._meta_domains[meta].add(domain)

        self._pending_observations.append((observation.copy(), domain))
        return {
            "success": True,
            "queued": True,
            "observation_count": self._observation_count,
            "pending": len(self._pending_observations),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # ──────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics"""
        with self._lock:
            cognitive_metrics = self.cognitive_system.cognitive_metrics.copy()
            try:
                abstraction_insights = self.cognitive_system.abstraction.get_insights()
            except Exception:
                abstraction_insights = {"total_concepts": len(self.cognitive_system.abstraction.concepts)}
            try:
                reasoning_insights = self.cognitive_system.reasoning.get_insights()
            except Exception:
                reasoning_insights = {"total_rules": 0, "total_facts": 0}
            try:
                cross_domain_insights = self.cognitive_system.cross_domain.get_insights()
            except Exception:
                cross_domain_insights = {"total_domains": 0, "total_mappings": 0}
            try:
                goal_insights = self.cognitive_system.goals.get_insights()
            except Exception:
                goal_insights = {"total_goals": 0, "active_goals": 0, "achieved_goals": 0}
            try:
                learning_insights = self.cognitive_system.learning_engine.get_insights()
            except Exception:
                learning_insights = {"metrics": {"accuracy": 0, "samples_processed": 0}, "total_patterns": 0}
            try:
                prediction_insights = self.prediction_engine.get_insights()
            except Exception:
                prediction_insights = {"phi": 0.5, "sigma": 0.5, "global_accuracy": 0, "total_validated": 0, "total_correct": 0, "symbols_tracked": 0}
        phi = prediction_insights.get('phi', 0.5)
        sigma = prediction_insights.get('sigma', 0.5)

        return {
            "global_coherence_phi": round(phi, 4),
            "noise_level_sigma": round(sigma, 4),
            "prediction_accuracy": prediction_insights.get('global_accuracy', 0),
            "predictions_validated": prediction_insights.get('total_validated', 0),
            "predictions_correct": prediction_insights.get('total_correct', 0),
            "symbols_tracked": prediction_insights.get('symbols_tracked', 0),
            "rules_learned": cognitive_metrics.get('rules_learned', 0),
            "analogies_found": cognitive_metrics.get('analogies_found', 0),
            "goals_achieved": cognitive_metrics.get('goals_achieved', 0),
            "knowledge_transfers": cognitive_metrics.get('knowledge_transfers', 0),
            "causal_links_discovered": cognitive_metrics.get('causal_links_discovered', 0),
            "total_concepts": abstraction_insights.get('total_concepts', 0),
            "total_rules": reasoning_insights.get('total_rules', 0),
            "total_facts": reasoning_insights.get('total_facts', 0),
            "total_domains": cross_domain_insights.get('total_domains', 0),
            "total_goals": goal_insights.get('total_goals', 0),
            "active_goals": goal_insights.get('active_goals', 0),
            "achieved_goals": goal_insights.get('achieved_goals', 0),
            "total_mappings": cross_domain_insights.get('total_mappings', 0),
            "learning_accuracy": learning_insights.get('metrics', {}).get('accuracy', 0),
            "patterns_discovered": learning_insights.get('total_patterns', 0),
            "samples_processed": learning_insights.get('metrics', {}).get('samples_processed', 0),
            "concepts_merged": self._concepts_merged,
            "concepts_pruned": self._concepts_pruned,
            "total_observations": self._observation_count,
            "pending_observations": len(self._pending_observations),
            "errors": self._errors,
            "uptime_seconds": time.time() - self._start_time,
            "last_observation_time": self._last_observation_time,
            "cognitive_loop_running": self._running,
            "transfers_made": cognitive_metrics.get('knowledge_transfers', 0),
        }

    # ──────────────────────────────────────────
    # Snapshots
    # ──────────────────────────────────────────
    def get_introspection(self) -> Dict[str, Any]:
        """Full system introspection from all cognitive engines."""
        with self._lock:
            try:
                abstraction_insights = self.cognitive_system.abstraction.get_insights()
            except Exception:
                abstraction_insights = {}
            try:
                reasoning_insights = self.cognitive_system.reasoning.get_insights()
            except Exception:
                reasoning_insights = {}
            try:
                cross_domain_insights = self.cognitive_system.cross_domain.get_insights()
            except Exception:
                cross_domain_insights = {}
            try:
                goal_insights = self.cognitive_system.goals.get_insights()
            except Exception:
                goal_insights = {}
            try:
                learning_insights = self.cognitive_system.learning_engine.get_insights()
            except Exception:
                learning_insights = {}
            try:
                prediction_insights = self.prediction_engine.get_insights()
            except Exception:
                prediction_insights = {}
        return {
            "abstraction": abstraction_insights,
            "reasoning": reasoning_insights,
            "cross_domain": cross_domain_insights,
            "goals": goal_insights,
            "learning": learning_insights,
            "predictions": prediction_insights,
            "node_id": self.node_id,
            "uptime_seconds": time.time() - self._start_time,
        }

    def get_concepts_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            concepts = {}
            concept_domain_map = {}
            # Use list() to avoid iteration errors
            domain_items = list(self.cognitive_system.cross_domain.domains.items())
            for domain_id, domain in domain_items:
                for concept_id in getattr(domain, 'concepts', []):
                    concept_domain_map[concept_id] = domain_id

            concept_items = list(self.cognitive_system.abstraction.concepts.items())
            # Limit to 100 most recent/relevant concepts to prevent 504 timeouts
            concept_items.sort(key=lambda x: getattr(x[1], 'created_at', datetime.now()), reverse=True)
            for cid, concept in concept_items[:100]:
                examples = getattr(concept, 'examples', [])
                recent_examples = [ex for ex in examples[-5:]]
                
                # Extract entity identifier (domain-agnostic)
                entity_id = None
                for ex in examples[-3:]:
                    if isinstance(ex, dict):
                        entity_id = ex.get('entity_id') or ex.get('symbol') or ex.get('sensor_id')
                        if entity_id:
                            entity_id = str(entity_id)
                            break

                pred_data = {}
                if entity_id and entity_id in self.prediction_engine.streams:
                    stream_hist = self.prediction_engine.streams[entity_id]
                    pred_data = {
                        "prediction_accuracy": round(stream_hist.accuracy, 4),
                        "total_predictions": stream_hist.total_predictions,
                        "trend": stream_hist.value_trend.value if stream_hist.value_trend else "unknown",
                        "momentum": round(stream_hist.momentum, 4),
                        "volatility": round(sym_hist.volatility, 6),
                    }

                concepts[cid] = {
                    "id": cid,
                    "name": getattr(concept, 'name', cid),
                    "domain": concept_domain_map.get(cid, 'unknown'),
                    "entity_id": entity_id,
                    "symbol": entity_id,  # backward-compat alias
                    "confidence": round(getattr(concept, 'confidence', 0), 4),
                    "level": getattr(concept, 'level', 0),
                    "observation_count": len(examples),
                    "examples": recent_examples,
                    "prediction": pred_data,
                    "created_at": getattr(concept, 'created_at', None),
                }
            return concepts

    def get_rules_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            rules = {}
            # Create a static list of items to avoid "dictionary changed size during iteration"
            items = list(self.cognitive_system.reasoning.rules.items())
            for rid, rule in items:
                rules[rid] = {
                    "id": rid,
                    "rule_type": _serialize_enum(getattr(rule, 'rule_type', 'unknown')),
                    "confidence": getattr(rule, 'confidence', 0),
                    "support": getattr(rule, 'support', 0),
                    "antecedents": list(getattr(rule, 'antecedents', [])),
                    "consequents": list(getattr(rule, 'consequents', [])),
                }
            return rules

    def get_goals_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            goals = {}
            # Create a static list of items to avoid "dictionary changed size during iteration"
            items = list(self.cognitive_system.goals.goals.items())
            for gid, goal in items:
                goals[gid] = {
                    "id": gid,
                    "description": getattr(goal, 'description', ''),
                    "goal_type": _serialize_enum(getattr(goal, 'goal_type', 'unknown')),
                    "status": str(getattr(goal, 'status', 'unknown')),
                    "priority": getattr(goal, 'priority', 0),
                    "progress": getattr(goal, 'progress', 0),
                }
            return goals

    def get_cross_domain_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            mappings = {}
            for mid, mapping in self.cognitive_system.cross_domain.mappings.items():
                mappings[mid] = {
                    "id": mid,
                    "source_domain": getattr(mapping, 'source_domain', ''),
                    "target_domain": getattr(mapping, 'target_domain', ''),
                    "confidence": getattr(mapping, 'confidence', 0),
                }
            return mappings

    def get_learning_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.cognitive_system.learning_engine.get_insights()

    def get_prediction_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.prediction_engine.get_insights()

    def get_state_summary(self) -> Dict[str, Any]:
        metrics = self.get_metrics()
        return {
            "node_id": self.node_id,
            "phi": metrics["global_coherence_phi"],
            "sigma": metrics["noise_level_sigma"],
            "metrics": metrics,
            "timestamp": time.time()
        }

    # Hidden Intelligence Snapshots
    def get_causal_graph(self) -> Dict[str, Any]:
        with self._lock:
            return self.cognitive_system.get_causal_graph_snapshot()

    def get_concept_hierarchy(self) -> Dict[str, Any]:
        with self._lock:
            return self.cognitive_system.get_concept_hierarchy_snapshot()

    def get_analogies(self) -> list:
        with self._lock:
            return self.cognitive_system.get_analogies_snapshot()

    def get_explanations(self) -> list:
        with self._lock:
            return self.cognitive_system.get_explanations_snapshot()

    def get_plans(self) -> list:
        with self._lock:
            return self.cognitive_system.get_plans_snapshot()

    def get_pursuit_log(self) -> list:
        with self._lock:
            return self.cognitive_system.get_pursuit_log()

    def get_strategy_performance(self) -> Dict[str, Any]:
        with self._lock:
            return self.cognitive_system.get_strategy_performance()

    def get_feature_importances(self) -> Dict[str, float]:
        with self._lock:
            return self.cognitive_system.get_feature_importances()

    def get_drift_events(self) -> list:
        with self._lock:
            return self.cognitive_system.get_drift_events()

    def get_orchestrator_status(self) -> Dict[str, Any]:
        with self._lock:
            return self.cognitive_system.orchestrator.get_status()

    def get_transfer_suggestions(self) -> list:
        with self._lock:
            return self.cognitive_system.get_transfer_suggestions_snapshot()

    def get_toggles(self) -> Dict[str, Any]:
        """Return current toggle states"""
        return dict(self._toggles)

    def set_toggle(self, key: str, value: Any) -> Dict[str, Any]:
        """Set a toggle value and return the updated toggles dict"""
        if key in self._toggles:
            self._toggles[key] = value
            logger.info(f"Toggle '{key}' set to {value}")
            return {"success": True, "key": key, "value": value, "toggles": dict(self._toggles)}
        else:
            return {"success": False, "error": f"Unknown toggle '{key}'", "toggles": dict(self._toggles)}

    def apply_goal_control(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bounded runtime control changes from an active goal.

        This is intentionally conservative: goals may tune existing module
        controls, but they may not execute arbitrary code or alter providers.
        """
        title = str(goal.get("goal") or goal.get("description") or "").lower()
        changes = []
        cs = self.cognitive_system

        try:
            if any(k in title for k in ("coherence", "stabilize", "stability", "information flow", "optimize")):
                old = float(getattr(cs.learning_engine, "learning_rate", 0.01))
                new = self._clamp(old * 0.99, 0.001, 0.05)
                cs.learning_engine.learning_rate = new
                self._toggles["concept_convergence"] = True
                self._toggles["deep_introspection"] = True
                changes.append(f"learning_rate {old:.5f}->{new:.5f}")
                changes.append("concept_convergence=True")
                changes.append("deep_introspection=True")

            if any(k in title for k in ("pattern", "recognition", "concept", "abstraction")):
                old = float(getattr(cs.abstraction, "similarity_threshold", 0.85))
                new = self._clamp(old - 0.005, 0.50, 0.95)
                cs.abstraction.similarity_threshold = new
                changes.append(f"abstraction_threshold {old:.3f}->{new:.3f}")

            if any(k in title for k in ("exploration", "exploit", "undersampled", "feature space")):
                old = float(getattr(cs.goals, "exploration_rate", 0.3))
                new = self._clamp(old + 0.02, 0.05, 0.80)
                cs.goals.exploration_rate = new
                changes.append(f"exploration_rate {old:.3f}->{new:.3f}")

            if any(k in title for k in ("cross-domain", "transfer", "domain")):
                self._toggles["knowledge_transfer"] = True
                changes.append("knowledge_transfer=True")

            if changes:
                event = {
                    "ts": time.time(),
                    "observation_count": self._observation_count,
                    "domain": "goal_control",
                    "loss": self._recursive_state.get("coherence_loss", 0.5),
                    "phi": self._recursive_state.get("phi", 0.5),
                    "sigma": self._recursive_state.get("sigma", 0.5),
                    "actions": [f"goal:{goal.get('id', goal.get('goal_id', '?'))}"] + changes,
                }
                self._recursive_feedback_log.append(event)
                self._activity_log.append({
                    "ts": time.time() * 1000,
                    "msg": "Goal control: " + "; ".join(changes[:4]),
                })

            return {"success": True, "changes": changes, "goal": goal.get("goal") or goal.get("description")}
        except Exception as e:
            logger.warning(f"Goal control failed: {e}")
            return {"success": False, "error": str(e), "changes": changes}

    # ──────────────────────────────────────────
    # Self-Reflection
    # ──────────────────────────────────────────

    def _run_self_reflection(self):
        """
        The closed self-reflection loop:
          1. Call get_introspection() to snapshot the full system state.
          2. Compare the snapshot against the stored baseline.
          3. Act on what changed — adjust exploration rate, rule-mining
             frequency, concept-convergence aggressiveness, and goal
             strategy weights based on measured performance deltas.
          4. Store the snapshot in history so the system can track its
             own trajectory over time.
        """
        try:
            snapshot = self.get_introspection()
        except Exception as e:
            logger.warning(f"Self-reflection: get_introspection failed: {e}")
            return

        # ── Store in rolling history ─────────────────────────────────────────
        import time as _time
        snapshot['_ts'] = _time.time()
        self._introspection_history.append(snapshot)

        # If no baseline yet, set it and return—nothing to compare against.
        if not self._introspection_baseline:
            self._introspection_baseline = snapshot
            logger.info("Self-reflection: baseline established")
            return

        baseline = self._introspection_baseline
        actions_taken = []

        # ── 1. Prediction accuracy trend ──────────────────────────────────
        current_accuracy = snapshot.get('predictions', {}).get('overall_accuracy', 0.5)
        baseline_accuracy = baseline.get('predictions', {}).get('overall_accuracy', 0.5)
        accuracy_delta = current_accuracy - baseline_accuracy

        goals_sys = self.cognitive_system.goals
        if accuracy_delta < -0.05:
            # Accuracy is declining — increase exploration to find better strategies
            old_rate = goals_sys.exploration_rate
            goals_sys.exploration_rate = min(0.7, goals_sys.exploration_rate + 0.05)
            actions_taken.append(
                f"exploration_rate {old_rate:.2f}→{goals_sys.exploration_rate:.2f} "
                f"(accuracy delta {accuracy_delta:+.3f})"
            )
        elif accuracy_delta > 0.05:
            # Accuracy is improving — reduce exploration, exploit what's working
            old_rate = goals_sys.exploration_rate
            goals_sys.exploration_rate = max(0.1, goals_sys.exploration_rate - 0.03)
            actions_taken.append(
                f"exploration_rate {old_rate:.2f}→{goals_sys.exploration_rate:.2f} "
                f"(accuracy delta {accuracy_delta:+.3f})"
            )

        # ── 2. Rule count trend ─────────────────────────────────────────────
        current_rules = snapshot.get('reasoning', {}).get('total_rules', 0)
        baseline_rules = baseline.get('reasoning', {}).get('total_rules', 0)
        if current_rules == baseline_rules and current_rules < 5:
            # No new rules being learned — lower the min_support threshold
            # by signalling the learning engine to be more permissive
            le = self.cognitive_system.learning_engine
            if hasattr(le, '_min_support_override'):
                le._min_support_override = max(2, getattr(le, '_min_support_override', 3) - 1)
            else:
                le._min_support_override = 2
            actions_taken.append(
                f"min_support_override→{le._min_support_override} "
                f"(rules stagnant at {current_rules})"
            )

        # ── 3. Concept count trend ──────────────────────────────────────────
        current_concepts = snapshot.get('abstraction', {}).get('total_concepts', 0)
        baseline_concepts = baseline.get('abstraction', {}).get('total_concepts', 0)
        concept_growth_rate = (current_concepts - baseline_concepts) / max(baseline_concepts, 1)

        if concept_growth_rate > 0.5:
            # Concept explosion — tighten convergence to merge similar concepts
            self._toggles['concept_convergence'] = True
            actions_taken.append(
                f"concept_convergence=True (growth rate {concept_growth_rate:.2f})"
            )
        elif concept_growth_rate < 0.01 and current_concepts < 10:
            # Almost no new concepts — loosen abstraction threshold
            abst = self.cognitive_system.abstraction
            if hasattr(abst, 'similarity_threshold'):
                old_thresh = abst.similarity_threshold
                abst.similarity_threshold = max(0.5, abst.similarity_threshold - 0.05)
                actions_taken.append(
                    f"similarity_threshold {old_thresh:.2f}→{abst.similarity_threshold:.2f} "
                    f"(concept growth stagnant)"
                )

        # ── 4. Goal strategy performance ──────────────────────────────────
        current_strat = snapshot.get('goals', {}).get('strategy_performance', {})
        baseline_strat = baseline.get('goals', {}).get('strategy_performance', {})
        for strat, score in current_strat.items():
            baseline_score = baseline_strat.get(strat, 0.5)
            delta = score - baseline_score
            if delta < -0.1:
                # This strategy is performing worse — reduce its weight
                goals_sys.strategy_performance[strat] = max(
                    0.1, goals_sys.strategy_performance[strat] - 0.05
                )
                actions_taken.append(f"strategy '{strat}' weight reduced (delta {delta:+.3f})")
            elif delta > 0.1:
                goals_sys.strategy_performance[strat] = min(
                    0.9, goals_sys.strategy_performance[strat] + 0.05
                )
                actions_taken.append(f"strategy '{strat}' weight increased (delta {delta:+.3f})")

        # ── 5. Update baseline ─────────────────────────────────────────────────
        # Slowly blend the baseline toward the current snapshot so the system
        # tracks gradual drift rather than comparing against a stale origin.
        self._introspection_baseline = snapshot

        if actions_taken:
            logger.info(
                f"Self-reflection [{len(self._introspection_history)} snapshots]: "
                + "; ".join(actions_taken)
            )
        else:
            logger.debug(
                f"Self-reflection: no adjustments needed "
                f"(accuracy={current_accuracy:.3f}, rules={current_rules}, "
                f"concepts={current_concepts})"
            )

    # ──────────────────────────────────────────
    # Self-Evolution
    # ──────────────────────────────────────────

    def _run_self_evolution(self):
        """
        Periodically evolve the learning engine's hyperparameters using the
        SelfEvolvingSystem.  The fitness function evaluates each mutated
        learning-rate candidate against a held-out window of recent
        observations — the candidate that minimises prediction error on that
        window wins and is applied back to the live learning engine.
        """
        self._evolution_counter += 1
        le = self.cognitive_system.learning_engine
        current_lr = getattr(le, 'learning_rate', 0.01)

        # ── Held-out evaluation window ────────────────────────────────────
        # Take the last 30 observations as a held-out set.  For each
        # candidate learning-rate we simulate a forward pass (no weight
        # update) and measure mean-absolute prediction error.
        held_out = list(self.cognitive_system._observation_history)[-30:]
        if len(held_out) < 5:
            logger.debug("Self-evolution skipped: not enough observations yet")
            return

        import copy as _copy
        import ast as _ast
        import numpy as _np

        def _eval_lr(candidate_lr: float) -> float:
            """Return mean-abs-error of a candidate LR on the held-out window.
            Lower is better."""
            # Clone weights so we don't touch the live engine
            w = _copy.deepcopy(le.weights)
            b = float(getattr(le, 'bias', 0.0))
            correct = 0
            for obs in held_out:
                x = le._encode_observation(obs)
                pred = float(_np.dot(w[:len(x)], x[:len(w)]) + b)
                pred = 1.0 / (1.0 + _np.exp(-_np.clip(pred, -10, 10)))
                # Use the normalised value field as a proxy outcome (0–1)
                outcome = obs.get('value_norm', obs.get('value', 0.5))
                if isinstance(outcome, (int, float)):
                    outcome = float(outcome)
                    # Clamp to [0, 1]
                    outcome = max(0.0, min(1.0, outcome))
                    error = outcome - pred
                    grad = error * x[:len(w)]
                    w[:len(x)] += candidate_lr * grad
                    b += candidate_lr * error
                    correct += int((pred > 0.5) == (outcome > 0.5))
            accuracy = correct / max(len(held_out), 1)
            return accuracy  # higher accuracy = better fitness

        baseline_accuracy = _eval_lr(current_lr)

        # Build code snippet encoding the current LR for the genetic engine
        code_snippet = f"learning_rate = {current_lr:.6f}\nfeature_dim = {le.feature_dim}"

        # Test cases carry the held-out baseline so the fitness fn can compare
        test_cases = [{"baseline_accuracy": baseline_accuracy, "held_out_size": len(held_out)}]

        def _fitness_fn(code: str, cases) -> float:
            """Parse the candidate LR from the mutated code and evaluate it."""
            try:
                tree = _ast.parse(code)
                candidate_lr = current_lr
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.Assign):
                        for t in node.targets:
                            if isinstance(t, _ast.Name) and t.id == 'learning_rate':
                                if isinstance(node.value, _ast.Constant):
                                    candidate_lr = float(node.value.value)
                candidate_lr = max(1e-5, min(0.1, candidate_lr))
                return _eval_lr(candidate_lr)
            except Exception:
                return 0.0

        try:
            # ── Use the correct method name: evolve() not evolve_code() ──
            best = self.code_evolver.evolve(
                base_code=code_snippet,
                test_cases=test_cases,
            )
            if best and best.performance_score > baseline_accuracy + 0.005:
                # Parse the winning LR from the evolved code
                try:
                    tree = _ast.parse(best.code)
                    for node in _ast.walk(tree):
                        if isinstance(node, _ast.Assign):
                            for t in node.targets:
                                if isinstance(t, _ast.Name) and t.id == 'learning_rate':
                                    if isinstance(node.value, _ast.Constant):
                                        new_lr = float(node.value.value)
                                        new_lr = max(1e-5, min(0.1, new_lr))
                                        le.learning_rate = new_lr
                                        logger.info(
                                            f"Self-evolution cycle {self._evolution_counter}: "
                                            f"LR {current_lr:.6f} → {new_lr:.6f} "
                                            f"(accuracy {baseline_accuracy:.4f} → {best.performance_score:.4f})"
                                        )
                except Exception:
                    pass
            else:
                logger.debug(
                    f"Self-evolution cycle {self._evolution_counter}: "
                    f"no improvement over baseline {baseline_accuracy:.4f}"
                )
        except Exception as e:
            logger.warning(f"Self-evolution error cycle {self._evolution_counter}: {e}")

    def _build_goal_context(self):
        """Build context for goal generation with real observations and patterns."""
        from goal_formation_system import GoalGenerationContext

        # Pass the last 50 real observations so curiosity/exploration goals can fire
        obs_history = list(self.cognitive_system._observation_history)[-50:]

        # Collect patterns from all active concepts (domain-agnostic)
        patterns = []
        for cid, concept in self.cognitive_system.abstraction.concepts.items():
            patterns.append({
                "pattern_id": cid,
                "domain": getattr(concept, 'domain', 'unknown'),
                "confidence": getattr(concept, 'confidence', 0.5),
                "examples_count": len(getattr(concept, 'examples', [])),
            })

        pred_insights = {}
        try:
            pred_insights = self.prediction_engine.get_insights() or {}
        except Exception:
            pass

        constitutional_state = {}
        try:
            constitutional_state = self.cognitive_system.constitutional_physics.export_state()
        except Exception:
            constitutional_state = {}

        current_state = self.get_metrics()
        current_state['constitutional'] = constitutional_state.get('last_snapshot', {})
        current_state['constitutional_totals'] = {
            'agents': constitutional_state.get('total_agents', 0),
            'attractors': constitutional_state.get('total_attractors', 0),
        }

        performance_metrics = dict(pred_insights)
        performance_metrics['constitutional'] = constitutional_state.get('last_snapshot', {})
        performance_metrics.update({
            'recursive_loss': self._recursive_state.get('coherence_loss', 0.5),
            'world_model_loss': self._recursive_state.get('world_model_loss', 0.5),
            'world_model_prediction_loss': self._recursive_state.get('world_model_prediction_loss', 0.5),
            'world_model_reconstruction_loss': self._recursive_state.get('world_model_reconstruction_loss', 0.5),
            'world_model_memory_loss': self._recursive_state.get('world_model_memory_loss', 1.0),
        })

        return GoalGenerationContext(
            observations=obs_history,
            patterns=patterns,
            capabilities=set(['reasoning', 'abstraction', 'prediction', 'self_evolution', 'constitutional_physics', 'world_model']),
            constraints={
                'must_preserve_constitutional_anchor': True,
                'goal_priority_should_track_coherence': True,
            },
            current_state=current_state,
            performance_metrics=performance_metrics,
        )

    async def save_state(self):
        """Save the entire cognitive state to persistent storage."""
        logger.info("Saving cognitive state...")
        if self.postgres:
            # Save Abstraction Engine state
            concepts_to_save = [c.to_dict() for c in self.cognitive_system.abstraction.concepts.values()]
            await self.postgres.save_concepts(concepts_to_save)
            logger.debug(f"Saved {len(concepts_to_save)} concepts to Postgres.")

            # Save Reasoning Engine state
            rules_to_save = [r.to_dict() for r in self.cognitive_system.reasoning.rules.values()]
            await self.postgres.save_rules(rules_to_save)
            await self.postgres.save_facts(self.cognitive_system.reasoning.facts)
            logger.debug(f"Saved {len(rules_to_save)} rules and {len(self.cognitive_system.reasoning.facts)} facts to Postgres.")

            # Save Cross-Domain Engine state (mappings)
            mappings_to_save = [m.to_dict() for m in self.cognitive_system.cross_domain.mappings.values()]
            await self.postgres.save_cross_domain_mappings(mappings_to_save)
            logger.debug(f"Saved {len(mappings_to_save)} cross-domain mappings to Postgres.")

            # Save Goal Formation System state
            goals_to_save = [g.to_dict() for g in self.cognitive_system.goals.goals.values()]
            await self.postgres.save_goals(goals_to_save)
            logger.debug(f"Saved {len(goals_to_save)} goals to Postgres.")

            # Save Prediction Validation Engine state
            prediction_engine_state = {
                "symbols": {s_id: s.__dict__ for s_id, s in self.prediction_engine.symbols.items()},
                "all_predictions": [p.__dict__ for p in self.prediction_engine.all_predictions],
                "prediction_counter": self.prediction_engine.prediction_counter,
                "rule_performance": {r_id: r.__dict__ for r_id, r in self.prediction_engine.rule_performance.items()},
                "accuracy_history": list(self.prediction_engine.accuracy_history),
                "phi_history": list(self.prediction_engine.phi_history),
                "sigma_history": list(self.prediction_engine.sigma_history),
                "total_predictions": self.prediction_engine.total_predictions,
                "total_correct": self.prediction_engine.total_correct,
                "total_validated": self.prediction_engine.total_validated,
            }
            await self.postgres.save_prediction_engine_state(prediction_engine_state)
            logger.debug("Saved Prediction Validation Engine state to Postgres.")

            # Save other RAM-only states from CognitiveIntelligentSystem
            # _observation_history
            observation_history_to_save = list(self.cognitive_system._observation_history)
            await self.postgres.save_observation_history(observation_history_to_save)
            logger.debug(f"Saved {len(observation_history_to_save)} observations to Postgres.")

            # _price_history
            price_history_to_save = {k: list(v) for k, v in self.cognitive_system._price_history.items()}
            await self.postgres.save_price_history(price_history_to_save)
            logger.debug(f"Saved price history for {len(price_history_to_save)} symbols to Postgres.")

            # _meta_domains
            meta_domains_to_save = {k: list(v) for k, v in self.cognitive_system._meta_domains.items()}
            await self.postgres.save_meta_domains(meta_domains_to_save)
            logger.debug(f"Saved {len(meta_domains_to_save)} meta domains to Postgres.")

            # learning_engine
            learning_engine_state = {
                "weights": self.cognitive_system.learning_engine.weights.tolist(),
                "bias": self.cognitive_system.learning_engine.bias,
                "feature_names": self.cognitive_system.learning_engine.feature_names,
                "feature_index": self.cognitive_system.learning_engine.feature_index,
                "long_term_patterns": {p_id: p.to_dict() for p_id, p in self.cognitive_system.learning_engine.long_term_patterns.items()},
                "metrics": self.cognitive_system.learning_engine.metrics.to_dict(),
                "prediction_history": list(self.cognitive_system.learning_engine.prediction_history),
                "feature_means": dict(self.cognitive_system.learning_engine.feature_means),
                "feature_vars": dict(self.cognitive_system.learning_engine.feature_vars),
                "sample_count": self.cognitive_system.learning_engine.sample_count,
                "drift_events": list(self.cognitive_system.learning_engine.drift_events),
                "_current_lr_history": list(self.cognitive_system.learning_engine._current_lr_history),
            }
            await self.postgres.save_learning_engine_state(learning_engine_state)
            logger.debug("Saved Learning Engine state to Postgres.")

            # Caches from CognitiveIntelligentSystem
            caches_to_save = {
                "_recent_analogies": list(self.cognitive_system._recent_analogies),
                "_recent_explanations": list(self.cognitive_system._recent_explanations),
                "_recent_plans": list(self.cognitive_system._recent_plans),
                "_causal_discovery_log": list(self.cognitive_system._causal_discovery_log),
                "_transfer_suggestions_cache": self.cognitive_system._transfer_suggestions_cache,
                "_pursuit_log": list(self.cognitive_system._pursuit_log),
                "resonant_memory_state": self.cognitive_system.resonant_memory.export_state(),
                "world_model_state": self.world_model.save_state(),
            }
            await self.postgres.save_caches(caches_to_save)
            logger.debug("Saved CognitiveIntelligentSystem caches to Postgres.")

        if self.milvus:
            logger.debug("Milvus is assumed to be kept in sync by the Abstraction Engine.")

        logger.info("Cognitive state saved.")

    async def load_state(self):
        """Load the entire cognitive state from persistent storage."""
        logger.info("Loading cognitive state...")
        if self.postgres:
            # Load Abstraction Engine state
            loaded_concepts = await self.postgres.load_concepts()
            for c_data in loaded_concepts:
                c_data["created_at"] = datetime.fromisoformat(c_data["created_at"]) if c_data.get("created_at") else datetime.now()
                c_data["examples"] = []
                c_data["parent_concepts"] = set(c_data.get("parents", []))
                c_data["child_concepts"] = set(c_data.get("children", []))
                concept = Concept(concept_id=c_data["id"], name=c_data["name"], level=c_data["level"], attributes=c_data["signature"], examples=[], confidence=c_data["confidence"], created_at=c_data["created_at"], parent_concepts=c_data["parent_concepts"], child_concepts=c_data["child_concepts"])
                self.cognitive_system.abstraction.concepts[concept.concept_id] = concept
            logger.debug(f"Loaded {len(loaded_concepts)} concepts from Postgres.")

            # Load Reasoning Engine state
            loaded_rules = await self.postgres.load_rules()
            for r_data in loaded_rules:
                r_data["created_at"] = datetime.fromisoformat(r_data["created_at"]) if r_data.get("created_at") else datetime.now()
                rule = Rule(rule_id=r_data["id"], rule_type=RuleType(r_data["type"]), antecedents=r_data["antecedent"], consequent=r_data["consequent"], confidence=r_data["confidence"], support_count=r_data["support"], created_at=r_data["created_at"])
                self.cognitive_system.reasoning.rules[rule.rule_id] = rule
            self.cognitive_system.reasoning.facts = await self.postgres.load_facts()
            logger.debug(f"Loaded {len(loaded_rules)} rules and {len(self.cognitive_system.reasoning.facts)} facts from Postgres.")

            # Load Cross-Domain Engine state (mappings)
            loaded_mappings = await self.postgres.load_cross_domain_mappings()
            for m_data in loaded_mappings:
                mapping = DomainMapping(mapping_id=m_data["mapping_id"], source_domain=m_data["source_domain"], target_domain=m_data["target_domain"], concept_mappings=m_data["concept_mappings"], confidence=m_data["confidence"], bidirectional=m_data["bidirectional"])
                self.cognitive_system.cross_domain.mappings[mapping.mapping_id] = mapping
            logger.debug(f"Loaded {len(loaded_mappings)} cross-domain mappings from Postgres.")

            # Load Goal Formation System state
            loaded_goals = await self.postgres.load_goals()
            for g_data in loaded_goals:
                g_data["created_at"] = datetime.fromisoformat(g_data["created_at"]) if g_data.get("created_at") else datetime.now()
                g_data["deadline"] = datetime.fromisoformat(g_data["deadline"]) if g_data.get("deadline") else None
                g_data["last_attempt"] = datetime.fromisoformat(g_data["last_attempt"]) if g_data.get("last_attempt") else None
                g_data["achieved_at"] = datetime.fromisoformat(g_data["achieved_at"]) if g_data.get("achieved_at") else None
                goal = Goal(goal_id=g_data["id"], goal_type=GoalType(g_data["type"]), description=g_data["description"], success_criteria=g_data["success_criteria"], priority=g_data["priority"], status=GoalStatus(g_data["status"]), progress=g_data["progress"], value_estimate=g_data["value_estimate"], created_at=g_data["created_at"], deadline=g_data["deadline"], parent_goal=g_data["parent_goal"], sub_goals=g_data["sub_goals"], attempts=g_data["attempts"], last_attempt=g_data["last_attempt"], achieved_at=g_data["achieved_at"])
                self.cognitive_system.goals.goals[goal.goal_id] = goal
            logger.debug(f"Loaded {len(loaded_goals)} goals from Postgres.")

            # Load Prediction Validation Engine state
            prediction_engine_state = await self.postgres.load_prediction_engine_state()
            if prediction_engine_state:
                for s_id, s_data in prediction_engine_state["symbols"].items():
                    history = SymbolHistory(symbol=s_data["symbol"], domain=s_data["domain"])
                    history.prices = deque(s_data["prices"], maxlen=200)
                    history.volumes = deque(s_data["volumes"], maxlen=200)
                    history.timestamps = deque(s_data["timestamps"], maxlen=200)
                    if s_data["pending_prediction"]:
                        pred_data = s_data["pending_prediction"]
                        history.pending_prediction = Prediction(prediction_id=pred_data["prediction_id"], symbol=pred_data["symbol"], domain=pred_data["domain"], direction=Direction(pred_data["direction"]), confidence=pred_data["confidence"], basis=pred_data["basis"], predicted_at=pred_data["predicted_at"], predicted_price=pred_data["predicted_price"], horizon=pred_data["horizon"], ticks_remaining=pred_data["ticks_remaining"], dead_zone_pct=pred_data["dead_zone_pct"], target_price=pred_data["target_price"], actual_price=pred_data["actual_price"], actual_direction=Direction(pred_data["actual_direction"]) if pred_data["actual_direction"] else None, validated=pred_data["validated"], max_post_signal_move_pct=pred_data["max_post_signal_move_pct"], correct=pred_data["correct"], validation_time=pred_data["validation_time"])
                    history.total_predictions = s_data["total_predictions"]
                    history.correct_predictions = s_data["correct_predictions"]
                    history.recent_accuracy = deque(s_data["recent_accuracy"], maxlen=50)
                    self.prediction_engine.symbols[s_id] = history
                self.prediction_engine.all_predictions = deque([Prediction(prediction_id=p["prediction_id"], symbol=p["symbol"], domain=p["domain"], direction=Direction(p["direction"]), confidence=p["confidence"], basis=p["basis"], predicted_at=p["predicted_at"], predicted_price=p["predicted_price"], horizon=p["horizon"], ticks_remaining=p["ticks_remaining"], dead_zone_pct=p["dead_zone_pct"], target_price=p["target_price"], actual_price=p["actual_price"], actual_direction=Direction(p["actual_direction"]) if p["actual_direction"] else None, validated=p["validated"], max_post_signal_move_pct=p["max_post_signal_move_pct"], correct=p["correct"], validation_time=p["validation_time"]) for p in prediction_engine_state["all_predictions"]], maxlen=5000)
                self.prediction_engine.prediction_counter = prediction_engine_state["prediction_counter"]
                for r_id, r_data in prediction_engine_state["rule_performance"].items():
                    rule_perf = RulePerformance(rule_id=r_data["rule_id"])
                    rule_perf.predictions_made = r_data["predictions_made"]
                    rule_perf.correct_predictions = r_data["correct_predictions"]
                    rule_perf.recent_accuracy = deque(r_data["recent_accuracy"], maxlen=30)
                    self.prediction_engine.rule_performance[r_id] = rule_perf
                self.prediction_engine.accuracy_history = deque(prediction_engine_state["accuracy_history"], maxlen=500)
                self.prediction_engine.phi_history = deque(prediction_engine_state["phi_history"], maxlen=500)
                self.prediction_engine.sigma_history = deque(prediction_engine_state["sigma_history"], maxlen=500)
                self.prediction_engine.total_predictions = prediction_engine_state["total_predictions"]
                self.prediction_engine.total_correct = prediction_engine_state["total_correct"]
                self.prediction_engine.total_validated = prediction_engine_state["total_validated"]
            logger.debug("Loaded Prediction Validation Engine state from Postgres.")

            # Load other RAM-only states from CognitiveIntelligentSystem
            loaded_observation_history = await self.postgres.load_observation_history()
            self.cognitive_system._observation_history = deque(loaded_observation_history, maxlen=self.cognitive_system._max_observation_history)
            logger.debug(f"Loaded {len(loaded_observation_history)} observations into history.")

            loaded_price_history = await self.postgres.load_price_history()
            for symbol, prices in loaded_price_history.items():
                self.cognitive_system._price_history[symbol] = deque(prices, maxlen=self.cognitive_system._max_price_history)
            logger.debug(f"Loaded price history for {len(loaded_price_history)} symbols.")

            loaded_meta_domains = await self.postgres.load_meta_domains()
            for meta_domain, domains in loaded_meta_domains.items():
                self.cognitive_system._meta_domains[meta_domain] = set(domains)
            logger.debug(f"Loaded {len(loaded_meta_domains)} meta domains.")

            learning_engine_state = await self.postgres.load_learning_engine_state()
            if learning_engine_state:
                self.cognitive_system.learning_engine.weights = np.array(learning_engine_state["weights"])
                self.cognitive_system.learning_engine.bias = learning_engine_state["bias"]
                self.cognitive_system.learning_engine.feature_names = learning_engine_state["feature_names"]
                self.cognitive_system.learning_engine.feature_index = learning_engine_state["feature_index"]
                self.cognitive_system.learning_engine.long_term_patterns = {p_id: Pattern(pattern_id=p["pattern_id"], centroid=np.array(p["centroid"]), examples=[], confidence=p["confidence"], created_at=datetime.fromisoformat(p["created_at"]), hit_count=p["hit_count"]) for p_id, p in learning_engine_state["long_term_patterns"].items()}
                self.cognitive_system.learning_engine.metrics = LearningMetrics(accuracy=learning_engine_state["metrics"]["accuracy"], samples_processed=learning_engine_state["metrics"]["samples_processed"], patterns_discovered=learning_engine_state["metrics"]["patterns_discovered"], adaptations=learning_engine_state["metrics"]["adaptations"], last_update=datetime.fromisoformat(learning_engine_state["metrics"]["last_update"]))
                self.cognitive_system.learning_engine.prediction_history = deque(learning_engine_state["prediction_history"], maxlen=100)
                self.cognitive_system.learning_engine.feature_means = defaultdict(float, learning_engine_state["feature_means"])
                self.cognitive_system.learning_engine.feature_vars = defaultdict(lambda: 1.0, learning_engine_state["feature_vars"])
                self.cognitive_system.learning_engine.sample_count = learning_engine_state["sample_count"]
                self.cognitive_system.learning_engine.drift_events = deque(learning_engine_state["drift_events"], maxlen=200)
                self.cognitive_system.learning_engine._current_lr_history = deque(learning_engine_state["_current_lr_history"], maxlen=200)
            logger.debug("Loaded Learning Engine state from Postgres.")

            loaded_caches = await self.postgres.load_caches()
            self.cognitive_system._recent_analogies = deque(loaded_caches.get("_recent_analogies", []), maxlen=100)
            self.cognitive_system._recent_explanations = deque(loaded_caches.get("_recent_explanations", []), maxlen=100)
            self.cognitive_system._recent_plans = deque(loaded_caches.get("_recent_plans", []), maxlen=100)
            self.cognitive_system._causal_discovery_log = deque(loaded_caches.get("_causal_discovery_log", []), maxlen=100)
            self.cognitive_system._transfer_suggestions_cache = loaded_caches.get("_transfer_suggestions_cache", {})
            self.cognitive_system._pursuit_log = deque(loaded_caches.get("_pursuit_log", []), maxlen=100)
            self.cognitive_system.resonant_memory.load_state(loaded_caches.get("resonant_memory_state", {}))
            self.world_model.load_state(loaded_caches.get("world_model_state", {}))
            logger.debug("Loaded CognitiveIntelligentSystem caches and world model from Postgres.")

        if self.milvus:
            logger.debug("Milvus is assumed to be kept in sync by the Abstraction Engine.")

        logger.info("Cognitive state loaded.")
