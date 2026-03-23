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
from config.config import Config

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
        network=None,
        pubsub=None
    ):
        self.node_id = node_id
        self.postgres = postgres
        self.milvus = milvus
        self.network = network
        self.pubsub = pubsub

        # Core Engines
        self.cognitive_system = CognitiveIntelligentSystem(system_id=node_id)
        self.prediction_engine = PredictionValidationEngine()

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
            # Milvus stores concept vectors. The concepts themselves are saved in Postgres.
            # We need to ensure that the Milvus store is populated with the vectors from the concepts.
            # This might require iterating through the concepts and inserting them into Milvus if they are not already there.
            # For now, I will assume that Milvus is kept in sync by the abstraction engine.
            logger.debug("Milvus is assumed to be kept in sync by the Abstraction Engine.")

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
            logger.debug("Loaded CognitiveIntelligentSystem caches from Postgres.")

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
            # Milvus stores concept vectors. The concepts themselves are loaded from Postgres.
            # We need to ensure that the Milvus store is populated with the vectors from the concepts.
            # This might require iterating through the concepts and inserting them into Milvus if they are not already there.
            # For now, I will assume that Milvus is kept in sync by the abstraction engine.
            logger.debug("Milvus is assumed to be kept in sync by the Abstraction Engine.")

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
                            "status": p.get('state', 'unknown'),
                            "assets_tracked": p.get('failures', 0),
                            "latency_ms": 0,
                        }
                except Exception as e:
                    logger.debug(f"Provider status error in cache: {e}")

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
            }
            with self._state_cache_lock:
                self._state_cache = state
                self._state_cache_ready = True
        except Exception as e:
            logger.error(f"Error updating state cache: {e}", exc_info=True)

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
            "toggles": dict(self._toggles),
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
                                # Process through cognitive engines
                                res = self.cognitive_system.process_observation(obs, domain)
                                if not isinstance(res, dict):
                                    logger.warning(f"process_observation returned unexpected type: {type(res)}. Expected dict.")
                                    res = {}
                                
                                                # Feed into prediction engine (domain-agnostic)
                                self.prediction_engine.record_observation(obs, domain)
                            except Exception as e:
                                self._errors += 1
                                logger.error(f"Error processing observation: {e}")

                    # 1b. Update state cache every 5s OR if queue is large
                    # Only update if we've actually processed something new
                    now = time.time()
                    if (now - last_cache_update > 5.0) or (queue_size > 100 and iteration % 20 == 0):
                        self._update_state_cache()
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
                                except Exception as e:
                                    logger.error(f"Rule learning error: {e}")

                            # Converge concepts (merge/prune)
                            self._run_concept_convergence()

                    # 3. Deep Introspection (every ~30s)
                    if iteration % 300 == 0:
                        with self._lock:
                            try:
                                # Trigger goal formation with real observations
                                ctx = self._build_goal_context()
                                new_goals = self.cognitive_system.goals.generate_goals(ctx)
                                if new_goals:
                                    logger.info(f"Goal formation: {len(new_goals)} new goals generated")
                            except Exception as e:
                                logger.error(f"Error in introspection: {e}")

                    # 4. Self-evolution (every ~5 min)
                    if iteration % 3000 == 0 and self._toggles.get('self_evolution', True):
                        try:
                            self._run_self_evolution()
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

    # ──────────────────────────────────────────
    # Self-Evolution
    # ──────────────────────────────────────────

    def _run_self_evolution(self):
        """
        Periodically evolve the learning engine's hyperparameters using the
        SelfEvolvingSystem.  The fitness function is the current prediction
        accuracy — the code evolver mutates learning-rate and feature-weight
        parameters and keeps the best-performing variant.
        """
        self._evolution_counter += 1
        metrics = self.get_metrics()
        accuracy = metrics.get('prediction_accuracy', 0.5)
        phi = metrics.get('global_coherence_phi', 0.5)

        # Build a minimal code snippet representing the current LR
        current_lr = getattr(self.cognitive_system.learning_engine, '_current_lr', 0.01)
        code_snippet = f"learning_rate = {current_lr:.6f}\nfeature_dim = {self.cognitive_system.learning_engine.feature_dim}"

        # Build test cases from recent prediction outcomes
        test_cases = [
            {"accuracy": accuracy, "phi": phi, "expected_improvement": 0.01}
        ]

        def _fitness_fn(code: str, cases) -> float:
            """Evaluate a code variant by simulated accuracy improvement"""
            return accuracy * 0.7 + phi * 0.3

        try:
            best = self.code_evolver.evolve_code(
                base_code=code_snippet,
                test_cases=test_cases,
                fitness_fn=_fitness_fn,
            )
            if best and best.performance_score > accuracy:
                logger.info(
                    f"Self-evolution cycle {self._evolution_counter}: "
                    f"found variant gen={best.generation} "
                    f"score={best.performance_score:.4f} (was {accuracy:.4f})"
                )
                # Apply the best learning rate back to the learning engine
                try:
                    import ast as _ast
                    tree = _ast.parse(best.code)
                    for node in _ast.walk(tree):
                        if isinstance(node, _ast.Assign):
                            for t in node.targets:
                                if isinstance(t, _ast.Name) and t.id == 'learning_rate':
                                    if isinstance(node.value, _ast.Constant):
                                        new_lr = float(node.value.value)
                                        new_lr = max(1e-5, min(0.1, new_lr))
                                        if hasattr(self.cognitive_system.learning_engine, '_current_lr'):
                                            self.cognitive_system.learning_engine._current_lr = new_lr
                                            logger.info(f"Applied evolved learning rate: {new_lr:.6f}")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Self-evolution skipped this cycle: {e}")

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

        return GoalGenerationContext(
            observations=obs_history,
            patterns=patterns,
            capabilities=set(['reasoning', 'abstraction', 'prediction', 'self_evolution']),
            constraints={},
            current_state=self.get_metrics(),
            performance_metrics=pred_insights,
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
            logger.debug("Loaded CognitiveIntelligentSystem caches from Postgres.")

        if self.milvus:
            logger.debug("Milvus is assumed to be kept in sync by the Abstraction Engine.")

        logger.info("Cognitive state loaded.")
