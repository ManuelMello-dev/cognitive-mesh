"""
Distributed Cognitive Core — The Orchestrator
==============================================
Manages multiple cognitive engines, prediction engine, and infrastructure.
Enables cross-symbol coherence, rule feedback, and concept convergence.
"""

import logging
import time
import threading
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict

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

        # Buffer for async ingestion
        self._pending_observations = deque(maxlen=500)

        logger.info(f"Distributed Cognitive Core '{node_id}' initialized")

    # ──────────────────────────────────────────
    # Cognitive Loop
    # ──────────────────────────────────────────

    def start_cognitive_loop(self):
        """Start the background cognitive loop"""
        if self._running:
            return

        self._running = True

        def _loop():
            iteration = 0
            while self._running:
                try:
                    iteration += 1
                    
                    # 1. Process pending observations
                    batch = []
                    while self._pending_observations and len(batch) < 10:
                        batch.append(self._pending_observations.popleft())

                    for obs, domain in batch:
                        try:
                            # Process through cognitive engines
                            res = self.cognitive_system.process_observation(obs, domain)
                            
                            # Feed into prediction engine
                            symbol = obs.get('symbol')
                            price = obs.get('price')
                            if symbol and price:
                                self.prediction_engine.observe(symbol, price, obs)
                        except Exception as e:
                            self._errors += 1
                            logger.error(f"Error processing observation: {e}")

                    # 2. Perform periodic maintenance (every ~10s)
                    if iteration % 100 == 0:
                        with self._lock:
                            # Feed prediction accuracy back to rules
                            self._feed_rule_confidence_back()
                            
                            # Converge concepts (merge/prune)
                            self._run_concept_convergence()

                    # 3. Deep Introspection (every ~30s)
                    if iteration % 300 == 0:
                        with self._lock:
                            try:
                                # Trigger goal formation
                                ctx = self._build_goal_context()
                                self.cognitive_system.goals.generate_goals(ctx)
                            except Exception as e:
                                logger.error(f"Error in introspection: {e}")

                    time.sleep(0.1)

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

        symbol_concepts: Dict[str, List[str]] = defaultdict(list)
        for cid, concept in concepts.items():
            examples = getattr(concept, 'examples', [])
            for ex in examples[-3:]:
                if isinstance(ex, dict) and 'symbol' in ex:
                    symbol_concepts[ex['symbol']].append(cid)
                    break

        merged = 0
        for symbol, cids in symbol_concepts.items():
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
            meta = "crypto" if domain.startswith("crypto:") else "stock" if domain.startswith("stock:") else None
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
            abstraction_insights = self.cognitive_system.abstraction.get_insights()
            reasoning_insights = self.cognitive_system.reasoning.get_insights()
            cross_domain_insights = self.cognitive_system.cross_domain.get_insights()
            goal_insights = self.cognitive_system.goals.get_insights()
            learning_insights = self.cognitive_system.learning_engine.get_insights()
            prediction_insights = self.prediction_engine.get_insights()

        phi = prediction_insights.get('phi', 0.5)
        sigma = prediction_insights.get('sigma', 0.5)

        return {
            "global_coherence_phi": round(phi, 4),
            "noise_level_sigma": round(sigma, 4),
            "prediction_accuracy": prediction_insights.get('global_accuracy', 0),
            "predictions_validated": prediction_insights.get('total_validated', 0),
            "predictions_correct": prediction_insights.get('total_correct', 0),
            "symbols_tracked": prediction_insights.get('symbols_tracked', 0),
            "attention_density": round(abstraction_insights.get('total_concepts', 0) / 1000.0, 4),
            "concepts_formed": cognitive_metrics.get('concepts_formed', 0),
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

    def get_concepts_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            concepts = {}
            concept_domain_map = {}
            for domain_id, domain in self.cognitive_system.cross_domain.domains.items():
                for concept_id in getattr(domain, 'concepts', []):
                    concept_domain_map[concept_id] = domain_id

            for cid, concept in self.cognitive_system.abstraction.concepts.items():
                examples = getattr(concept, 'examples', [])
                recent_examples = [ex for ex in examples[-5:]]
                
                symbol = None
                for ex in examples[-3:]:
                    if isinstance(ex, dict) and 'symbol' in ex:
                        symbol = ex['symbol']
                        break

                pred_data = {}
                if symbol and symbol in self.prediction_engine.symbols:
                    sym_hist = self.prediction_engine.symbols[symbol]
                    pred_data = {
                        "prediction_accuracy": round(sym_hist.accuracy, 4),
                        "total_predictions": sym_hist.total_predictions,
                        "trend": sym_hist.price_trend.value if sym_hist.price_trend else "unknown",
                        "momentum": round(sym_hist.momentum, 4),
                        "volatility": round(sym_hist.volatility, 6),
                    }

                concepts[cid] = {
                    "id": cid,
                    "name": getattr(concept, 'name', cid),
                    "domain": concept_domain_map.get(cid, 'unknown'),
                    "symbol": symbol,
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
            for rid, rule in self.cognitive_system.reasoning.rules.items():
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
            for gid, goal in self.cognitive_system.goals.goals.items():
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

    def get_transfer_suggestions(self) -> list:
        with self._lock:
            return self.cognitive_system.get_transfer_suggestions_snapshot()

    def _build_goal_context(self):
        """Build context for goal generation"""
        return {
            'observations': [],
            'patterns': self.cognitive_system.abstraction.patterns,
            'capabilities': set(['reasoning', 'abstraction', 'prediction']),
            'performance_metrics': self.prediction_engine.get_insights()
        }
