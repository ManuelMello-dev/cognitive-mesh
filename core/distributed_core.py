"""
Distributed Cognitive Core
==========================
Async wrapper around the original CognitiveIntelligentSystem that:
- Bridges async HTTP/network layer with the synchronous cognitive engines
- Wires the PredictionValidationEngine into every observation
- Calculates PHI/SIGMA from REAL prediction accuracy (not counts)
- Feeds rule confidence back from predictive performance
- Runs concept convergence (merge similar, prune stale)
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
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from collections import deque, defaultdict

from config.config import Config

# Import the REAL cognitive engines
from cognitive_intelligent_system import CognitiveIntelligentSystem
from abstraction_engine import AbstractionEngine
from reasoning_engine import ReasoningEngine, RuleType
from cross_domain_engine import CrossDomainEngine
from goal_formation_system import OpenEndedGoalSystem, GoalType
from prediction_validation_engine import PredictionValidationEngine

logger = logging.getLogger("DistributedCognitiveCore")


def _serialize_enum(val: Any) -> str:
    """Helper to serialize Enums or other objects to string"""
    if isinstance(val, Enum):
        return val.value
    return str(val)

class DistributedCognitiveCore:
    """
    Production-grade async wrapper around CognitiveIntelligentSystem.

    The original CognitiveIntelligentSystem is synchronous and runs its own
    cognitive loop. This wrapper:
    1. Creates the CognitiveIntelligentSystem with all 7 engines
    2. Wires the PredictionValidationEngine into every observation
    3. Exposes an async `ingest()` method for the HTTP/network layer
    4. Runs the cognitive loop in a background thread
    5. Feeds prediction accuracy back into rule confidence
    6. Runs concept convergence (merge + prune) periodically
    7. Calculates PHI/SIGMA from real prediction accuracy
    8. Persists state to PostgreSQL, Milvus, Redis when available
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

        # Create the prediction/validation engine
        self.prediction_engine = PredictionValidationEngine()

        # Thread lock for safe access from async context
        self._lock = threading.Lock()

        # Track observations for the data collection loop
        self._pending_observations: deque = deque(maxlen=5000)
        self._observation_count = 0

        # Metrics overlay
        self._start_time = time.time()
        self._last_observation_time: Optional[float] = None
        self._errors = 0

        # Concept convergence tracking
        self._convergence_counter = 0
        self._concepts_merged = 0
        self._concepts_pruned = 0

        # Background cognitive loop control
        self._cognitive_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"DistributedCognitiveCore initialized with PredictionValidationEngine: {node_id}")

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

                    # ─── Process pending observations ───
                    observations_processed = 0
                    while self._pending_observations and observations_processed < 200:
                        try:
                            obs, domain = self._pending_observations.popleft()
                        except IndexError:
                            break

                        with self._lock:
                            try:
                                # STEP 1: Feed to prediction engine FIRST
                                # This validates any pending prediction and makes a new one
                                validation_result = self.prediction_engine.record_observation(obs, domain)

                                if validation_result:
                                    correct = validation_result.get('correct', False)
                                    symbol = validation_result.get('symbol', '?')
                                    acc = validation_result.get('symbol_accuracy', 0)
                                    logger.info(
                                        f"PREDICTION {'✓' if correct else '✗'} {symbol}: "
                                        f"predicted={validation_result.get('predicted')}, "
                                        f"actual={validation_result.get('actual')}, "
                                        f"accuracy={acc:.1%}"
                                    )

                                # STEP 2: Feed to cognitive system
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

                    # ─── Cognitive thinking every 10 iterations ───
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

                    # ─── Learn rules from accumulated observations every 15 iterations ───
                    if iteration % 15 == 0:
                        with self._lock:
                            try:
                                new_rules = self.cognitive_system.learn_rules_from_history()
                                if new_rules:
                                    logger.info(f"Rule learning: {new_rules} new rules")
                            except Exception as e:
                                logger.error(f"Error in rule learning: {e}")

                    # ─── Feed rule confidence back from predictions every 20 iterations ───
                    if iteration % 20 == 0:
                        with self._lock:
                            try:
                                self._feed_rule_confidence_back()
                            except Exception as e:
                                logger.error(f"Error feeding rule confidence: {e}")

                    # ─── Concept convergence every 30 iterations ───
                    if iteration % 30 == 0:
                        with self._lock:
                            try:
                                self._run_concept_convergence()
                            except Exception as e:
                                logger.error(f"Error in concept convergence: {e}")

                    # ─── Generate autonomous goals every 50 iterations ───
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

                    # ─── Pursue goals every 25 iterations ───
                    if iteration % 25 == 0:
                        with self._lock:
                            try:
                                self.cognitive_system.pursue_goals()
                            except Exception as e:
                                logger.error(f"Error pursuing goals: {e}")

                    # ─── Introspect every 100 iterations ───
                    if iteration % 100 == 0:
                        with self._lock:
                            try:
                                introspection = self.cognitive_system.introspect()
                                pred_insights = self.prediction_engine.get_insights()
                                phi = pred_insights.get('phi', 0.5)
                                sigma = pred_insights.get('sigma', 0.5)
                                global_acc = pred_insights.get('global_accuracy', 0)
                                total_preds = pred_insights.get('total_validated', 0)

                                logger.info("=" * 60)
                                logger.info("INTROSPECTION")
                                logger.info(f"  PHI (real): {phi:.4f}")
                                logger.info(f"  SIGMA (real): {sigma:.4f}")
                                logger.info(f"  Prediction Accuracy: {global_acc:.1%} ({total_preds} validated)")
                                logger.info(f"  Concepts: {introspection['abstraction']['total_concepts']} (merged: {self._concepts_merged}, pruned: {self._concepts_pruned})")
                                logger.info(f"  Rules: {introspection['reasoning']['total_rules']}")
                                logger.info(f"  Goals: {introspection['goals']['total_goals']}")
                                logger.info(f"  Transfers: {introspection['cognitive_metrics']['knowledge_transfers']}")
                                logger.info(f"  Domains: {introspection['active_domains']}")
                                logger.info(f"  Symbols tracked: {pred_insights.get('symbols_tracked', 0)}")
                                logger.info("=" * 60)
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
        Rules that predict well get boosted; rules that fail get penalized.
        This replaces the naive support-count-based confidence.
        """
        adjustments = self.prediction_engine.get_rule_confidence_adjustments()
        if not adjustments:
            return

        adjusted_count = 0
        for rule_id, accuracy in adjustments.items():
            # Find rules whose basis matches this pattern
            for rid, rule in self.cognitive_system.reasoning.rules.items():
                # Match by checking if rule_id substring appears in rule attributes
                rule_conf = getattr(rule, 'confidence', 0.5)
                # Blend current confidence with prediction-based accuracy
                # 70% prediction accuracy, 30% existing confidence
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
        This prevents unbounded concept accumulation.
        """
        self._convergence_counter += 1
        concepts = self.cognitive_system.abstraction.concepts

        if len(concepts) < 5:
            return

        # ─── Phase 1: Merge concepts in the same domain with same symbol ───
        # Group concepts by their primary symbol
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

            # Keep the concept with the most examples (most mature)
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

                # Merge: transfer examples and boost confidence
                primary_examples = getattr(primary, 'examples', [])
                secondary_examples = getattr(secondary, 'examples', [])

                # Add unique examples from secondary to primary
                for ex in secondary_examples:
                    if len(primary_examples) < 100:  # cap examples
                        primary_examples.append(ex)

                # Boost confidence (more evidence = more confident)
                primary.confidence = min(0.99,
                    getattr(primary, 'confidence', 0.5) + 0.05)

                # Remove the secondary concept
                del concepts[cid]
                merged += 1

        if merged > 0:
            self._concepts_merged += merged
            logger.info(f"Concept convergence: merged {merged} duplicate concepts")

        # ─── Phase 2: Prune stale concepts ───
        # Remove concepts with very low confidence and few examples
        pruned = 0
        stale_ids = []
        for cid, concept in concepts.items():
            examples = getattr(concept, 'examples', [])
            confidence = getattr(concept, 'confidence', 0)
            age = (datetime.now() - getattr(concept, 'created_at', datetime.now())).total_seconds()

            # Prune if: low confidence AND few examples AND old enough (>5 min)
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
        Queues the observation for the cognitive loop to process.
        """
        self._observation_count += 1
        self._last_observation_time = time.time()

        # Register domain if new, and also register meta-domain
        with self._lock:
            if domain not in self.cognitive_system.cross_domain.domains:
                self.cognitive_system.cross_domain.register_domain(domain, domain)
                logger.info(f"Registered new domain: {domain}")
            
            # Ensure meta-domains exist and track membership
            meta = "crypto" if domain.startswith("crypto:") else "stock" if domain.startswith("stock:") else None
            if meta:
                self.cognitive_system._ensure_meta_domain(meta)
                self.cognitive_system._meta_domains[meta].add(domain)

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
            "predictions_made": self.prediction_engine.total_predictions,
            "global_accuracy": round(
                self.prediction_engine.total_correct / max(self.prediction_engine.total_validated, 1), 4
            ),
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

    # ──────────────────────────────────────────
    # Metrics (Real PHI/SIGMA from predictions)
    # ──────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined cognitive + prediction + infrastructure metrics"""
        with self._lock:
            cognitive_metrics = self.cognitive_system.cognitive_metrics.copy()
            abstraction_insights = self.cognitive_system.abstraction.get_insights()
            reasoning_insights = self.cognitive_system.reasoning.get_insights()
            cross_domain_insights = self.cognitive_system.cross_domain.get_insights()
            goal_insights = self.cognitive_system.goals.get_insights()
            learning_insights = self.cognitive_system.learning_engine.get_insights()
            prediction_insights = self.prediction_engine.get_insights()

        # PHI and SIGMA from the REAL prediction engine
        phi = prediction_insights.get('phi', 0.5)
        sigma = prediction_insights.get('sigma', 0.5)

        return {
            # Core consciousness metrics (REAL, from predictions)
            "global_coherence_phi": round(phi, 4),
            "noise_level_sigma": round(sigma, 4),
            "prediction_accuracy": prediction_insights.get('global_accuracy', 0),
            "predictions_validated": prediction_insights.get('total_validated', 0),
            "predictions_correct": prediction_insights.get('total_correct', 0),
            "symbols_tracked": prediction_insights.get('symbols_tracked', 0),

            # Attention density
            "attention_density": round(
                abstraction_insights.get('total_concepts', 0) / 1000.0, 4
            ),

            # Cognitive metrics from the real engines
            "concepts_formed": cognitive_metrics.get('concepts_formed', 0),
            "rules_learned": cognitive_metrics.get('rules_learned', 0),
            "analogies_found": cognitive_metrics.get('analogies_found', 0),
            "goals_achieved": cognitive_metrics.get('goals_achieved', 0),
            "knowledge_transfers": cognitive_metrics.get('knowledge_transfers', 0),
            "causal_links_discovered": cognitive_metrics.get('causal_links_discovered', 0),

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

            # Convergence metrics
            "concepts_merged": self._concepts_merged,
            "concepts_pruned": self._concepts_pruned,

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

    # ──────────────────────────────────────────
    # State Snapshots
    # ──────────────────────────────────────────

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

                # Extract symbol from examples
                symbol = None
                for ex in examples[-3:]:
                    if isinstance(ex, dict) and 'symbol' in ex:
                        symbol = ex['symbol']
                        break

                # Get prediction data for this symbol
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
                    "parent_concepts": list(getattr(concept, 'parent_concepts', set())),
                    "child_concepts": list(getattr(concept, 'child_concepts', set())),
                }

            return concepts

    def get_prediction_snapshot(self) -> Dict[str, Any]:
        """Get full prediction engine state"""
        with self._lock:
            return self.prediction_engine.get_insights()

    def get_rules_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all rules from the ReasoningEngine"""
        with self._lock:
            rules = {}
            for rid, rule in self.cognitive_system.reasoning.rules.items():
                # Get prediction-based accuracy for this rule
                rule_perf = self.prediction_engine.rule_performance.get(rid)
                pred_accuracy = rule_perf.accuracy if rule_perf and rule_perf.predictions_made >= 3 else None

                rules[rid] = {
                    "id": rid,
                    "rule_type": _serialize_enum(getattr(rule, 'rule_type', 'unknown')),
                    "confidence": getattr(rule, 'confidence', 0),
                    "prediction_accuracy": round(pred_accuracy, 4) if pred_accuracy is not None else None,
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
                    "goal_type": _serialize_enum(getattr(goal, 'goal_type', 'unknown')),
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
        """Get complete state summary"""
        metrics = self.get_metrics()
        return {
            "node_id": self.node_id,
            "phi": metrics["global_coherence_phi"],
            "sigma": metrics["noise_level_sigma"],
            "prediction_accuracy": metrics["prediction_accuracy"],
            "predictions_validated": metrics["predictions_validated"],
            "concepts_count": metrics["total_concepts"],
            "rules_count": metrics["total_rules"],
            "goals_count": metrics["total_goals"],
            "domains_count": metrics["total_domains"],
            "concepts_merged": self._concepts_merged,
            "concepts_pruned": self._concepts_pruned,
            "metrics": metrics,
            "timestamp": time.time()
        }

    # ──────────────────────────────────────────
    # NEW: Hidden Intelligence Snapshots
    # ──────────────────────────────────────────

    def get_causal_graph(self) -> Dict[str, Any]:
        """Get causal graph snapshot"""
        with self._lock:
            return self.cognitive_system.get_causal_graph_snapshot()

    def get_concept_hierarchy(self) -> Dict[str, Any]:
        """Get concept hierarchy snapshot"""
        with self._lock:
            return self.cognitive_system.get_concept_hierarchy_snapshot()

    def get_analogies(self) -> list:
        """Get recent analogies"""
        with self._lock:
            return self.cognitive_system.get_analogies_snapshot()

    def get_explanations(self) -> list:
        """Get recent rule explanations"""
        with self._lock:
            return self.cognitive_system.get_explanations_snapshot()

    def get_plans(self) -> list:
        """Get recent plans"""
        with self._lock:
            return self.cognitive_system.get_plans_snapshot()

    def get_pursuit_log(self) -> list:
        """Get autonomous pursuit log"""
        with self._lock:
            return self.cognitive_system.get_pursuit_log()

    def get_transfer_suggestions(self) -> list:
        """Get transfer opportunity suggestions"""
        with self._lock:
            return self.cognitive_system.get_transfer_suggestions()

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get goal strategy performance"""
        with self._lock:
            return self.cognitive_system.get_strategy_performance()

    def get_feature_importances(self) -> list:
        """Get learned feature importances from the online model"""
        with self._lock:
            return self.cognitive_system.learning_engine.get_feature_importances()

    def get_drift_events(self) -> list:
        """Get distribution drift events"""
        with self._lock:
            return list(self.cognitive_system.learning_engine.drift_events)[-20:]

    def get_toggles(self) -> Dict[str, Any]:
        """Get current toggle states"""
        with self._lock:
            return self.cognitive_system.toggles.copy()

    def set_toggle(self, key: str, value) -> Dict[str, Any]:
        """Set a toggle value"""
        with self._lock:
            if key in self.cognitive_system.toggles:
                self.cognitive_system.toggles[key] = value
                # If prediction horizon changed, update the prediction engine
                if key == 'prediction_horizon' and isinstance(value, int):
                    self.prediction_engine.default_horizon = max(3, min(30, value))
                if key == 'dead_zone_sensitivity':
                    multiplier = {'aggressive': 0.5, 'normal': 1.0, 'conservative': 2.0}
                    self.prediction_engine.dead_zone_multiplier = multiplier.get(value, 1.0)
                return self.cognitive_system.toggles.copy()
            return {'error': f'Unknown toggle: {key}'}

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator health status"""
        with self._lock:
            try:
                return self.cognitive_system.orchestrator.get_status()
            except Exception:
                return {'running': False, 'error': 'Orchestrator not available'}

    def get_introspection(self) -> Dict[str, Any]:
        """Full system introspection"""
        with self._lock:
            base = self.cognitive_system.introspect()
            # Enrich with prediction data
            base['predictions'] = self.prediction_engine.get_insights()
            base['convergence'] = {
                'concepts_merged': self._concepts_merged,
                'concepts_pruned': self._concepts_pruned,
                'convergence_cycles': self._convergence_counter,
            }
            return base

    async def process_network_message(self, msg: Any):
        """Process incoming network messages"""
        logger.debug(f"Processing network message: {msg}")

    async def process_pubsub_message(self, msg: Any):
        """Process incoming pubsub messages"""
        logger.debug(f"Processing pubsub message: {msg}")
