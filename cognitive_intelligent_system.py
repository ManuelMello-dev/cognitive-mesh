"""
Advanced Cognitive Intelligent System
Integrates abstraction, reasoning, cross-domain transfer, and goal formation
with continuous learning, self-evolution, and always-on operation
"""
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import json

# Core engines
from continuous_learning_engine import ContinuousLearningEngine
from self_writing_engine import SelfEvolvingSystem
from always_on_orchestrator import AlwaysOnOrchestrator

# Cognitive capabilities
from abstraction_engine import AbstractionEngine
from reasoning_engine import ReasoningEngine, RuleType
from cross_domain_engine import CrossDomainEngine
from goal_formation_system import OpenEndedGoalSystem, GoalGenerationContext, GoalType
from resonant_memory import ResonantMemoryGeometry
from core.constitutional_physics import ConstitutionalMeshPhysics
from core.contracts import ConstitutionalOutput

logger = logging.getLogger(__name__)


class CognitiveIntelligentSystem:
    """
    Advanced intelligent system with full cognitive capabilities:
    - Continuous learning
    - Self-writing code evolution
    - Always-on operation
    - Abstraction and concept formation
    - Logical reasoning and planning
    - Cross-domain knowledge transfer
    - Open-ended goal formation
    """
    
    def __init__(
        self,
        system_id: str = "cognitive-ai-system",
        feature_dim: int = 50,
        learning_rate: float = 0.01,
        enable_all_features: bool = True
    ):
        self.system_id = system_id
        self.enable_all = enable_all_features
        
        logger.info(f"Initializing {system_id} with full cognitive capabilities...")
        
        # Core learning and evolution
        self.learning_engine = ContinuousLearningEngine(
            feature_dim=feature_dim,
            learning_rate=learning_rate,
            pattern_mining=True,
            auto_adapt=True
        )
        
        self.code_evolver = SelfEvolvingSystem(
            max_gen=10,
            pop_size=20
        )
        
        self.orchestrator = AlwaysOnOrchestrator(
            checkpoint_interval=300,
            auto_restart=True
        )
        
        # Cognitive capabilities
        self.abstraction = AbstractionEngine(
            max_concepts=1000,
            min_examples_for_concept=2  # Reduced from 3 for faster initial activation
        )
        
        self.reasoning = ReasoningEngine(
            max_rules=1000,
            min_confidence=0.6,
            max_inference_depth=5
        )
        
        self.cross_domain = CrossDomainEngine(
            min_mapping_confidence=0.5,
            enable_auto_transfer=True
        )
        
        self.goals = OpenEndedGoalSystem(
            max_active_goals=5,
            exploration_rate=0.3,
            enable_meta_goals=True
        )
        
        # System state
        self.current_domain = None
        self.active_concepts = set()
        self.running = False
        self.iteration = 0
        
        # Data streams
        self.data_streams: Dict[str, Any] = {}
        
        # --- NEW: Observation history for rule learning ---
        self._observation_history: List[Dict[str, Any]] = []
        self._max_observation_history = 500
        
        # --- NEW: Domain grouping for cross-domain (crypto vs stock vs macro) ---
        self._meta_domains: Dict[str, Set[str]] = defaultdict(set)
        # Register the two meta-domains
        self._ensure_meta_domain("crypto")
        self._ensure_meta_domain("stock")
        
        # --- NEW: Price history per symbol for richer fact extraction ---
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._max_price_history = 50

        # Constitutional physics — the runtime foundation beneath all cognition
        self.constitutional_physics = ConstitutionalMeshPhysics(dimension=32)
        self._last_constitutional_snapshot: Dict[str, Any] = {}

        # Resonant memory geometry — preserves temporal phase relations across rings
        self.resonant_memory = ResonantMemoryGeometry(max_rings=512, resonance_horizon=96)
        
        # Performance tracking
        self.cognitive_metrics = {
            'concepts_formed': 0,
            'rules_learned': 0,
            'analogies_found': 0,
            'goals_achieved': 0,
            'knowledge_transfers': 0,
            'causal_links_discovered': 0,
            'resonance_events': 0,
            'memory_reconstructions': 0,
        }

        # --- NEW: Toggles for optional engines ---
        self.toggles = {
            'causal_discovery': True,
            'self_evolution': False,
            'autonomous_reasoning': False,
            'prediction_horizon': 5,
            'dead_zone_sensitivity': 'normal',  # aggressive / normal / conservative
        }

        # --- NEW: Caches for hidden intelligence ---
        from collections import deque
        self._recent_analogies: deque = deque(maxlen=100)
        self._recent_explanations: deque = deque(maxlen=100)
        self._recent_plans: deque = deque(maxlen=50)
        self._causal_discovery_log: deque = deque(maxlen=50)
        self._transfer_suggestions_cache: List[Dict] = []
        self._pursuit_log: deque = deque(maxlen=100)

        logger.info("System initialized with all cognitive capabilities")
    
    def _ensure_meta_domain(self, meta_domain: str):
        """Register a meta-domain for cross-domain analysis."""
        if meta_domain not in self.cross_domain.domains:
            self.cross_domain.register_domain(meta_domain, meta_domain)
    
    def register_data_stream(self, stream_id: str, fetch_fn, domain: str):
        """Register data stream with domain association"""
        self.data_streams[stream_id] = {
            'fetch_fn': fetch_fn,
            'domain': domain
        }
        
        # Register per-ticker domain
        if domain not in self.cross_domain.domains:
            self.cross_domain.register_domain(domain, domain)
            logger.debug(f"Registered new domain: {domain}")
        
        # Also register to meta-domain (crypto or stock)
        if domain.startswith("crypto:"):
            self._meta_domains["crypto"].add(domain)
        elif domain.startswith("stock:"):
            self._meta_domains["stock"].add(domain)
        
        logger.debug(f"Registered stream: {stream_id} in domain {domain}")
    
    def process_observation(
        self,
        observation: Dict[str, Any],
        domain: str,
        outcome: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process observation through full cognitive pipeline
        """
        results = {}

        # 0. Constitutional layer — local identity evolves relative to a global attractor
        if domain not in self.cross_domain.domains:
            self.cross_domain.register_domain(domain, domain)
        meta = domain.split(':')[0] if ':' in domain else domain
        self._ensure_meta_domain(meta)
        if meta in self._meta_domains:
            self._meta_domains[meta].add(domain)

        enriched_obs = self._enrich_observation(observation, domain)
        constitutional_snapshot = self.constitutional_physics.observe(enriched_obs, domain)
        self._last_constitutional_snapshot = constitutional_snapshot
        enriched_obs['constitutional_phi'] = constitutional_snapshot.get('phi', 0.5)
        enriched_obs['constitutional_sigma'] = constitutional_snapshot.get('sigma', 0.5)
        enriched_obs['constitutional_coherence'] = constitutional_snapshot.get('coherence', 0.0)
        enriched_obs['constitutional_drift'] = constitutional_snapshot.get('drift', 0.0)
        enriched_obs['constitutional_regime'] = constitutional_snapshot.get('regime', 'critical')
        enriched_obs['constitutional_collapse_probability'] = constitutional_snapshot.get('collapse_probability', 0.0)
        checkpoint_state = constitutional_snapshot.get('checkpoint_state', {}) or {}
        interference_state = constitutional_snapshot.get('interference_state', {}) or {}
        logos_state = constitutional_snapshot.get('logos_state', {}) or {}
        enriched_obs['constitutional_checkpoint_continuity'] = checkpoint_state.get('continuity', 0.0)
        enriched_obs['constitutional_checkpoint_amplification'] = checkpoint_state.get('amplification', 0.0)
        enriched_obs['constitutional_interference_net'] = interference_state.get('net', 0.0)
        enriched_obs['constitutional_interference_constructive'] = interference_state.get('constructive', 0.0)
        enriched_obs['constitutional_interference_destructive'] = interference_state.get('destructive', 0.0)
        enriched_obs['constitutional_logos_reflective_energy'] = logos_state.get('reflective_energy', 0.0)
        results['constitutional'] = constitutional_snapshot
        
        # 1. Learn from observation through constitutional context
        learning_result = self.learning_engine.process_observation(
            enriched_obs,
            outcome=outcome
        )
        results['learning'] = learning_result
        
        # 2. Form abstractions and concepts
        concept_output = self.abstraction.observe(enriched_obs)
        if concept_output:
            concept_id = getattr(concept_output, 'concept_id', concept_output)
            results['concept'] = concept_output
            # Signal new concept for activity log
            results['new_concept'] = concept_id not in self.active_concepts
            concept_obj = self.abstraction.concepts.get(concept_id)
            results['concept_name'] = getattr(concept_obj, 'name', concept_id) if concept_obj else concept_id
            self.cognitive_metrics['concepts_formed'] += 1

            # Add to per-entity domain
            self.cross_domain.add_concept_to_domain(domain, concept_id)
            self.active_concepts.add(concept_id)

            # Also add to meta-domain
            meta_domain = "crypto" if domain.startswith("crypto:") else ("stock" if domain.startswith("stock:") else meta)
            self.cross_domain.add_concept_to_domain(meta_domain, concept_id)
        
        # 3. Extract RICH facts for reasoning
        facts = self._extract_facts(enriched_obs, domain)
        for fact in facts:
            self.reasoning.assert_fact(fact)
        
        # 4. Attempt inference
        inferred = self.reasoning.infer(max_steps=3)
        if inferred:
            results['inferred_facts'] = list(inferred)
        
        # 5. Store observation for rule learning
        self._observation_history.append(enriched_obs)
        if len(self._observation_history) > self._max_observation_history:
            self._observation_history = self._observation_history[-self._max_observation_history:]
        
        # 6. Resonant memory geometry — treat each enriched observation as a ring in time
        resonance = self.resonant_memory.observe(
            enriched_obs,
            domain,
            phi_hint=constitutional_snapshot.get('phi', 0.5),
            sigma_hint=constitutional_snapshot.get('sigma', 0.5),
            constitutional_context=constitutional_snapshot,
        )
        results['resonance'] = resonance
        self.cognitive_metrics['resonance_events'] += resonance.get('accessible_rings', 0)
        if resonance.get('reconstruction_confidence', 0) > 0:
            self.cognitive_metrics['memory_reconstructions'] += 1

        # 7. Look for cross-domain patterns (using meta-domains)
        if len(self.cross_domain.domains) > 2:
            self._check_cross_domain_opportunities(domain)
        
        return results
    
    def _enrich_observation(self, observation: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Enrich observation with derived features for rule learning.
        Works with any domain: uses 'entity_id' (or legacy 'symbol') and 'value'
        (or legacy 'price') as the primary identifiers.
        """
        enriched = dict(observation)

        # Support both generic and legacy field names
        entity_id = observation.get('entity_id') or observation.get('symbol') or ''
        value = observation.get('value') or observation.get('price') or 0

        # Track value history per entity
        if entity_id and value:
            history_key = f"{domain}:{entity_id}"
            self._price_history[history_key].append(float(value))
            if len(self._price_history[history_key]) > self._max_price_history:
                self._price_history[history_key] = self._price_history[history_key][-self._max_price_history:]

            history = self._price_history[history_key]
            if len(history) >= 2 and history[-2] != 0:
                pct_change = (history[-1] - history[-2]) / history[-2] * 100
                enriched['pct_change'] = round(pct_change, 4)
                enriched['direction'] = 'up' if pct_change > 0.1 else ('down' if pct_change < -0.1 else 'stable')

            if len(history) >= 5:
                avg_5 = sum(history[-5:]) / 5
                enriched['above_ma5'] = value > avg_5
                enriched['volatility_5'] = round(
                    (max(history[-5:]) - min(history[-5:])) / avg_5 * 100, 4
                ) if avg_5 > 0 else 0

        # Normalise domain label for rule mining
        enriched['entity_id'] = entity_id
        enriched['domain_prefix'] = domain.split(':')[0] if ':' in domain else domain
        enriched['domain'] = domain

        return enriched
    
    def _extract_facts(self, observation: Dict[str, Any], domain: str) -> List[str]:
        """
        Extract logical facts from an observation for reasoning.
        Domain-agnostic: uses entity_id/value rather than symbol/price.
        """
        facts = []
        entity_id = observation.get('entity_id') or observation.get('symbol') or 'unknown'
        value = observation.get('value') or observation.get('price') or 0
        domain_prefix = domain.split(':')[0] if ':' in domain else domain

        # Domain membership fact
        facts.append(f"in_domain({entity_id},{domain_prefix})")

        # Value trend facts from history
        history_key = f"{domain}:{entity_id}"
        history = self._price_history.get(history_key, [])
        if len(history) >= 2 and history[-2] != 0:
            pct_change = (history[-1] - history[-2]) / history[-2] * 100
            if pct_change > 1.0:
                facts.append(f"rising({entity_id})")
                facts.append(f"positive_signal({entity_id})")
            elif pct_change < -1.0:
                facts.append(f"falling({entity_id})")
                facts.append(f"negative_signal({entity_id})")
            else:
                facts.append(f"stable({entity_id})")

        coherence = float(observation.get('constitutional_coherence', 0.0))
        phi = float(observation.get('constitutional_phi', 0.5))
        sigma = float(observation.get('constitutional_sigma', 0.5))
        regime = str(observation.get('constitutional_regime', 'critical'))

        if phi >= 0.7:
            facts.append(f"coherent({entity_id})")
        elif phi <= 0.35:
            facts.append(f"fragmented({entity_id})")

        if sigma >= 0.65:
            facts.append(f"noise_high({entity_id})")
        elif sigma <= 0.35:
            facts.append(f"noise_low({entity_id})")

        if coherence > 0.0:
            facts.append(f"approaching_attractor({entity_id})")
        elif coherence < 0.0:
            facts.append(f"drifting_from_attractor({entity_id})")

        facts.append(f"regime={regime}({entity_id})")

        # Boolean/categorical fields become direct facts
        for k, v in observation.items():
            if isinstance(v, bool):
                if v:
                    facts.append(f"{k}({entity_id})")
            elif isinstance(v, str) and k not in ('entity_id', 'symbol', 'domain', 'timestamp'):
                facts.append(f"{k}={v}({entity_id})")

        return [str(f) for f in facts if isinstance(f, str)]

    def _check_cross_domain_opportunities(self, current_domain: str):
        """Check for cross-domain (crypto <-> stock) mapping opportunities.
        
        Only runs every 50 iterations and only compares the two meta-domains
        (crypto vs stock), not intra-domain symbol pairs, to avoid O(N²) log spam.
        """
        # Rate-limit: only run every 50 observations
        if self.iteration % 50 != 0:
            return

        # Only do true cross-domain: crypto meta-domain vs stock meta-domain
        crypto_domain = "crypto"
        stock_domain = "stock"
        if crypto_domain not in self.cross_domain.domains or stock_domain not in self.cross_domain.domains:
            return

        mapping = self.cross_domain.discover_domain_mapping(crypto_domain, stock_domain)
        if mapping and mapping.confidence > 0.5:
            self.cognitive_metrics['analogies_found'] += 1
            logger.info(
                f"Cross-domain mapping: crypto <-> stock "
                f"(conf: {mapping.confidence:.2f}, "
                f"{len(mapping.concept_mappings)} concept pairs)"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Return combined cognitive metrics."""
        metrics = self.cognitive_metrics.copy()
        resonance_metrics = self.resonant_memory.get_snapshot().get('metrics', {})
        constitutional = self._last_constitutional_snapshot or {}
        checkpoint_state = constitutional.get('checkpoint_state', {}) if constitutional else {}
        interference_state = constitutional.get('interference_state', {}) if constitutional else {}
        logos_state = constitutional.get('logos_state', {}) if constitutional else {}
        metrics.update({
            'active_concepts': len(self.active_concepts),
            'iteration': self.iteration,
            'learning_rate': self.learning_engine.learning_rate,
            'resonant_memory_rings': resonance_metrics.get('rings', 0),
            'phi_access_window': resonance_metrics.get('phi_access_window', 0),
            'average_resonance': resonance_metrics.get('average_resonance', 0),
            'memory_reconstruction_confidence': resonance_metrics.get('last_reconstruction_confidence', 0),
            'constitutional_phi': constitutional.get('phi', 0.5),
            'constitutional_sigma': constitutional.get('sigma', 0.5),
            'constitutional_drift': constitutional.get('drift', 0.0),
            'constitutional_regime': constitutional.get('regime', 'critical'),
            'constitutional_stability': constitutional.get('stability', 0.5),
            'constitutional_collapse_probability': constitutional.get('collapse_probability', 0.0),
            'constitutional_checkpoint_continuity': checkpoint_state.get('continuity', 0.0),
            'constitutional_checkpoint_amplification': checkpoint_state.get('amplification', 0.0),
            'constitutional_interference_net': interference_state.get('net', 0.0),
            'constitutional_interference_constructive': interference_state.get('constructive', 0.0),
            'constitutional_interference_destructive': interference_state.get('destructive', 0.0),
            'constitutional_logos_reflective_energy': logos_state.get('reflective_energy', 0.0),
        })
        return metrics

    def get_concepts_snapshot(self) -> Dict[str, Any]:
        """Return snapshot of current concepts."""
        return {cid: self.abstraction.concepts[cid].to_dict() for cid in self.active_concepts if cid in self.abstraction.concepts}

    def get_rules_snapshot(self) -> Dict[str, Any]:
        """Return snapshot of learned rules."""
        return {rid: rule.to_dict() for rid, rule in self.reasoning.rules.items()}

    def get_goals_snapshot(self) -> Dict[str, Any]:
        """Return snapshot of active goals."""
        return {gid: goal.to_dict() for gid, goal in self.goals.goals.items()}

    def get_cross_domain_snapshot(self) -> Dict[str, Any]:
        """Return snapshot of cross-domain mappings."""
        return {mid: mapping.to_dict() for mid, mapping in self.cross_domain.mappings.items()}

    def get_introspection(self) -> Dict[str, Any]:
        """Deep system introspection."""
        return {
            'system_id': self.system_id,
            'metrics': self.get_metrics(),
            'active_concepts': len(self.active_concepts),
            'domains': list(self.cross_domain.domains.keys()),
            'recent_analogies': list(self._recent_analogies),
            'recent_explanations': list(self._recent_explanations),
            'recent_plans': list(self._recent_plans),
            'pursuits': list(self._pursuit_log),
            'transfer_suggestions': self._transfer_suggestions_cache,
            'causal_graph': self.get_causal_graph_snapshot(),
            'concept_hierarchy': self.get_concept_hierarchy_snapshot(),
            'feature_importances': self.get_feature_importances(),
            'drift_events': self.get_drift_events(),
            'strategy_performance': self.get_strategy_performance(),
            'resonant_memory': self.get_resonant_memory_snapshot(),
            'constitutional_physics': self.constitutional_physics.export_state(),
        }


    async def ingest(self, observation: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Directly ingest an observation (GPT I/O)."""
        self.iteration += 1
        return self.process_observation(observation, domain)

    def get_causal_graph_snapshot(self) -> Dict[str, Any]:
        """Return snapshot of discovered causal links."""
        return {
            'causal_links': list(self._causal_discovery_log),
            'discovered_count': self.cognitive_metrics['causal_links_discovered']
        }

    def get_concept_hierarchy_snapshot(self) -> Dict[str, Any]:
        """Return snapshot of the concept hierarchy."""
        return self.abstraction.get_concept_hierarchy()

    def get_analogies_snapshot(self) -> list:
        """Return snapshot of recent analogies."""
        return list(self._recent_analogies)

    def get_explanations_snapshot(self) -> list:
        """Return snapshot of recent rule explanations."""
        return list(self._recent_explanations)

    def get_plans_snapshot(self) -> list:
        """Return snapshot of recent plans."""
        return list(self._recent_plans)

    def get_pursuit_log(self) -> list:
        """Return snapshot of the autonomous pursuit log."""
        return list(self._pursuit_log)

    def get_transfer_suggestions_snapshot(self) -> list:
        """Return snapshot of knowledge transfer suggestions."""
        return self._transfer_suggestions_cache

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Return goal strategy performance."""
        return self.goals.get_insights().get('strategy_performance', {})

    def get_feature_importances(self) -> Dict[str, float]:
        """Return learned feature importances."""
        return self.learning_engine.get_insights().get('feature_importances', {})

    def get_drift_events(self) -> list:
        """Return distribution drift events."""
        return self.learning_engine.get_insights().get('drift_events', [])

    def get_constitutional_output(self) -> Optional[ConstitutionalOutput]:
        """Return the latest constitutional snapshot as a structured contract."""
        if not self._last_constitutional_snapshot:
            return None
        snapshot = dict(self._last_constitutional_snapshot)
        return ConstitutionalOutput(
            agent_id=str(snapshot.get('agent_id', '')),
            attractor_id=str(snapshot.get('attractor_id', '')),
            domain=str(snapshot.get('domain', '')),
            phi=float(snapshot.get('phi', 0.5)),
            sigma=float(snapshot.get('sigma', 0.5)),
            coherence=float(snapshot.get('coherence', 0.0)),
            drift=float(snapshot.get('drift', 0.0)),
            stability=float(snapshot.get('stability', 0.5)),
            regime=str(snapshot.get('regime', 'critical')),
            awareness=float(snapshot.get('awareness', 0.5)),
            assignment_distance=float(snapshot.get('assignment_distance', 0.0)),
            distance_to_attractor=float(snapshot.get('distance_to_attractor', 0.0)),
            gradient_norm=float(snapshot.get('gradient_norm', 0.0)),
            collapse_probability=float(snapshot.get('collapse_probability', 0.0)),
            z_state=list(snapshot.get('z_state', [])),
            z_prime_state=list(snapshot.get('z_prime_state', [])),
            z_double_prime_state=list(snapshot.get('z_double_prime_state', [])),
            checkpoint_state=dict(snapshot.get('checkpoint_state', {})),
            interference_state=dict(snapshot.get('interference_state', {})),
            logos_state=dict(snapshot.get('logos_state', {})),
            z_cubed_state=dict(snapshot.get('z_cubed_state', {})),
        )

    def get_resonant_memory_snapshot(self) -> Dict[str, Any]:
        """Return the current resonant memory geometry state."""
        return self.resonant_memory.get_snapshot()
