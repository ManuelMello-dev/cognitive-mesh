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
            max_generations=10,
            population_size=20
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
        
        # Performance tracking
        self.cognitive_metrics = {
            'concepts_formed': 0,
            'rules_learned': 0,
            'analogies_found': 0,
            'goals_achieved': 0,
            'knowledge_transfers': 0,
            'causal_links_discovered': 0,
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
        
        # 1. Learn from observation
        learning_result = self.learning_engine.process_observation(
            observation,
            outcome=outcome
        )
        results['learning'] = learning_result
        
        # 2. Form abstractions and concepts
        concept_id = self.abstraction.observe(observation)
        if concept_id:
            results['concept'] = concept_id
            self.cognitive_metrics['concepts_formed'] += 1
            
            # Add to per-ticker domain
            self.cross_domain.add_concept_to_domain(domain, concept_id)
            self.active_concepts.add(concept_id)
            
            # Also add to meta-domain (crypto or stock)
            meta = "crypto" if domain.startswith("crypto:") else "stock"
            self.cross_domain.add_concept_to_domain(meta, concept_id)
        
        # 3. Extract RICH facts for reasoning
        facts = self._extract_facts(observation, domain)
        for fact in facts:
            self.reasoning.assert_fact(fact)
        
        # 4. Attempt inference
        inferred = self.reasoning.infer(max_steps=3)
        if inferred:
            results['inferred_facts'] = list(inferred)
        
        # 5. Store observation for rule learning
        enriched_obs = self._enrich_observation(observation, domain)
        self._observation_history.append(enriched_obs)
        if len(self._observation_history) > self._max_observation_history:
            self._observation_history = self._observation_history[-self._max_observation_history:]
        
        # 6. Look for cross-domain patterns (using meta-domains)
        if len(self.cross_domain.domains) > 2:
            self._check_cross_domain_opportunities(domain)
        
        return results
    
    def _enrich_observation(self, observation: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Enrich observation with derived features for rule learning."""
        enriched = dict(observation)
        symbol = observation.get('symbol', '')
        price = observation.get('price', 0)
        
        # Track price history
        if symbol and price:
            self._price_history[symbol].append(price)
            if len(self._price_history[symbol]) > self._max_price_history:
                self._price_history[symbol] = self._price_history[symbol][-self._max_price_history:]
            
            history = self._price_history[symbol]
            if len(history) >= 2:
                pct_change = (history[-1] - history[-2]) / history[-2] * 100 if history[-2] != 0 else 0
                enriched['pct_change'] = round(pct_change, 4)
                enriched['direction'] = 'up' if pct_change > 0.01 else ('down' if pct_change < -0.01 else 'critical')
            
            if len(history) >= 5:
                avg_5 = sum(history[-5:]) / 5
                enriched['above_ma5'] = price > avg_5
                enriched['volatility_5'] = round(
                    (max(history[-5:]) - min(history[-5:])) / avg_5 * 100, 4
                ) if avg_5 > 0 else 0
        
        enriched['asset_type'] = 'crypto' if domain.startswith('crypto:') else 'stock'
        enriched['domain'] = domain
        
        return enriched
    
    def _extract_facts(self, observation: Dict[str, Any], domain: str) -> List[str]:
        """Extract rich logical facts from observation for reasoning."""
        facts = []
        symbol = observation.get('symbol', 'unknown')
        price = observation.get('price', 0)
        
        # Asset type facts
        asset_type = 'crypto' if domain.startswith('crypto:') else 'stock'
        facts.append(f"is_{asset_type}({symbol})")
        
        # Price history-based facts
        history = self._price_history.get(symbol, [])
        if len(history) >= 2 and history[-2] != 0:
            pct_change = (history[-1] - history[-2]) / history[-2] * 100
            if pct_change > 1.0:
                facts.append(f"rising({symbol})")
                facts.append(f"bullish_signal({symbol})")
            elif pct_change < -1.0:
                facts.append(f"falling({symbol})")
                facts.append(f"bearish_signal({symbol})")
            else:
                facts.append(f"stable({symbol})")
        
        # Ensure facts are always returned as a list of strings, not raised as exceptions
        return [str(f) for f in facts if isinstance(f, str)]

    def _check_cross_domain_opportunities(self, current_domain: str):
        """Check for opportunities to transfer knowledge from other domains."""
        # Simple implementation: look for analogous domains
        meta = "crypto" if current_domain.startswith("crypto:") else "stock"
        other_domains = [d for d in self._meta_domains[meta] if d != current_domain]
        
        for other in other_domains:
            mapping = self.cross_domain.discover_domain_mapping(other, current_domain)
            if mapping and mapping.confidence > 0.7:
                self.cognitive_metrics['analogies_found'] += 1
                logger.info(f"Found cross-domain mapping: {other} -> {current_domain}")

    def get_metrics(self) -> Dict[str, Any]:
        """Return combined cognitive metrics."""
        metrics = self.cognitive_metrics.copy()
        metrics.update({
            'active_concepts': len(self.active_concepts),
            'iteration': self.iteration,
            'learning_rate': self.learning_engine.learning_rate
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
            'toggles': self.toggles,
            'recent_analogies': list(self._recent_analogies),
            'causal_log': list(self._causal_discovery_log),
            'explanations': list(self._recent_explanations),
            'plans': list(self._recent_plans),
            'pursuits': list(self._pursuit_log),
            'transfer_suggestions': self._transfer_suggestions_cache,
            'causal_graph': self.get_causal_graph_snapshot(),
            'concept_hierarchy': self.get_concept_hierarchy_snapshot(),
            'feature_importances': self.get_feature_importances(),
            'drift_events': self.get_drift_events(),
            'strategy_performance': self.get_strategy_performance()
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
