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
            min_examples_for_concept=3
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
        volume = observation.get('volume', 0)
        
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
                facts.append(f"critical({symbol})")
            
            if abs(pct_change) > 5.0:
                facts.append(f"volatile({symbol})")
            
            if pct_change > 0:
                facts.append(f"positive_momentum({symbol})")
            else:
                facts.append(f"negative_momentum({symbol})")
        
        if len(history) >= 5:
            avg_5 = sum(history[-5:]) / 5
            if price > avg_5:
                facts.append(f"above_ma5({symbol})")
            else:
                facts.append(f"below_ma5({symbol})")
            
            volatility = (max(history[-5:]) - min(history[-5:])) / avg_5 * 100 if avg_5 > 0 else 0
            if volatility > 5:
                facts.append(f"high_volatility({symbol})")
            elif volatility < 1:
                facts.append(f"low_volatility({symbol})")
        
        # Volume facts
        if volume and volume > 0:
            facts.append(f"has_volume({symbol})")
            if volume > 1_000_000:
                facts.append(f"high_volume({symbol})")
        
        # Market state facts (based on observation fields)
        market_state = observation.get('market_state', '')
        if market_state:
            facts.append(f"market_state_{market_state}")
        
        return facts
    
    def learn_rules_from_history(self):
        """
        Periodically learn association rules from accumulated observations.
        This is the critical missing piece — turns observations into knowledge.
        """
        if len(self._observation_history) < 10:
            return 0
        
        # Learn from recent observations
        recent = self._observation_history[-100:]
        
        rules_before = len(self.reasoning.rules)
        
        try:
            self.reasoning.learn_rules_from_observations(recent, min_support=3)
        except Exception as e:
            logger.warning(f"Rule learning error: {e}")
        
        rules_after = len(self.reasoning.rules)
        new_rules = rules_after - rules_before
        
        if new_rules > 0:
            self.cognitive_metrics['rules_learned'] += new_rules
            logger.info(f"Learned {new_rules} new rules (total: {rules_after})")
        
        return new_rules
    
    def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform higher-level cognitive processing
        - Abstract thinking
        - Analogical reasoning
        - Goal-directed planning
        - Causal discovery (NEW)
        """
        thoughts = {}

        # Abstract reasoning — find and LOG analogies
        if len(self.active_concepts) > 0:
            recent_concepts = list(self.active_concepts)[-5:]
            if len(recent_concepts) >= 2:
                analogies = self.abstraction.find_analogies(recent_concepts[0])
                if analogies:
                    thoughts['analogies'] = [a.to_dict() for a in analogies[:3]]
                    self.cognitive_metrics['analogies_found'] += len(analogies)
                    # Cache for API exposure
                    for a in analogies[:3]:
                        self._recent_analogies.append({
                            'timestamp': datetime.now().isoformat(),
                            **a.to_dict()
                        })

        # Logical reasoning — explain and CACHE explanations
        if self.reasoning.facts:
            recent_facts = list(self.reasoning.facts)[-5:]
            thoughts['explanations'] = {}
            for fact in recent_facts:
                explanation = self.reasoning.explain(fact)
                if explanation:
                    thoughts['explanations'][fact] = explanation
                    self._recent_explanations.append({
                        'timestamp': datetime.now().isoformat(),
                        'fact': fact,
                        'explanation': explanation,
                    })

        # Goal-oriented thinking
        if len(self.goals.goals) > 0:
            active_goals = [
                self.goals.goals[gid]
                for gid in self.goals.active_goals
            ]
            thoughts['active_goals'] = [g.to_dict() for g in active_goals]

        # --- NEW: Causal discovery (if toggled on) ---
        if self.toggles.get('causal_discovery', False) and len(self._observation_history) >= 30:
            try:
                new_links = self.reasoning.discover_causal_relationships(
                    self._observation_history[-100:]
                )
                if new_links:
                    self.cognitive_metrics['causal_links_discovered'] += new_links
                    self._causal_discovery_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'new_links': new_links,
                        'total_causal_links': sum(
                            len(v) for v in self.reasoning.causal_graph.values()
                        ),
                    })
                    logger.info(f"Causal discovery: {new_links} new links")
            except Exception as e:
                logger.warning(f"Causal discovery error: {e}")

        # --- NEW: Transfer suggestions ---
        try:
            suggestions = self.cross_domain.suggest_transfer_opportunities()
            if suggestions:
                self._transfer_suggestions_cache = suggestions
        except Exception as e:
            logger.debug(f"Transfer suggestion error: {e}")

        return thoughts
    
    def set_goal(self, goal_description: str, goal_type: GoalType, criteria: Dict[str, Any]):
        """Explicitly set a goal"""
        from goal_formation_system import Goal, GoalStatus
        
        goal_id = f"goal_{self.goals.goal_counter}"
        self.goals.goal_counter += 1
        
        goal = Goal(
            goal_id=goal_id,
            goal_type=goal_type,
            description=goal_description,
            success_criteria=criteria,
            priority=0.8,
            status=GoalStatus.PROPOSED
        )
        
        self.goals.goals[goal_id] = goal
        logger.info(f"Set explicit goal: {goal_description}")
        
        return goal_id
    
    def generate_autonomous_goals(self) -> List[str]:
        """Generate goals autonomously based on current state"""
        
        # Create context for goal generation
        context = GoalGenerationContext(
            observations=list(self.learning_engine.short_term_memory)[-50:],
            patterns=list(self.abstraction.patterns.values()),
            capabilities=self.active_concepts,
            constraints={},
            current_state={
                'concepts': len(self.abstraction.concepts),
                'rules': len(self.reasoning.rules),
                'domains': len(self.cross_domain.domains)
            },
            performance_metrics={
                'learning_accuracy': self.learning_engine.metrics.accuracy,
                'concept_confidence': sum(
                    c.confidence for c in self.abstraction.concepts.values()
                ) / max(len(self.abstraction.concepts), 1)
            }
        )
        
        # Generate goals
        new_goals = self.goals.generate_goals(context)
        
        logger.info(f"Generated {len(new_goals)} autonomous goals")
        
        return [g.goal_id for g in new_goals]
    
    def pursue_goals(self) -> Dict[str, Any]:
        """Actively pursue current goals — now logs plans and pursuits."""

        active_goal_ids = self.goals.select_active_goals()
        results = {}

        for goal_id in active_goal_ids:
            goal = self.goals.goals[goal_id]

            plan = self.reasoning.create_plan(
                goal=goal.description,
                current_state={},
                available_actions=[]
            )

            if plan:
                # Cache the plan for API exposure
                self._recent_plans.append({
                    'timestamp': datetime.now().isoformat(),
                    'goal_id': goal_id,
                    'goal': goal.description,
                    'plan_id': getattr(plan, 'plan_id', str(goal_id)),
                    'steps': getattr(plan, 'steps', []),
                })

                execution_result = self.reasoning.execute_plan(plan)
                results[goal_id] = execution_result

                # Log the pursuit
                self._pursuit_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'goal_id': goal_id,
                    'goal': goal.description,
                    'success': execution_result.get('success', False),
                    'steps_completed': execution_result.get('steps_completed', 0),
                })

                self.goals.update_goal_progress(
                    goal_id,
                    {'achieved': execution_result['success']}
                )

                if execution_result['success']:
                    self.cognitive_metrics['goals_achieved'] += 1

        return results
    
    def _check_cross_domain_opportunities(self, current_domain: str):
        """Check for knowledge transfer opportunities using meta-domains."""
        
        # Use meta-domains (crypto, stock) for cross-domain discovery
        meta_domains_to_check = ["crypto", "stock"]
        
        eligible = []
        for md in meta_domains_to_check:
            if md in self.cross_domain.domains:
                domain_obj = self.cross_domain.domains[md]
                if len(domain_obj.concepts) >= 3:
                    eligible.append(domain_obj)
        
        if len(eligible) < 2:
            return
        
        # Try to discover mappings between meta-domains
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                src = eligible[i].domain_id
                tgt = eligible[j].domain_id
                
                existing = any(
                    m.source_domain == src and m.target_domain == tgt
                    for m in self.cross_domain.mappings.values()
                )
                
                if not existing:
                    mapping = self.cross_domain.discover_domain_mapping(src, tgt)
                    if mapping:
                        self.cognitive_metrics['knowledge_transfers'] += 1
                        logger.info(f"Discovered cross-domain mapping: {src} -> {tgt}")
        
        # Also check per-ticker domains if they have enough concepts
        per_ticker_eligible = [
            d for d in self.cross_domain.domains.values()
            if len(d.concepts) >= 3 and d.domain_id not in meta_domains_to_check
        ]
        
        if len(per_ticker_eligible) >= 2:
            # Sample a few pairs to avoid O(n^2) explosion
            import random
            pairs = []
            for i in range(min(5, len(per_ticker_eligible))):
                for j in range(i + 1, min(5, len(per_ticker_eligible))):
                    pairs.append((per_ticker_eligible[i], per_ticker_eligible[j]))
            
            random.shuffle(pairs)
            for src_d, tgt_d in pairs[:3]:
                existing = any(
                    m.source_domain == src_d.domain_id and m.target_domain == tgt_d.domain_id
                    for m in self.cross_domain.mappings.values()
                )
                if not existing:
                    mapping = self.cross_domain.discover_domain_mapping(
                        src_d.domain_id, tgt_d.domain_id
                    )
                    if mapping:
                        self.cognitive_metrics['knowledge_transfers'] += 1
                        logger.info(
                            f"Discovered cross-domain mapping: "
                            f"{src_d.domain_id} -> {tgt_d.domain_id}"
                        )
    
    def transfer_knowledge(
        self,
        from_domain: str,
        to_domain: str,
        knowledge_type: str = 'pattern'
    ):
        """Transfer knowledge between domains"""
        
        # Get knowledge from source domain
        if knowledge_type == 'pattern':
            patterns = list(self.abstraction.patterns.values())[:3]
            
            for pattern in patterns:
                try:
                    transfer = self.cross_domain.transfer_knowledge(
                        knowledge=pattern,
                        knowledge_type='pattern',
                        source_domain_id=from_domain,
                        target_domain_id=to_domain
                    )
                    
                    if transfer:
                        self.cognitive_metrics['knowledge_transfers'] += 1
                        logger.info(f"Transferred pattern from {from_domain} to {to_domain}")
                except Exception as e:
                    logger.warning(f"Transfer error: {e}")
        
        elif knowledge_type == 'rule':
            relevant_rules = [
                r for r in self.reasoning.rules.values()
                if r.confidence > 0.7
            ]
            
            for rule in relevant_rules[:3]:
                try:
                    transfer = self.cross_domain.transfer_knowledge(
                        knowledge=rule.to_dict(),
                        knowledge_type='rule',
                        source_domain_id=from_domain,
                        target_domain_id=to_domain
                    )
                    
                    if transfer:
                        self.cognitive_metrics['knowledge_transfers'] += 1
                        logger.info(f"Transferred rule from {from_domain} to {to_domain}")
                except Exception as e:
                    logger.warning(f"Transfer error: {e}")
    
    def get_causal_graph_snapshot(self) -> Dict[str, Any]:
        """Return the full causal graph for API exposure."""
        graph = {}
        for cause, links in self.reasoning.causal_graph.items():
            graph[cause] = [
                {
                    'effect': getattr(link, 'effect', str(link)),
                    'lag': getattr(link, 'lag', 0),
                    'correlation': round(getattr(link, 'correlation', 0), 4),
                    'p_value': round(getattr(link, 'p_value', 1.0), 6),
                }
                for link in links
            ]
        return {
            'total_links': sum(len(v) for v in self.reasoning.causal_graph.values()),
            'graph': graph,
            'discovery_log': list(self._causal_discovery_log)[-20:],
        }

    def get_concept_hierarchy_snapshot(self) -> Dict[str, Any]:
        """Return the concept hierarchy for API exposure."""
        try:
            hierarchy = self.abstraction.get_concept_hierarchy()
            return hierarchy
        except Exception:
            return {'levels': {}, 'total_concepts': len(self.abstraction.concepts)}

    def get_analogies_snapshot(self) -> List[Dict]:
        """Return recent analogies for API exposure."""
        return list(self._recent_analogies)[-20:]

    def get_explanations_snapshot(self) -> List[Dict]:
        """Return recent rule explanations for API exposure."""
        return list(self._recent_explanations)[-20:]

    def get_plans_snapshot(self) -> List[Dict]:
        """Return recent plans for API exposure."""
        return list(self._recent_plans)[-20:]

    def get_pursuit_log(self) -> List[Dict]:
        """Return the autonomous pursuit log for API exposure."""
        return list(self._pursuit_log)[-30:]

    def get_transfer_suggestions(self) -> List[Dict]:
        """Return cached transfer suggestions."""
        return self._transfer_suggestions_cache

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Return goal strategy performance from the goal system."""
        try:
            perf = {}
            for strategy, stats in self.goals.strategy_performance.items():
                perf[strategy] = {
                    'attempts': getattr(stats, 'attempts', 0),
                    'successes': getattr(stats, 'successes', 0),
                    'success_rate': round(
                        getattr(stats, 'successes', 0) / max(getattr(stats, 'attempts', 1), 1), 3
                    ),
                }
            return perf
        except Exception:
            return {}

    def introspect(self) -> Dict[str, Any]:
        """Perform system introspection — now includes all hidden intelligence."""

        return {
            'system_id': self.system_id,
            'iteration': self.iteration,
            'cognitive_metrics': self.cognitive_metrics,

            # Learning state
            'learning': self.learning_engine.get_insights(),

            # Abstraction state
            'abstraction': self.abstraction.get_insights(),

            # Reasoning state
            'reasoning': self.reasoning.get_insights(),

            # Cross-domain state
            'cross_domain': self.cross_domain.get_insights(),

            # Goal state
            'goals': self.goals.get_insights(),

            # Active elements
            'active_concepts': len(self.active_concepts),
            'active_domains': len(self.cross_domain.domains),

            # --- NEW: Hidden intelligence now exposed ---
            'causal_graph': self.get_causal_graph_snapshot(),
            'concept_hierarchy': self.get_concept_hierarchy_snapshot(),
            'recent_analogies': self.get_analogies_snapshot(),
            'recent_explanations': self.get_explanations_snapshot(),
            'recent_plans': self.get_plans_snapshot(),
            'pursuit_log': self.get_pursuit_log(),
            'transfer_suggestions': self.get_transfer_suggestions(),
            'strategy_performance': self.get_strategy_performance(),
            'toggles': self.toggles,
        }
    
    def run_cognitive_loop(
        self,
        iterations: Optional[int] = None,
        interval: float = 1.0
    ):
        """
        Run main cognitive processing loop
        """
        self.running = True
        self.iteration = 0
        
        logger.info("Starting cognitive loop...")
        
        try:
            while self.running:
                if iterations and self.iteration >= iterations:
                    break
                
                self.iteration += 1
                
                # Process data streams
                for stream_id, stream_info in self.data_streams.items():
                    try:
                        data = stream_info['fetch_fn']()
                        domain = stream_info['domain']
                        
                        outcome = data.pop('outcome', None)
                        
                        result = self.process_observation(data, domain, outcome)
                        
                    except Exception as e:
                        logger.error(f"Error processing stream {stream_id}: {e}")
                
                # Cognitive thinking every 10 iterations
                if self.iteration % 10 == 0:
                    thoughts = self.think({})
                    
                    if thoughts:
                        logger.info(f"Cognitive thoughts: {len(thoughts)} insights")
                
                # --- NEW: Learn rules every 20 iterations ---
                if self.iteration % 20 == 0:
                    self.learn_rules_from_history()
                
                # Generate autonomous goals every 50 iterations
                if self.iteration % 50 == 0:
                    self.generate_autonomous_goals()
                
                # Pursue goals every 25 iterations
                if self.iteration % 25 == 0:
                    self.pursue_goals()
                
                # Introspect every 100 iterations
                if self.iteration % 100 == 0:
                    introspection = self.introspect()
                    logger.info("="*60)
                    logger.info("INTROSPECTION")
                    logger.info("="*60)
                    logger.info(f"Concepts: {introspection['abstraction']['total_concepts']}")
                    logger.info(f"Rules: {introspection['reasoning']['total_rules']}")
                    logger.info(f"Goals: {introspection['goals']['total_goals']}")
                    logger.info(f"Transfers: {introspection['cognitive_metrics']['knowledge_transfers']}")
                    logger.info("="*60)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()
    
    def save_complete_state(self):
        """Save complete system state"""
        logger.info("Saving complete cognitive state...")
        
        try:
            self.learning_engine.save_state('/tmp/learning_state.pkl')
            self.abstraction.save_state('/tmp/abstraction_state.json')
            self.reasoning.save_state('/tmp/reasoning_state.json')
            self.cross_domain.save_state('/tmp/cross_domain_state.json')
            self.goals.save_state('/tmp/goals_state.json')
            
            with open('/tmp/cognitive_metrics.json', 'w') as f:
                json.dump(self.cognitive_metrics, f, indent=2)
            
            logger.info("Complete state saved")
        except Exception as e:
            logger.warning(f"State save error (non-fatal): {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down cognitive system...")
        
        self.running = False
        
        try:
            self.save_complete_state()
            self.orchestrator.stop_all()
        except Exception as e:
            logger.warning(f"Shutdown cleanup error: {e}")
        
        try:
            introspection = self.introspect()
            
            logger.info("="*60)
            logger.info("SHUTDOWN SUMMARY")
            logger.info("="*60)
            logger.info(f"Total iterations: {self.iteration}")
            logger.info(f"Concepts formed: {introspection['abstraction']['total_concepts']}")
            logger.info(f"Rules learned: {introspection['reasoning']['total_rules']}")
            logger.info(f"Goals achieved: {introspection['goals']['achieved_goals']}")
            logger.info(f"Knowledge transfers: {self.cognitive_metrics['knowledge_transfers']}")
            logger.info(f"Learning accuracy: {introspection['learning']['metrics']['accuracy']:.3f}")
            logger.info("="*60)
        except Exception as e:
            logger.warning(f"Shutdown summary error: {e}")
