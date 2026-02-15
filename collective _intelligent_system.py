"""
Advanced Cognitive Intelligent System
Integrates abstraction, reasoning, cross-domain transfer, and goal formation
with continuous learning, self-evolution, and always-on operation
"""
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
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
            min_mapping_confidence=0.6,
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
        
        # Performance tracking
        self.cognitive_metrics = {
            'concepts_formed': 0,
            'rules_learned': 0,
            'analogies_found': 0,
            'goals_achieved': 0,
            'knowledge_transfers': 0
        }
        
        logger.info("System initialized with all cognitive capabilities")
    
    def register_data_stream(self, stream_id: str, fetch_fn, domain: str):
        """Register data stream with domain association"""
        self.data_streams[stream_id] = {
            'fetch_fn': fetch_fn,
            'domain': domain
        }
        
        # Register domain if new
        if domain not in self.cross_domain.domains:
            self.cross_domain.register_domain(domain, domain)
            logger.info(f"Registered new domain: {domain}")
        
        logger.info(f"Registered stream: {stream_id} in domain {domain}")
    
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
            
            # Add to domain
            self.cross_domain.add_concept_to_domain(domain, concept_id)
            self.active_concepts.add(concept_id)
        
        # 3. Extract facts for reasoning
        facts = self._extract_facts(observation)
        for fact in facts:
            self.reasoning.assert_fact(fact)
        
        # 4. Attempt inference
        inferred = self.reasoning.infer(max_steps=3)
        if inferred:
            results['inferred_facts'] = list(inferred)
        
        # 5. Look for cross-domain patterns
        if len(self.cross_domain.domains) > 1:
            self._check_cross_domain_opportunities(domain)
        
        return results
    
    def _extract_facts(self, observation: Dict[str, Any]) -> List[str]:
        """Extract logical facts from observation"""
        facts = []
        
        for key, value in observation.items():
            if isinstance(value, bool):
                if value:
                    facts.append(key)
            elif isinstance(value, (int, float)):
                # Create threshold-based facts
                if value > 0:
                    facts.append(f"{key}_positive")
                if value > 0.5:
                    facts.append(f"{key}_high")
        
        return facts
    
    def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform higher-level cognitive processing
        - Abstract thinking
        - Analogical reasoning
        - Goal-directed planning
        """
        thoughts = {}
        
        # Abstract reasoning
        if len(self.active_concepts) > 0:
            # Find analogies between recent concepts
            recent_concepts = list(self.active_concepts)[-5:]
            
            if len(recent_concepts) >= 2:
                analogies = self.abstraction.find_analogies(recent_concepts[0])
                if analogies:
                    thoughts['analogies'] = [a.to_dict() for a in analogies[:3]]
                    self.cognitive_metrics['analogies_found'] += len(analogies)
        
        # Logical reasoning
        if self.reasoning.facts:
            # Try to explain recent facts
            recent_facts = list(self.reasoning.facts)[-5:]
            thoughts['explanations'] = {}
            
            for fact in recent_facts:
                explanation = self.reasoning.explain(fact)
                if explanation:
                    thoughts['explanations'][fact] = explanation
        
        # Goal-oriented thinking
        if len(self.goals.goals) > 0:
            active_goals = [
                self.goals.goals[gid]
                for gid in self.goals.active_goals
            ]
            thoughts['active_goals'] = [g.to_dict() for g in active_goals]
        
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
            patterns=[p.to_dict() for p in self.abstraction.patterns.values()],
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
        """Actively pursue current goals"""
        
        # Select which goals to work on
        active_goal_ids = self.goals.select_active_goals()
        
        results = {}
        
        for goal_id in active_goal_ids:
            goal = self.goals.goals[goal_id]
            
            # Create plan for goal
            plan = self.reasoning.create_plan(
                goal=goal.description,
                current_state={},
                available_actions=[]
            )
            
            if plan:
                # Execute plan
                execution_result = self.reasoning.execute_plan(plan)
                results[goal_id] = execution_result
                
                # Update goal progress
                self.goals.update_goal_progress(
                    goal_id,
                    {'achieved': execution_result['success']}
                )
                
                if execution_result['success']:
                    self.cognitive_metrics['goals_achieved'] += 1
        
        return results
    
    def _check_cross_domain_opportunities(self, current_domain: str):
        """Check for knowledge transfer opportunities"""
        
        # Get domains with sufficient concepts
        eligible_domains = [
            d for d in self.cross_domain.domains.values()
            if len(d.concepts) >= 3
        ]
        
        if len(eligible_domains) < 2:
            return
        
        # Try to discover mappings
        for other_domain in eligible_domains:
            if other_domain.domain_id != current_domain:
                
                # Check if mapping exists
                existing_mapping = any(
                    m.source_domain == current_domain and m.target_domain == other_domain.domain_id
                    for m in self.cross_domain.mappings.values()
                )
                
                if not existing_mapping:
                    mapping = self.cross_domain.discover_domain_mapping(
                        current_domain,
                        other_domain.domain_id
                    )
                    
                    if mapping:
                        logger.info(
                            f"Discovered cross-domain mapping: "
                            f"{current_domain} -> {other_domain.domain_id}"
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
            # Transfer patterns
            patterns = [
                p for p in self.abstraction.patterns.values()
                if p.concept_id in self.cross_domain.domains.get(from_domain, Domain()).concepts
            ]
            
            for pattern in patterns[:3]:  # Transfer top 3
                transfer = self.cross_domain.transfer_knowledge(
                    knowledge=pattern.to_dict(),
                    knowledge_type='pattern',
                    source_domain_id=from_domain,
                    target_domain_id=to_domain
                )
                
                if transfer:
                    self.cognitive_metrics['knowledge_transfers'] += 1
                    logger.info(f"Transferred pattern from {from_domain} to {to_domain}")
        
        elif knowledge_type == 'rule':
            # Transfer rules
            # Find rules relevant to source domain
            relevant_rules = [
                r for r in self.reasoning.rules.values()
                if r.confidence > 0.7
            ]
            
            for rule in relevant_rules[:3]:
                transfer = self.cross_domain.transfer_knowledge(
                    knowledge=rule.to_dict(),
                    knowledge_type='rule',
                    source_domain_id=from_domain,
                    target_domain_id=to_domain
                )
                
                if transfer:
                    self.cognitive_metrics['knowledge_transfers'] += 1
                    logger.info(f"Transferred rule from {from_domain} to {to_domain}")
    
    def introspect(self) -> Dict[str, Any]:
        """Perform system introspection"""
        
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
            'active_domains': len(self.cross_domain.domains)
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
        
        # Save each component
        self.learning_engine.save_state('/home/claude/learning_state.pkl')
        self.abstraction.save_state('/home/claude/abstraction_state.json')
        self.reasoning.save_state('/home/claude/reasoning_state.json')
        self.cross_domain.save_state('/home/claude/cross_domain_state.json')
        self.goals.save_state('/home/claude/goals_state.json')
        
        # Save cognitive metrics
        with open('/home/claude/cognitive_metrics.json', 'w') as f:
            json.dump(self.cognitive_metrics, f, indent=2)
        
        logger.info("Complete state saved")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down cognitive system...")
        
        self.running = False
        
        # Save everything
        self.save_complete_state()
        
        # Stop orchestrator
        self.orchestrator.stop_all()
        
        # Print summary
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


if __name__ == "__main__":
    import numpy as np
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system
    system = CognitiveIntelligentSystem(
        system_id="demo-cognitive-system",
        feature_dim=20,
        learning_rate=0.01
    )
    
    # Example data streams from different domains
    def physics_stream():
        return {
            'force': np.random.randn() * 10,
            'mass': np.random.randn() * 5 + 10,
            'acceleration': np.random.randn() * 2,
            'outcome': np.random.randn()
        }
    
    def economics_stream():
        return {
            'price': np.random.randn() * 100 + 500,
            'supply': np.random.randn() * 50 + 100,
            'demand': np.random.randn() * 50 + 100,
            'outcome': np.random.randn()
        }
    
    # Register streams
    system.register_data_stream('physics', physics_stream, 'physics')
    system.register_data_stream('economics', economics_stream, 'economics')
    
    # Set an explicit goal
    system.set_goal(
        "Discover analogies between physics and economics",
        GoalType.UNDERSTANDING,
        {'analogies_found': 3, 'confidence': 0.7}
    )
    
    # Run!
    print("="*60)
    print("COGNITIVE INTELLIGENT SYSTEM")
    print("="*60)
    print("Capabilities:")
    print("  ✓ Continuous Learning")
    print("  ✓ Abstraction & Concept Formation")
    print("  ✓ Logical Reasoning & Planning")
    print("  ✓ Cross-Domain Knowledge Transfer")
    print("  ✓ Autonomous Goal Formation")
    print("  ✓ Self-Evolution")
    print("  ✓ Always-On Operation")
    print("="*60)
    print("\nStarting... (Press Ctrl+C to stop)\n")
    
    system.run_cognitive_loop(iterations=500, interval=0.5)