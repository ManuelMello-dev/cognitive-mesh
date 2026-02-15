"""
Reasoning Engine
Logical inference, causal reasoning, and planning capabilities
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of reasoning rules"""
    IMPLICATION = "if-then"
    EQUIVALENCE = "iff"
    CAUSAL = "causes"
    CORRELATION = "correlates"
    CONSTRAINT = "must"


@dataclass
class Rule:
    """Reasoning rule representation"""
    rule_id: str
    rule_type: RuleType
    antecedents: List[str]  # Conditions
    consequent: str  # Conclusion
    confidence: float
    support_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'rule_id': self.rule_id,
            'type': self.rule_type.value,
            'if': self.antecedents,
            'then': self.consequent,
            'confidence': self.confidence,
            'support': self.support_count,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        ant_str = ' AND '.join(self.antecedents)
        return f"IF {ant_str} THEN {self.consequent} (conf: {self.confidence:.2f})"


@dataclass
class CausalLink:
    """Causal relationship between variables"""
    cause: str
    effect: str
    strength: float  # -1 to 1 (negative to positive causation)
    confidence: float
    observations: int = 0
    
    def to_dict(self):
        return {
            'cause': self.cause,
            'effect': self.effect,
            'strength': self.strength,
            'confidence': self.confidence,
            'observations': self.observations
        }


@dataclass
class Plan:
    """Plan representation"""
    plan_id: str
    goal: str
    steps: List[Dict[str, Any]]
    expected_outcome: Any
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'plan_id': self.plan_id,
            'goal': self.goal,
            'steps': self.steps,
            'expected_outcome': self.expected_outcome,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }


class ReasoningEngine:
    """
    Performs logical inference, causal reasoning, and planning
    """
    
    def __init__(
        self,
        max_rules: int = 1000,
        min_confidence: float = 0.6,
        max_inference_depth: int = 5
    ):
        self.max_rules = max_rules
        self.min_confidence = min_confidence
        self.max_depth = max_inference_depth
        
        # Knowledge base
        self.rules: Dict[str, Rule] = {}
        self.facts: Set[str] = set()
        self.causal_graph: Dict[str, List[CausalLink]] = defaultdict(list)
        
        # Planning
        self.plans: Dict[str, Plan] = {}
        self.plan_counter = 0
        
        # Learning
        self.observation_history: List[Dict[str, Any]] = []
        self.rule_counter = 0
        
        logger.info("Reasoning Engine initialized")
    
    def assert_fact(self, fact: str):
        """Add a fact to knowledge base"""
        self.facts.add(fact)
        logger.debug(f"Asserted fact: {fact}")
    
    def retract_fact(self, fact: str):
        """Remove a fact from knowledge base"""
        self.facts.discard(fact)
        logger.debug(f"Retracted fact: {fact}")
    
    def add_rule(
        self,
        antecedents: List[str],
        consequent: str,
        rule_type: RuleType = RuleType.IMPLICATION,
        confidence: float = 1.0
    ) -> str:
        """Add reasoning rule"""
        if len(self.rules) >= self.max_rules:
            self._prune_weak_rules()
        
        rule_id = f"rule_{self.rule_counter}"
        self.rule_counter += 1
        
        rule = Rule(
            rule_id=rule_id,
            rule_type=rule_type,
            antecedents=antecedents,
            consequent=consequent,
            confidence=confidence
        )
        
        self.rules[rule_id] = rule
        logger.info(f"Added rule: {rule}")
        
        return rule_id
    
    def infer(self, max_steps: int = 10) -> Set[str]:
        """
        Perform forward chaining inference
        Returns newly inferred facts
        """
        inferred_facts = set()
        steps = 0
        
        while steps < max_steps:
            new_facts = set()
            
            for rule_id, rule in self.rules.items():
                if rule.confidence < self.min_confidence:
                    continue
                
                # Check if all antecedents are satisfied
                if all(ant in self.facts for ant in rule.antecedents):
                    if rule.consequent not in self.facts:
                        new_facts.add(rule.consequent)
                        rule.support_count += 1
                        logger.debug(f"Inferred: {rule.consequent} from {rule_id}")
            
            if not new_facts:
                break  # No new inferences
            
            self.facts.update(new_facts)
            inferred_facts.update(new_facts)
            steps += 1
        
        return inferred_facts
    
    def backward_chain(self, goal: str, depth: int = 0) -> bool:
        """
        Backward chaining to prove a goal
        Returns True if goal can be proven
        """
        if depth > self.max_depth:
            return False
        
        # Already a fact
        if goal in self.facts:
            return True
        
        # Try to prove via rules
        for rule in self.rules.values():
            if rule.consequent == goal and rule.confidence >= self.min_confidence:
                # Try to prove all antecedents
                if all(self.backward_chain(ant, depth + 1) for ant in rule.antecedents):
                    self.assert_fact(goal)
                    return True
        
        return False
    
    def explain(self, fact: str) -> List[str]:
        """
        Explain how a fact was derived
        Returns chain of reasoning
        """
        if fact not in self.facts:
            return []
        
        explanation = []
        
        # Find rules that conclude this fact
        for rule in self.rules.values():
            if rule.consequent == fact:
                explanation.append(str(rule))
                
                # Recursively explain antecedents
                for ant in rule.antecedents:
                    sub_explanation = self.explain(ant)
                    explanation.extend(sub_explanation)
        
        return explanation
    
    def learn_rules_from_observations(
        self,
        observations: List[Dict[str, Any]],
        min_support: int = 3
    ):
        """
        Learn rules from observation data
        Uses association rule mining approach
        """
        if len(observations) < min_support:
            return
        
        # Extract all attributes and values
        all_conditions = set()
        for obs in observations:
            for key, value in obs.items():
                all_conditions.add(f"{key}={value}")
        
        # Find frequent itemsets
        frequent_sets = self._find_frequent_itemsets(
            observations,
            all_conditions,
            min_support
        )
        
        # Generate rules from frequent sets
        for itemset in frequent_sets:
            if len(itemset) < 2:
                continue
            
            # Try different antecedent/consequent splits
            for i in range(1, len(itemset)):
                antecedents = list(itemset[:i])
                consequent = itemset[i]
                
                # Calculate confidence
                confidence = self._calculate_rule_confidence(
                    observations,
                    antecedents,
                    consequent
                )
                
                if confidence >= self.min_confidence:
                    self.add_rule(
                        antecedents,
                        consequent,
                        RuleType.IMPLICATION,
                        confidence
                    )
    
    def _find_frequent_itemsets(
        self,
        observations: List[Dict[str, Any]],
        conditions: Set[str],
        min_support: int
    ) -> List[Tuple[str, ...]]:
        """Find frequent itemsets using Apriori algorithm"""
        frequent = []
        
        # Single items
        for cond in conditions:
            support = sum(
                1 for obs in observations
                if self._condition_matches(cond, obs)
            )
            
            if support >= min_support:
                frequent.append((cond,))
        
        # Grow itemsets
        k = 1
        while k < 3:  # Limit to pairs/triples
            k += 1
            candidates = []
            
            # Generate candidates
            for i, set1 in enumerate(frequent):
                if len(set1) != k - 1:
                    continue
                    
                for set2 in frequent[i+1:]:
                    if len(set2) != k - 1:
                        continue
                    
                    # Merge if they share k-2 elements
                    combined = tuple(sorted(set(set1) | set(set2)))
                    if len(combined) == k and combined not in candidates:
                        candidates.append(combined)
            
            # Check support for candidates
            for candidate in candidates:
                support = sum(
                    1 for obs in observations
                    if all(self._condition_matches(cond, obs) for cond in candidate)
                )
                
                if support >= min_support:
                    frequent.append(candidate)
        
        return frequent
    
    def _condition_matches(self, condition: str, observation: Dict[str, Any]) -> bool:
        """Check if condition matches observation"""
        if '=' not in condition:
            return False
        
        key, value = condition.split('=', 1)
        
        # Convert value to appropriate type
        if key in observation:
            obs_value = str(observation[key])
            return obs_value == value
        
        return False
    
    def _calculate_rule_confidence(
        self,
        observations: List[Dict[str, Any]],
        antecedents: List[str],
        consequent: str
    ) -> float:
        """Calculate confidence of a rule"""
        antecedent_count = sum(
            1 for obs in observations
            if all(self._condition_matches(ant, obs) for ant in antecedents)
        )
        
        if antecedent_count == 0:
            return 0.0
        
        both_count = sum(
            1 for obs in observations
            if all(self._condition_matches(ant, obs) for ant in antecedents)
            and self._condition_matches(consequent, obs)
        )
        
        return both_count / antecedent_count
    
    def discover_causal_relationships(
        self,
        time_series_data: List[Dict[str, float]],
        lag: int = 1
    ):
        """
        Discover causal relationships using Granger causality approach
        """
        if len(time_series_data) < 10:
            return
        
        # Extract variables
        variables = list(time_series_data[0].keys())
        
        # Test each pair for causality
        for cause_var in variables:
            for effect_var in variables:
                if cause_var == effect_var:
                    continue
                
                # Extract time series
                cause_series = [d[cause_var] for d in time_series_data]
                effect_series = [d[effect_var] for d in time_series_data]
                
                # Simple correlation with lag
                strength = self._test_lagged_correlation(
                    cause_series,
                    effect_series,
                    lag
                )
                
                if abs(strength) > 0.3:  # Threshold
                    link = CausalLink(
                        cause=cause_var,
                        effect=effect_var,
                        strength=strength,
                        confidence=min(abs(strength), 1.0),
                        observations=len(time_series_data)
                    )
                    
                    self.causal_graph[cause_var].append(link)
                    logger.info(f"Discovered causal link: {cause_var} -> {effect_var} ({strength:.2f})")
    
    def _test_lagged_correlation(
        self,
        cause: List[float],
        effect: List[float],
        lag: int
    ) -> float:
        """Test lagged correlation between series"""
        if len(cause) <= lag or len(effect) <= lag:
            return 0.0
        
        # Align series with lag
        cause_lagged = cause[:-lag]
        effect_current = effect[lag:]
        
        # Calculate correlation
        if len(cause_lagged) < 2:
            return 0.0
        
        return np.corrcoef(cause_lagged, effect_current)[0, 1]
    
    def predict_effect(self, cause_var: str, cause_value: float) -> Dict[str, float]:
        """Predict effects of setting a variable to a value"""
        predictions = {}
        
        if cause_var not in self.causal_graph:
            return predictions
        
        for link in self.causal_graph[cause_var]:
            # Simple linear prediction
            predicted_effect = cause_value * link.strength
            predictions[link.effect] = predicted_effect
        
        return predictions
    
    def create_plan(
        self,
        goal: str,
        current_state: Dict[str, Any],
        available_actions: List[Callable]
    ) -> Optional[Plan]:
        """
        Create plan to achieve goal using backward planning
        """
        # Simple planning: try to find path from current to goal
        steps = []
        confidence = 1.0
        
        # Check if goal is already satisfied
        if self.backward_chain(goal):
            return Plan(
                plan_id=f"plan_{self.plan_counter}",
                goal=goal,
                steps=[],
                expected_outcome=goal,
                confidence=1.0
            )
        
        # Find rules that can achieve goal
        relevant_rules = [
            r for r in self.rules.values()
            if r.consequent == goal and r.confidence >= self.min_confidence
        ]
        
        if not relevant_rules:
            return None
        
        # Use highest confidence rule
        best_rule = max(relevant_rules, key=lambda r: r.confidence)
        confidence = best_rule.confidence
        
        # Create steps to satisfy antecedents
        for antecedent in best_rule.antecedents:
            steps.append({
                'action': 'satisfy',
                'condition': antecedent,
                'method': 'inference'
            })
        
        # Final step: apply rule
        steps.append({
            'action': 'apply_rule',
            'rule_id': best_rule.rule_id,
            'result': goal
        })
        
        self.plan_counter += 1
        plan = Plan(
            plan_id=f"plan_{self.plan_counter}",
            goal=goal,
            steps=steps,
            expected_outcome=goal,
            confidence=confidence
        )
        
        self.plans[plan.plan_id] = plan
        logger.info(f"Created plan for goal '{goal}' with {len(steps)} steps")
        
        return plan
    
    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Execute a plan and return results"""
        results = {
            'plan_id': plan.plan_id,
            'goal': plan.goal,
            'steps_executed': 0,
            'success': False,
            'outcomes': []
        }
        
        for i, step in enumerate(plan.steps):
            try:
                if step['action'] == 'satisfy':
                    # Try to satisfy condition
                    condition = step['condition']
                    self.assert_fact(condition)
                    results['outcomes'].append(f"Satisfied: {condition}")
                
                elif step['action'] == 'apply_rule':
                    # Apply inference rule
                    rule_id = step['rule_id']
                    if rule_id in self.rules:
                        self.infer(max_steps=1)
                        results['outcomes'].append(f"Applied rule: {rule_id}")
                
                results['steps_executed'] += 1
                
            except Exception as e:
                logger.error(f"Error executing plan step {i}: {e}")
                break
        
        # Check if goal achieved
        results['success'] = plan.goal in self.facts
        
        return results
    
    def _prune_weak_rules(self):
        """Remove low-confidence, low-support rules"""
        to_remove = [
            rid for rid, rule in self.rules.items()
            if rule.confidence < self.min_confidence and rule.support_count < 2
        ]
        
        for rid in to_remove:
            del self.rules[rid]
        
        if to_remove:
            logger.info(f"Pruned {len(to_remove)} weak rules")
    
    def get_insights(self) -> Dict[str, Any]:
        """Get reasoning engine insights"""
        return {
            'total_facts': len(self.facts),
            'total_rules': len(self.rules),
            'causal_links': sum(len(links) for links in self.causal_graph.values()),
            'plans_created': len(self.plans),
            'avg_rule_confidence': np.mean([r.confidence for r in self.rules.values()])
            if self.rules else 0
        }
    
    def save_state(self, filepath: str):
        """Save reasoning state"""
        state = {
            'facts': list(self.facts),
            'rules': {k: v.to_dict() for k, v in self.rules.items()},
            'causal_graph': {
                k: [link.to_dict() for link in v]
                for k, v in self.causal_graph.items()
            },
            'plans': {k: v.to_dict() for k, v in self.plans.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Reasoning state saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo
    engine = ReasoningEngine()
    
    print("=== Logical Inference Demo ===\n")
    
    # Add facts
    engine.assert_fact("raining")
    engine.assert_fact("cloudy")
    
    # Add rules
    engine.add_rule(["raining"], "ground_wet", confidence=0.95)
    engine.add_rule(["ground_wet", "cold"], "icy", confidence=0.8)
    engine.add_rule(["cloudy", "cold"], "might_snow", confidence=0.6)
    
    # Infer
    print("Initial facts:", engine.facts)
    new_facts = engine.infer()
    print("Inferred facts:", new_facts)
    print("All facts now:", engine.facts)
    
    # Explain
    print("\nExplanation for 'ground_wet':")
    explanation = engine.explain("ground_wet")
    for line in explanation:
        print(f"  {line}")
    
    print("\n=== Causal Discovery Demo ===\n")
    
    # Generate time series data
    time_series = []
    for t in range(50):
        x = np.sin(t / 10)
        y = x * 0.8 + np.random.randn() * 0.1  # y caused by x
        z = np.random.randn()  # independent
        time_series.append({'x': x, 'y': y, 'z': z})
    
    engine.discover_causal_relationships(time_series, lag=1)
    
    print("Causal graph:")
    for cause, links in engine.causal_graph.items():
        for link in links:
            print(f"  {cause} -> {link.effect} (strength: {link.strength:.2f})")
    
    print("\n=== Plann
