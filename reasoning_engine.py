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

# Keep legacy alias
IF_THEN = RuleType.IMPLICATION


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


# Keys to use for rule mining (categorical/boolean, not raw floats)
_RULE_MINING_KEYS = {
    'direction', 'above_ma5', 'asset_type',
}

# Keys whose values should be discretized into buckets
_DISCRETIZE_KEYS = {
    'pct_change': [(-100, -5, 'big_drop'), (-5, -1, 'drop'), (-1, 1, 'flat'), (1, 5, 'rise'), (5, 100, 'big_rise')],
    'volatility_5': [(0, 1, 'low_vol'), (1, 5, 'med_vol'), (5, 100, 'high_vol')],
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
        
        # Dedup: track which rule signatures already exist
        self._rule_signatures: Set[str] = set()
        
        logger.info("Reasoning Engine initialized")
    
    def assert_fact(self, fact: str):
        """Add a fact to knowledge base"""
        self.facts.add(fact)
    
    def retract_fact(self, fact: str):
        """Remove a fact from knowledge base"""
        self.facts.discard(fact)
    
    def add_rule(
        self,
        antecedents: List[str],
        consequent: str,
        rule_type: RuleType = RuleType.IMPLICATION,
        confidence: float = 1.0
    ) -> Optional[str]:
        """Add reasoning rule (deduplicates by signature)"""
        # Create a canonical signature
        sig = "|".join(sorted(antecedents)) + "=>" + consequent
        
        # If rule already exists, update confidence if higher
        if sig in self._rule_signatures:
            for rule in self.rules.values():
                if "|".join(sorted(rule.antecedents)) + "=>" + rule.consequent == sig:
                    if confidence > rule.confidence:
                        rule.confidence = confidence
                        rule.support_count += 1
                    return rule.rule_id
            return None
        
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
        self._rule_signatures.add(sig)
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
    
    def _discretize_observation(self, obs: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert a raw observation into a set of categorical key=value items
        suitable for association rule mining.
        """
        items = {}
        
        for key in _RULE_MINING_KEYS:
            if key in obs:
                val = obs[key]
                if isinstance(val, bool):
                    items[key] = str(val)
                else:
                    items[key] = str(val)
        
        for key, buckets in _DISCRETIZE_KEYS.items():
            if key in obs:
                val = obs[key]
                if isinstance(val, (int, float)):
                    for lo, hi, label in buckets:
                        if lo <= val < hi:
                            items[key] = label
                            break
        
        return items
    
    def learn_rules_from_observations(
        self,
        observations: List[Dict[str, Any]],
        min_support: int = 3
    ):
        """
        Learn rules from observation data using association rule mining.
        
        This version discretizes continuous values into categorical buckets
        so that frequent itemsets can actually be found.
        
        Example rules learned:
          IF asset_type=crypto AND direction=up THEN above_ma5=True
          IF pct_change=big_rise THEN direction=up
          IF volatility_5=high_vol AND asset_type=crypto THEN pct_change=big_rise
        """
        if len(observations) < min_support:
            return
        
        # Discretize all observations
        discretized = []
        for obs in observations:
            d = self._discretize_observation(obs)
            if d:
                discretized.append(d)
        
        if len(discretized) < min_support:
            return
        
        # Collect all unique items (key=value pairs)
        all_items: Set[str] = set()
        for d in discretized:
            for k, v in d.items():
                all_items.add(f"{k}={v}")
        
        # Find frequent single items
        item_support: Dict[str, int] = defaultdict(int)
        for d in discretized:
            items_in_obs = {f"{k}={v}" for k, v in d.items()}
            for item in items_in_obs:
                item_support[item] += 1
        
        frequent_1 = [item for item, count in item_support.items() if count >= min_support]
        
        if not frequent_1:
            return
        
        # Find frequent pairs
        frequent_pairs = []
        for i in range(len(frequent_1)):
            for j in range(i + 1, len(frequent_1)):
                a, b = frequent_1[i], frequent_1[j]
                # Don't pair items from the same key
                key_a = a.split('=')[0]
                key_b = b.split('=')[0]
                if key_a == key_b:
                    continue
                
                pair_support = 0
                for d in discretized:
                    items_in_obs = {f"{k}={v}" for k, v in d.items()}
                    if a in items_in_obs and b in items_in_obs:
                        pair_support += 1
                
                if pair_support >= min_support:
                    frequent_pairs.append((a, b, pair_support))
        
        # Generate rules from frequent pairs
        rules_added = 0
        for a, b, support in frequent_pairs:
            # Rule: a => b
            a_count = item_support[a]
            conf_ab = support / a_count if a_count > 0 else 0
            if conf_ab >= self.min_confidence:
                rid = self.add_rule([a], b, RuleType.IMPLICATION, round(conf_ab, 3))
                if rid:
                    rules_added += 1
            
            # Rule: b => a
            b_count = item_support[b]
            conf_ba = support / b_count if b_count > 0 else 0
            if conf_ba >= self.min_confidence:
                rid = self.add_rule([b], a, RuleType.IMPLICATION, round(conf_ba, 3))
                if rid:
                    rules_added += 1
        
        # Find frequent triples (limited)
        if len(frequent_pairs) >= 2:
            frequent_triple_items = set()
            for a, b, _ in frequent_pairs:
                frequent_triple_items.add(a)
                frequent_triple_items.add(b)
            
            triple_items = list(frequent_triple_items)
            triples_checked = 0
            max_triples = 50
            
            for i in range(len(triple_items)):
                if triples_checked >= max_triples:
                    break
                for j in range(i + 1, len(triple_items)):
                    if triples_checked >= max_triples:
                        break
                    for k in range(j + 1, len(triple_items)):
                        if triples_checked >= max_triples:
                            break
                        
                        a, b, c = triple_items[i], triple_items[j], triple_items[k]
                        keys = {x.split('=')[0] for x in [a, b, c]}
                        if len(keys) < 3:
                            continue
                        
                        triples_checked += 1
                        
                        triple_support = 0
                        for d in discretized:
                            items_in_obs = {f"{kk}={vv}" for kk, vv in d.items()}
                            if a in items_in_obs and b in items_in_obs and c in items_in_obs:
                                triple_support += 1
                        
                        if triple_support >= min_support:
                            # Generate rules: (a,b) => c, (a,c) => b, (b,c) => a
                            for ant_pair, cons in [([a, b], c), ([a, c], b), ([b, c], a)]:
                                ant_support = 0
                                for d in discretized:
                                    items_in_obs = {f"{kk}={vv}" for kk, vv in d.items()}
                                    if all(x in items_in_obs for x in ant_pair):
                                        ant_support += 1
                                
                                conf = triple_support / ant_support if ant_support > 0 else 0
                                if conf >= self.min_confidence:
                                    rid = self.add_rule(ant_pair, cons, RuleType.IMPLICATION, round(conf, 3))
                                    if rid:
                                        rules_added += 1
        
        if rules_added > 0:
            logger.info(f"Rule mining: learned {rules_added} rules from {len(discretized)} observations")
        
        return rules_added
    
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
                cause_series = [d.get(cause_var, 0) for d in time_series_data]
                effect_series = [d.get(effect_var, 0) for d in time_series_data]
                
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
        
        try:
            return float(np.corrcoef(cause_lagged, effect_current)[0, 1])
        except Exception:
            return 0.0
    
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
            sig = "|".join(sorted(self.rules[rid].antecedents)) + "=>" + self.rules[rid].consequent
            self._rule_signatures.discard(sig)
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
            'avg_rule_confidence': float(np.mean([r.confidence for r in self.rules.values()]))
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
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Reasoning state saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save reasoning state: {e}")
