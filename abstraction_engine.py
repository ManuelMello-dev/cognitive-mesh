"""
Abstraction Engine
Forms abstract concepts and hierarchical representations from concrete observations
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """Abstract concept representation"""
    concept_id: str
    name: str
    level: int  # Abstraction level (0=concrete, higher=more abstract)
    attributes: Dict[str, Any]
    examples: List[Dict[str, Any]]
    parent_concepts: Set[str] = field(default_factory=set)
    child_concepts: Set[str] = field(default_factory=set)
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'concept_id': self.concept_id,
            'name': self.name,
            'level': self.level,
            'attributes': self.attributes,
            'example_count': len(self.examples),
            'parents': list(self.parent_concepts),
            'children': list(self.child_concepts),
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class Analogy:
    """Structural analogy between concepts"""
    source_concept: str
    target_concept: str
    mappings: Dict[str, str]  # Attribute mappings
    similarity_score: float
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'source': self.source_concept,
            'target': self.target_concept,
            'mappings': self.mappings,
            'similarity': self.similarity_score,
            'discovered_at': self.discovered_at.isoformat()
        }


class AbstractionEngine:
    """
    Forms abstract concepts from concrete observations
    Builds hierarchical concept representations
    """
    
    def __init__(
        self,
        max_concepts: int = 1000,
        min_examples_for_concept: int = 3,
        similarity_threshold: float = 0.7
    ):
        self.max_concepts = max_concepts
        self.min_examples = min_examples_for_concept
        self.similarity_threshold = similarity_threshold
        
        # Concept hierarchy
        self.concepts: Dict[str, Concept] = {}
        self.concept_counter = 0
        
        # Analogies between concepts
        self.analogies: List[Analogy] = []
        
        # Observation buffer for concept formation
        self.observation_buffer: List[Dict[str, Any]] = []
        
        logger.info("Abstraction Engine initialized")

    @property
    def patterns(self) -> Dict[str, Dict]:
        """Derive patterns from formed concepts for goal generation context."""
        result = {}
        for cid, concept in self.concepts.items():
            result[cid] = {
                'pattern_id': cid,
                'name': concept.name,
                'confidence': concept.confidence,
                'level': concept.level,
                'examples_count': len(concept.examples),
                'attributes': dict(concept.attributes) if concept.attributes else {},
            }
        return result

    def observe(self, observation: Dict[str, Any]) -> Optional[str]:
        """
        Process observation and potentially form concepts
        Returns concept_id if observation matches or creates a concept
        """
        self.observation_buffer.append(observation)
        
        # Try to match existing concept
        matched_concept = self._match_concept(observation)
        
        if matched_concept:
            # Add to concept's examples
            self.concepts[matched_concept].examples.append(observation)
            self.concepts[matched_concept].confidence = min(
                1.0,
                self.concepts[matched_concept].confidence + 0.05
            )
            return matched_concept
        
        # Check if we can form a new concept
        if len(self.observation_buffer) >= self.min_examples:
            new_concept = self._form_concept()
            if new_concept:
                return new_concept
        
        return None
    
    def _match_concept(self, observation: Dict[str, Any]) -> Optional[str]:
        """Match observation to existing concept"""
        best_match = None
        best_score = 0
        
        for concept_id, concept in self.concepts.items():
            if concept.level > 0:  # Skip abstract concepts for now
                continue
                
            score = self._similarity_score(observation, concept.attributes)
            
            if score > self.similarity_threshold and score > best_score:
                best_score = score
                best_match = concept_id
        
        return best_match
    
    def _form_concept(self) -> Optional[str]:
        """Form new concept from observation buffer"""
        if len(self.concepts) >= self.max_concepts:
            return None
        
        # Extract common attributes
        attributes = self._extract_common_attributes(self.observation_buffer[-5:])
        
        if not attributes:
            return None
        
        # Create concept
        concept_id = f"concept_{self.concept_counter}"
        self.concept_counter += 1
        
        concept = Concept(
            concept_id=concept_id,
            name=self._generate_concept_name(attributes),
            level=0,  # Concrete level
            attributes=attributes,
            examples=self.observation_buffer[-5:].copy()
        )
        
        self.concepts[concept_id] = concept
        logger.info(f"Formed new concept: {concept.name} (id: {concept_id})")
        
        # Try to form higher-level abstractions
        self._attempt_abstraction()
        
        return concept_id
    
    def _extract_common_attributes(
        self, 
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract attributes common across observations"""
        if not observations:
            return {}
        
        # Find common keys
        common_keys = set(observations[0].keys())
        for obs in observations[1:]:
            common_keys &= set(obs.keys())
        
        attributes = {}
        
        for key in common_keys:
            values = [obs[key] for obs in observations]
            
            # Handle different value types
            if all(isinstance(v, (int, float)) for v in values):
                # Numerical: use range/mean
                attributes[key] = {
                    'type': 'numerical',
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': (min(values), max(values))
                }
            elif all(isinstance(v, bool) for v in values):
                # Boolean: use majority
                attributes[key] = {
                    'type': 'boolean',
                    'majority': Counter(values).most_common(1)[0][0]
                }
            elif all(isinstance(v, str) for v in values):
                # Categorical: use common values
                counter = Counter(values)
                if len(counter) <= 3:  # Limited categories
                    attributes[key] = {
                        'type': 'categorical',
                        'values': list(counter.keys())
                    }
        
        return attributes
    
    def _similarity_score(
        self, 
        observation: Dict[str, Any], 
        attributes: Dict[str, Any]
    ) -> float:
        """Calculate similarity between observation and concept attributes"""
        if not attributes:
            return 0.0
        
        scores = []
        
        for key, attr_def in attributes.items():
            if key not in observation:
                scores.append(0.0)
                continue
            
            value = observation[key]
            
            if attr_def['type'] == 'numerical':
                # Check if within range
                mean = attr_def['mean']
                std = attr_def['std']
                if std > 0:
                    z_score = abs((value - mean) / std)
                    scores.append(max(0, 1 - z_score / 3))  # 3-sigma rule
                else:
                    scores.append(1.0 if value == mean else 0.0)
            
            elif attr_def['type'] == 'boolean':
                scores.append(1.0 if value == attr_def['majority'] else 0.0)
            
            elif attr_def['type'] == 'categorical':
                scores.append(1.0 if value in attr_def['values'] else 0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_concept_name(self, attributes: Dict[str, Any]) -> str:
        """Generate human-readable concept name"""
        # Simple naming: use key attributes
        key_attrs = list(attributes.keys())[:3]
        return f"Concept_{'-'.join(key_attrs)}"
    
    def _attempt_abstraction(self):
        """Attempt to form higher-level abstract concepts"""
        concrete_concepts = [
            (cid, c) for cid, c in self.concepts.items() if c.level == 0
        ]
        
        # Look for similar concepts to group
        for i, (cid1, concept1) in enumerate(concrete_concepts):
            for cid2, concept2 in concrete_concepts[i+1:]:
                similarity = self._concept_similarity(concept1, concept2)
                
                if similarity > self.similarity_threshold:
                    # Form abstract concept
                    self._form_abstract_concept([concept1, concept2])
    
    def _concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Calculate similarity between two concepts"""
        # Compare attribute structures
        keys1 = set(concept1.attributes.keys())
        keys2 = set(concept2.attributes.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        # Jaccard similarity of keys
        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)
        
        return intersection / union if union > 0 else 0.0
    
    def _form_abstract_concept(self, child_concepts: List[Concept]) -> str:
        """Form abstract concept from similar child concepts"""
        if len(self.concepts) >= self.max_concepts:
            return None
        
        # Merge attributes
        merged_attributes = {}
        all_keys = set()
        
        for concept in child_concepts:
            all_keys.update(concept.attributes.keys())
        
        for key in all_keys:
            # Only include if present in all children
            if all(key in c.attributes for c in child_concepts):
                # Use first concept's attribute as template
                merged_attributes[key] = child_concepts[0].attributes[key].copy()
        
        # Create abstract concept
        concept_id = f"concept_{self.concept_counter}"
        self.concept_counter += 1
        
        abstract_concept = Concept(
            concept_id=concept_id,
            name=f"Abstract_{len([c for c in self.concepts.values() if c.level > 0])}",
            level=max(c.level for c in child_concepts) + 1,
            attributes=merged_attributes,
            examples=[],  # Abstract concepts don't have direct examples
            child_concepts={c.concept_id for c in child_concepts}
        )
        
        # Link children to parent
        for child in child_concepts:
            child.parent_concepts.add(concept_id)
        
        self.concepts[concept_id] = abstract_concept
        logger.info(f"Formed abstract concept: {abstract_concept.name} at level {abstract_concept.level}")
        
        return concept_id
    
    def find_analogies(self, source_concept_id: str) -> List[Analogy]:
        """Find analogies between source concept and others"""
        if source_concept_id not in self.concepts:
            return []
        
        source = self.concepts[source_concept_id]
        analogies = []
        
        for target_id, target in self.concepts.items():
            if target_id == source_concept_id:
                continue
            
            # Find attribute mappings
            mappings, score = self._find_structural_mapping(source, target)
            
            if score > 0.5:
                analogy = Analogy(
                    source_concept=source_concept_id,
                    target_concept=target_id,
                    mappings=mappings,
                    similarity_score=score
                )
                analogies.append(analogy)
        
        # Store top analogies
        analogies.sort(key=lambda a: a.similarity_score, reverse=True)
        self.analogies.extend(analogies[:5])
        
        return analogies
    
    def _find_structural_mapping(
        self, 
        source: Concept, 
        target: Concept
    ) -> Tuple[Dict[str, str], float]:
        """Find structural mapping between concepts"""
        mappings = {}
        scores = []
        
        source_attrs = set(source.attributes.keys())
        target_attrs = set(target.attributes.keys())
        
        # Try to map attributes based on type similarity
        for src_attr in source_attrs:
            src_type = source.attributes[src_attr].get('type')
            
            for tgt_attr in target_attrs:
                tgt_type = target.attributes[tgt_attr].get('type')
                
                if src_type == tgt_type:
                    mappings[src_attr] = tgt_attr
                    scores.append(1.0)
                    break
        
        avg_score = np.mean(scores) if scores else 0.0
        return mappings, avg_score
    
    def get_concept_hierarchy(self) -> Dict[str, Any]:
        """Get hierarchical concept structure"""
        hierarchy = defaultdict(list)
        
        for concept_id, concept in self.concepts.items():
            hierarchy[concept.level].append({
                'id': concept_id,
                'name': concept.name,
                'children': list(concept.child_concepts),
                'parents': list(concept.parent_concepts)
            })
        
        return dict(hierarchy)
    
    def transfer_knowledge(
        self, 
        from_concept: str, 
        to_concept: str
    ) -> Dict[str, Any]:
        """Transfer knowledge from one concept to another via analogy"""
        if from_concept not in self.concepts or to_concept not in self.concepts:
            return {}
        
        # Find or create analogy
        analogies = self.find_analogies(from_concept)
        relevant_analogy = next(
            (a for a in analogies if a.target_concept == to_concept),
            None
        )
        
        if not relevant_analogy:
            return {}
        
        source = self.concepts[from_concept]
        target = self.concepts[to_concept]
        
        # Transfer attributes via mapping
        transferred = {}
        for src_attr, tgt_attr in relevant_analogy.mappings.items():
            if src_attr in source.attributes:
                transferred[tgt_attr] = source.attributes[src_attr].copy()
        
        logger.info(f"Transferred {len(transferred)} attributes from {from_concept} to {to_concept}")
        
        return {
            'transferred_attributes': transferred,
            'analogy': relevant_analogy.to_dict(),
            'confidence': relevant_analogy.similarity_score
        }
    
    def get_insights(self) -> Dict[str, Any]:
        """Get abstraction engine insights"""
        concepts_by_level = defaultdict(int)
        for concept in self.concepts.values():
            concepts_by_level[concept.level] += 1
        
        return {
            'total_concepts': len(self.concepts),
            'concepts_by_level': dict(concepts_by_level),
            'max_level': max((c.level for c in self.concepts.values()), default=0),
            'analogies_found': len(self.analogies),
            'observations_buffered': len(self.observation_buffer)
        }
    
    def save_state(self, filepath: str):
        """Save abstraction state"""
        state = {
            'concepts': {k: v.to_dict() for k, v in self.concepts.items()},
            'analogies': [a.to_dict() for a in self.analogies],
            'concept_counter': self.concept_counter
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Abstraction state saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo
    engine = AbstractionEngine()
    
    # Simulate observations of similar things
    observations = [
        # Type 1: Hot liquids
        {'temperature': 90, 'volume': 250, 'container': 'cup', 'state': 'liquid'},
        {'temperature': 85, 'volume': 200, 'container': 'cup', 'state': 'liquid'},
        {'temperature': 95, 'volume': 300, 'container': 'mug', 'state': 'liquid'},
        
        # Type 2: Cold solids
        {'temperature': 5, 'volume': 100, 'container': 'box', 'state': 'solid'},
        {'temperature': 3, 'volume': 150, 'container': 'box', 'state': 'solid'},
        {'temperature': 7, 'volume': 120, 'container': 'container', 'state': 'solid'},
    ]
    
    print("Processing observations...")
    for i, obs in enumerate(observations):
        concept_id = engine.observe(obs)
        print(f"Observation {i}: matched/created concept {concept_id}")
    
    print("\nConcept Hierarchy:")
    hierarchy = engine.get_concept_hierarchy()
    print(json.dumps(hierarchy, indent=2, default=str))
    
    print("\nInsights:")
    insights = engine.get_insights()
    print(json.dumps(insights, indent=2))
    
    # Test analogies
    if len(engine.concepts) > 1:
        first_concept = list(engine.concepts.keys())[0]
        print(f"\nFinding analogies for {first_concept}...")
        analogies = engine.find_analogies(first_concept)
        for analogy in analogies:
            print(f"  {analogy.target_concept}: {analogy.similarity_score:.2f}")
