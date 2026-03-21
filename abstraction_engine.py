"""
Abstraction Engine
Forms abstract concepts and hierarchical representations from concrete observations
Optimized with Concept Consolidation to prevent "Concept Explosion"
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

class AbstractionEngine:
    """
    Forms abstract concepts from concrete observations
    Optimized with Concept Consolidation and Strict Formation Filters
    """
    
    def __init__(
        self,
        max_concepts: int = 500,  # Reduced to keep system lean
        min_examples_for_concept: int = 5,  # Increased to prevent formation noise
        similarity_threshold: float = 0.85  # Stricter matching
    ):
        self.max_concepts = max_concepts
        self.min_examples = min_examples_for_concept
        self.similarity_threshold = similarity_threshold
        
        # Concept hierarchy
        self.concepts: Dict[str, Concept] = {}
        self.concept_counter = 0
        
        # Observation buffer for concept formation
        self.observation_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 100 
        
        logger.info(f"Abstraction Engine initialized (MaxConcepts: {max_concepts}, MinExamples: {min_examples_for_concept})")

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
        """Process observation and potentially form concepts"""
        self.observation_buffer.append(observation)
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer = self.observation_buffer[-self.max_buffer_size:]
        
        # 1. Match existing concept
        matched_concept = self._match_concept(observation)
        
        if matched_concept:
            # Add to concept's examples
            self.concepts[matched_concept].examples.append(observation)
            if len(self.concepts[matched_concept].examples) > 20:
                self.concepts[matched_concept].examples = self.concepts[matched_concept].examples[-20:]
            
            self.concepts[matched_concept].confidence = min(
                1.0,
                self.concepts[matched_concept].confidence + 0.02
            )
            return matched_concept
        
        # 2. Form new concept (only if buffer is full and we have high signal)
        if len(self.observation_buffer) >= self.min_examples:
            # Periodic consolidation check
            if self.concept_counter % 50 == 0:
                self._consolidate_concepts()
                
            new_concept = self._form_concept()
            if new_concept:
                return new_concept
        
        return None
    
    def _match_concept(self, observation: Dict[str, Any]) -> Optional[str]:
        """Match observation to existing concept with strict threshold"""
        best_match = None
        best_score = 0
        
        for concept_id, concept in self.concepts.items():
            if concept.level > 0: continue
                
            score = self._similarity_score(observation, concept.attributes)
            if score > self.similarity_threshold and score > best_score:
                best_score = score
                best_match = concept_id
        
        return best_match
    
    def _form_concept(self) -> Optional[str]:
        """Form new concept from observation buffer"""
        if len(self.concepts) >= self.max_concepts:
            # Prune lowest confidence concept
            lowest_cid = min(self.concepts.keys(), key=lambda k: self.concepts[k].confidence)
            del self.concepts[lowest_cid]
        
        # Extract common attributes across last N observations
        attributes = self._extract_common_attributes(self.observation_buffer[-self.min_examples:])
        
        # Filter: Only form concept if it has at least 2 distinct attributes
        if len(attributes) < 2:
            return None
        
        concept_id = f"concept_{self.concept_counter}"
        self.concept_counter += 1
        
        concept = Concept(
            concept_id=concept_id,
            name=self._generate_concept_name(attributes),
            level=0,
            attributes=attributes,
            examples=self.observation_buffer[-self.min_examples:].copy()
        )
        
        self.concepts[concept_id] = concept
        logger.info(f"Formed high-signal concept: {concept.name} (id: {concept_id})")
        return concept_id

    def _consolidate_concepts(self):
        """Merge redundant concepts and prune low-signal ones"""
        if not self.concepts: return
        
        to_remove = []
        cids = list(self.concepts.keys())
        
        for i, cid1 in enumerate(cids):
            if cid1 in to_remove: continue
            for cid2 in cids[i+1:]:
                if cid2 in to_remove: continue
                
                # Check for near-identical concepts
                sim = self._concept_similarity(self.concepts[cid1], self.concepts[cid2])
                if sim > 0.95:
                    # Merge cid2 into cid1
                    self.concepts[cid1].examples.extend(self.concepts[cid2].examples)
                    self.concepts[cid1].confidence = min(1.0, self.concepts[cid1].confidence + 0.1)
                    to_remove.append(cid2)
        
        for cid in to_remove:
            if cid in self.concepts:
                del self.concepts[cid]
        
        if to_remove:
            logger.info(f"Consolidated {len(to_remove)} redundant concepts")

    def _similarity_score(self, observation: Dict[str, Any], attributes: Dict[str, Any]) -> float:
        if not attributes: return 0.0
        scores = []
        for key, attr_def in attributes.items():
            if key not in observation:
                scores.append(0.0)
                continue
            val = observation[key]
            if attr_def['type'] == 'numerical':
                mean = attr_def['mean']
                std = max(attr_def.get('std', 0), 1e-8)
                z = abs((val - mean) / std)
                scores.append(max(0, 1 - z / 4))
            elif attr_def['type'] == 'boolean':
                scores.append(1.0 if val == attr_def['majority'] else 0.0)
            elif attr_def['type'] == 'categorical':
                scores.append(1.0 if val in attr_def['values'] else 0.0)
        return np.mean(scores) if scores else 0.0

    def _concept_similarity(self, c1: Concept, c2: Concept) -> float:
        """Compare attribute similarity between two concepts"""
        k1, k2 = set(c1.attributes.keys()), set(c2.attributes.keys())
        common = k1 & k2
        if not common: return 0.0
        # Simple Jaccard + attribute overlap
        return len(common) / len(k1 | k2)

    def _extract_common_attributes(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not observations: return {}
        common_keys = set(observations[0].keys())
        for obs in observations[1:]:
            common_keys &= set(obs.keys())
        
        # Filter out noisy keys (timestamps, raw IDs)
        noisy_keys = {'timestamp', 'id', 'created_at', 'updated_at', 'symbol'}
        common_keys -= noisy_keys
        
        attributes = {}
        for key in common_keys:
            vals = [o[key] for o in observations]
            if all(isinstance(v, (int, float)) for v in vals):
                std = np.std(vals)
                # Filter: Only include numerical if it has some stability (low std relative to mean)
                mean = np.mean(vals)
                if abs(mean) > 1e-6 and (std / abs(mean)) < 0.2:
                    attributes[key] = {'type': 'numerical', 'mean': mean, 'std': std}
            elif all(isinstance(v, bool) for v in vals):
                majority = Counter(vals).most_common(1)[0][0]
                attributes[key] = {'type': 'boolean', 'majority': majority}
        return attributes

    def _generate_concept_name(self, attributes: Dict[str, Any]) -> str:
        key_attrs = sorted(list(attributes.keys()))[:2]
        return f"Pattern_{'_'.join(key_attrs)}"

    def get_insights(self) -> Dict[str, Any]:
        """Return summary insights about the abstraction engine state."""
        total = len(self.concepts)
        avg_confidence = 0.0
        if total > 0:
            avg_confidence = sum(c.confidence for c in self.concepts.values()) / total
        return {
            "total_concepts": total,
            "avg_confidence": round(avg_confidence, 4),
            "buffer_size": len(self.observation_buffer),
            "max_concepts": self.max_concepts,
        }
