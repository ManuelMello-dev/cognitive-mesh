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
try:
    from core.contracts import AbstractionOutput
except ImportError:
    AbstractionOutput = None

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
    Forms abstract concepts from concrete observations.
    Optimized with Concept Consolidation and Strict Formation Filters.
    When a Milvus store is attached, concept signature vectors are stored and
    used for fast similarity search — enabling cross-domain analogical recall.
    """
    
    def __init__(
        self,
        max_concepts: int = 500,
        min_examples_for_concept: int = 5,
        similarity_threshold: float = 0.85
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

        # Optional Milvus vector store — injected by DistributedCognitiveCore after connect
        self.milvus = None
        
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

    def observe(self, observation: Dict[str, Any], domain: str = "general") -> Optional[Any]:
        """Process observation and potentially form concepts.
        Returns AbstractionOutput contract if a concept was matched or formed,
        otherwise returns None. Falls back to str if contracts not available.
        """
        self.observation_buffer.append(observation)
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer = self.observation_buffer[-self.max_buffer_size:]
        
        # 1. Match existing concept
        matched_concept = self._match_concept(observation)
        
        def _wrap(cid: str, is_new: bool):
            """Wrap a concept id in an AbstractionOutput contract if available."""
            if AbstractionOutput is None:
                return cid
            concept = self.concepts.get(cid)
            if concept is None:
                return cid
            return AbstractionOutput(
                concept_id=cid,
                concept_name=concept.name,
                domain=domain,
                confidence=concept.confidence,
                attributes=dict(concept.attributes) if concept.attributes else {},
                is_new=is_new,
                abstraction_level=concept.level,
            )

        if matched_concept:
            # Add to concept's examples
            self.concepts[matched_concept].examples.append(observation)
            if len(self.concepts[matched_concept].examples) > 20:
                self.concepts[matched_concept].examples = self.concepts[matched_concept].examples[-20:]
            self.concepts[matched_concept].confidence = min(
                1.0,
                self.concepts[matched_concept].confidence + 0.02
            )
            return _wrap(matched_concept, is_new=False)

        # 2. Form new concept (only if buffer is full and we have high signal)
        if len(self.observation_buffer) >= self.min_examples:
            # Periodic consolidation check
            if self.concept_counter % 50 == 0:
                self._consolidate_concepts()
            new_concept = self._form_concept()
            if new_concept:
                return _wrap(new_concept, is_new=True)

        return None
    
    def _match_concept(self, observation: Dict[str, Any]) -> Optional[str]:
        """Match observation to existing concept.
        Uses local structural matching first. When Milvus is available, also
        queries vector similarity to find cross-domain analogues and boosts
        the match score for any concept that appears in both results.
        """
        best_match = None
        best_score = 0

        for concept_id, concept in self.concepts.items():
            if concept.level > 0:
                continue
            score = self._similarity_score(observation, concept.attributes)
            if score > self.similarity_threshold and score > best_score:
                best_score = score
                best_match = concept_id

        # Vector-similarity boost: if Milvus is connected, query for similar
        # concepts and give a small confidence bump to any concept that appears
        # in both the structural match and the vector-similarity results.
        if self.milvus and best_match is None:
            # Only run vector search when structural matching found nothing,
            # to avoid unnecessary latency on hot paths.
            try:
                num_sig = {
                    k: v['mean'] if isinstance(v, dict) else float(v)
                    for k, v in observation.items()
                    if isinstance(v, (int, float)) or (isinstance(v, dict) and 'mean' in v)
                }
                if num_sig:
                    similar = self.find_similar_concepts_vector(num_sig, top_k=3)
                    for hit in similar:
                        cid = hit.get('id')
                        if cid and cid in self.concepts:
                            # Use vector distance as a fallback score
                            # Milvus L2 distance: lower = more similar; convert to [0,1]
                            dist = hit.get('distance', 1.0)
                            vec_score = max(0.0, 1.0 - dist)
                            if vec_score > self.similarity_threshold * 0.8 and vec_score > best_score:
                                best_score = vec_score
                                best_match = cid
                                logger.debug(f"Vector-similarity match: {cid} (score={vec_score:.3f})")
            except Exception as e:
                logger.debug(f"Vector match boost skipped: {e}")

        return best_match

    def find_similar_concepts_vector(self, signature: Dict[str, float], domain: str = None, top_k: int = 5) -> List[Dict]:
        """Query Milvus for structurally similar concepts by vector proximity.
        Uses run_coroutine_threadsafe so it works even when called from a
        synchronous context inside a running event loop thread.
        Falls back to empty list if Milvus is not connected or query fails.
        """
        if not self.milvus:
            return []
        try:
            import asyncio
            import concurrent.futures
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine on the running loop and wait with a short timeout
                future = asyncio.run_coroutine_threadsafe(
                    self.milvus.find_similar_concepts(signature, domain=domain, top_k=top_k),
                    loop
                )
                try:
                    return future.result(timeout=0.5)
                except concurrent.futures.TimeoutError:
                    logger.debug("Milvus similarity search timed out (0.5s) — skipping")
                    return []
            else:
                return loop.run_until_complete(
                    self.milvus.find_similar_concepts(signature, domain=domain, top_k=top_k)
                )
        except Exception as e:
            logger.debug(f"Milvus similarity search skipped: {e}")
        return []
    
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

        # Store concept vector in Milvus for similarity search (fire-and-forget)
        if self.milvus:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self.milvus.store_concept(
                        concept_id=concept_id,
                        signature=attributes,
                        domain=attributes.get('domain', 'general'),
                        confidence=concept.confidence
                    ))
            except Exception as e:
                logger.debug(f"Milvus concept store skipped: {e}")

        return concept_id

    def _consolidate_concepts(self):
        """Merge redundant concepts and prune low-signal ones.
        Also removes pruned concept vectors from Milvus.
        """
        if not self.concepts:
            return

        to_remove = []
        cids = list(self.concepts.keys())

        for i, cid1 in enumerate(cids):
            if cid1 in to_remove:
                continue
            for cid2 in cids[i+1:]:
                if cid2 in to_remove:
                    continue
                sim = self._concept_similarity(self.concepts[cid1], self.concepts[cid2])
                if sim > 0.95:
                    self.concepts[cid1].examples.extend(self.concepts[cid2].examples)
                    self.concepts[cid1].confidence = min(1.0, self.concepts[cid1].confidence + 0.1)
                    to_remove.append(cid2)

        for cid in to_remove:
            if cid in self.concepts:
                del self.concepts[cid]
            # Remove from Milvus too
            if self.milvus:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self.milvus.delete_concept(cid))
                except Exception:
                    pass

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

    def get_concept_hierarchy(self) -> Dict[str, Any]:
        """Return a hierarchical view of all formed concepts for the dashboard."""
        nodes = []
        edges = []
        for cid, concept in self.concepts.items():
            nodes.append({
                "id": cid,
                "name": concept.name,
                "level": concept.level,
                "confidence": round(concept.confidence, 4),
                "example_count": len(concept.examples),
                "created_at": concept.created_at.isoformat(),
            })
            for parent_id in concept.parent_concepts:
                edges.append({"from": parent_id, "to": cid})
        return {
            "nodes": nodes,
            "edges": edges,
            "total_levels": max((c.level for c in self.concepts.values()), default=0) + 1,
        }

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
