"""
Cross-Domain Conceptualization Engine
Transfers knowledge and concepts across different domains
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class Domain:
    """Represents a knowledge domain"""
    domain_id: str
    name: str
    vocabulary: Set[str]  # Domain-specific terms
    concepts: Set[str]  # Concept IDs in this domain
    relationships: Dict[str, List[str]]  # Concept relationships
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'domain_id': self.domain_id,
            'name': self.name,
            'vocabulary_size': len(self.vocabulary),
            'concept_count': len(self.concepts),
            'relationship_count': sum(len(v) for v in self.relationships.values()),
            'created_at': self.created_at.isoformat()
        }


@dataclass
class DomainMapping:
    """Mapping between concepts across domains"""
    mapping_id: str
    source_domain: str
    target_domain: str
    concept_mappings: Dict[str, str]  # source_concept -> target_concept
    confidence: float
    bidirectional: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'mapping_id': self.mapping_id,
            'source_domain': self.source_domain,
            'target_domain': self.target_domain,
            'mappings': self.concept_mappings,
            'confidence': self.confidence,
            'bidirectional': self.bidirectional,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class TransferredKnowledge:
    """Knowledge transferred from one domain to another"""
    transfer_id: str
    source_domain: str
    target_domain: str
    knowledge_type: str  # 'rule', 'pattern', 'concept', etc.
    source_knowledge: Any
    adapted_knowledge: Any
    confidence: float
    validated: bool = False
    performance_gain: Optional[float] = None
    
    def to_dict(self):
        return {
            'transfer_id': self.transfer_id,
            'source_domain': self.source_domain,
            'target_domain': self.target_domain,
            'knowledge_type': self.knowledge_type,
            'confidence': self.confidence,
            'validated': self.validated,
            'performance_gain': self.performance_gain
        }


class CrossDomainEngine:
    """
    Enables knowledge transfer and conceptualization across domains
    """
    
    def __init__(
        self,
        min_mapping_confidence: float = 0.6,
        enable_auto_transfer: bool = True
    ):
        self.min_confidence = min_mapping_confidence
        self.enable_auto_transfer = enable_auto_transfer
        
        # Domain registry
        self.domains: Dict[str, Domain] = {}
        
        # Cross-domain mappings
        self.mappings: Dict[str, DomainMapping] = {}
        self.mapping_counter = 0
        
        # Transfer history
        self.transfers: Dict[str, TransferredKnowledge] = {}
        self.transfer_counter = 0
        
        # Meta-knowledge: patterns that work across domains
        self.universal_patterns: List[Dict[str, Any]] = []
        
        logger.info("Cross-Domain Engine initialized")
    
    def register_domain(
        self,
        domain_id: str,
        name: str,
        vocabulary: Optional[Set[str]] = None
    ) -> Domain:
        """Register a new knowledge domain"""
        domain = Domain(
            domain_id=domain_id,
            name=name,
            vocabulary=vocabulary or set(),
            concepts=set(),
            relationships=defaultdict(list)
        )
        
        self.domains[domain_id] = domain
        logger.info(f"Registered domain: {name} (id: {domain_id})")
        
        return domain
    
    def add_concept_to_domain(
        self,
        domain_id: str,
        concept_id: str,
        related_concepts: Optional[List[str]] = None
    ):
        """Add concept to a domain"""
        if domain_id not in self.domains:
            logger.error(f"Unknown domain: {domain_id}")
            return
        
        domain = self.domains[domain_id]
        domain.concepts.add(concept_id)
        
        if related_concepts:
            domain.relationships[concept_id] = related_concepts
        
        logger.debug(f"Added concept {concept_id} to domain {domain_id}")
    
    def discover_domain_mapping(
        self,
        source_domain_id: str,
        target_domain_id: str,
        concept_vectors: Optional[Dict[str, np.ndarray]] = None
    ) -> Optional[DomainMapping]:
        """
        Discover structural mapping between domains
        """
        if source_domain_id not in self.domains or target_domain_id not in self.domains:
            return None
        
        source = self.domains[source_domain_id]
        target = self.domains[target_domain_id]
        
        # Find concept mappings
        mappings = {}
        
        if concept_vectors:
            # Use vector similarity
            mappings = self._map_via_vectors(
                source.concepts,
                target.concepts,
                concept_vectors
            )
        else:
            # Use structural similarity
            mappings = self._map_via_structure(source, target)
        
        if not mappings:
            return None
        
        # Calculate confidence based on mapping coverage
        # Use the SMALLER domain as denominator (we want to know how well
        # the smaller domain maps into the larger one)
        min_size = min(len(source.concepts), len(target.concepts))
        confidence = len(mappings) / max(min_size, 1)
        confidence = min(confidence, 1.0)
        
        # Lower threshold: even partial mappings are valuable
        if confidence < 0.1 or len(mappings) < 2:
            return None
        
        # Create mapping
        mapping_id = f"mapping_{self.mapping_counter}"
        self.mapping_counter += 1
        
        mapping = DomainMapping(
            mapping_id=mapping_id,
            source_domain=source_domain_id,
            target_domain=target_domain_id,
            concept_mappings=mappings,
            confidence=confidence
        )
        
        self.mappings[mapping_id] = mapping
        logger.info(
            f"Discovered mapping: {source.name} -> {target.name} "
            f"({len(mappings)} concepts, conf: {confidence:.2f})"
        )
        
        return mapping
    
    def _map_via_vectors(
        self,
        source_concepts: Set[str],
        target_concepts: Set[str],
        vectors: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        """Map concepts using vector similarity"""
        mappings = {}
        
        for src_concept in source_concepts:
            if src_concept not in vectors:
                continue
            
            src_vec = vectors[src_concept]
            best_match = None
            best_similarity = 0
            
            for tgt_concept in target_concepts:
                if tgt_concept not in vectors:
                    continue
                
                tgt_vec = vectors[tgt_concept]
                
                # Cosine similarity
                similarity = np.dot(src_vec, tgt_vec) / (
                    np.linalg.norm(src_vec) * np.linalg.norm(tgt_vec)
                )
                
                if similarity > best_similarity and similarity > 0.7:
                    best_similarity = similarity
                    best_match = tgt_concept
            
            if best_match:
                mappings[src_concept] = best_match
        
        return mappings
    
    def _map_via_structure(
        self,
        source: Domain,
        target: Domain
    ) -> Dict[str, str]:
        """
        Map concepts using structural and name-based similarity.
        
        Strategy:
        1. Try name-based matching (concept IDs often contain descriptive tokens)
        2. Fall back to degree-based matching
        3. For market domains, use positional pairing (both domains have
           similarly-structured concepts from the same abstraction engine)
        """
        mappings = {}
        used_targets = set()
        
        # Strategy 1: Name-token overlap
        src_tokens = {}
        for c in source.concepts:
            tokens = set(c.lower().replace('_', ' ').replace('-', ' ').split())
            src_tokens[c] = tokens
        
        tgt_tokens = {}
        for c in target.concepts:
            tokens = set(c.lower().replace('_', ' ').replace('-', ' ').split())
            tgt_tokens[c] = tokens
        
        for src_c, s_tok in src_tokens.items():
            best_match = None
            best_overlap = 0
            for tgt_c, t_tok in tgt_tokens.items():
                if tgt_c in used_targets:
                    continue
                overlap = len(s_tok & t_tok)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = tgt_c
            if best_match and best_overlap >= 1:
                mappings[src_c] = best_match
                used_targets.add(best_match)
        
        # Strategy 2: Degree-based for unmapped concepts
        unmapped_src = [c for c in source.concepts if c not in mappings]
        unmapped_tgt = [c for c in target.concepts if c not in used_targets]
        
        if unmapped_src and unmapped_tgt:
            src_degrees = {
                c: len(source.relationships.get(c, []))
                for c in unmapped_src
            }
            tgt_degrees = {
                c: len(target.relationships.get(c, []))
                for c in unmapped_tgt
            }
            
            src_sorted = sorted(src_degrees.items(), key=lambda x: x[1], reverse=True)
            tgt_sorted = sorted(tgt_degrees.items(), key=lambda x: x[1], reverse=True)
            
            for i, (src_c, src_deg) in enumerate(src_sorted):
                if i < len(tgt_sorted):
                    tgt_c, tgt_deg = tgt_sorted[i]
                    if abs(src_deg - tgt_deg) <= 2:
                        mappings[src_c] = tgt_c
        
        # Strategy 3: If both domains have many concepts and few mappings,
        # do positional pairing (both come from same abstraction engine)
        if len(mappings) < 3 and len(source.concepts) >= 3 and len(target.concepts) >= 3:
            src_list = sorted(source.concepts)
            tgt_list = sorted(target.concepts)
            pair_count = min(len(src_list), len(tgt_list), 20)
            for i in range(pair_count):
                if src_list[i] not in mappings:
                    mappings[src_list[i]] = tgt_list[i]
        
        return mappings
    
    def transfer_knowledge(
        self,
        knowledge: Any,
        knowledge_type: str,
        source_domain_id: str,
        target_domain_id: str,
        mapping_id: Optional[str] = None
    ) -> Optional[TransferredKnowledge]:
        """
        Transfer knowledge from source to target domain
        """
        if source_domain_id not in self.domains or target_domain_id not in self.domains:
            return None
        
        # Find or use mapping
        if mapping_id and mapping_id in self.mappings:
            mapping = self.mappings[mapping_id]
        else:
            mapping = self.discover_domain_mapping(source_domain_id, target_domain_id)
            if not mapping:
                logger.warning(f"No mapping found between {source_domain_id} and {target_domain_id}")
                return None
        
        # Adapt knowledge to target domain
        adapted = self._adapt_knowledge(knowledge, knowledge_type, mapping)
        
        if not adapted:
            return None
        
        # Create transfer record
        transfer_id = f"transfer_{self.transfer_counter}"
        self.transfer_counter += 1
        
        transfer = TransferredKnowledge(
            transfer_id=transfer_id,
            source_domain=source_domain_id,
            target_domain=target_domain_id,
            knowledge_type=knowledge_type,
            source_knowledge=knowledge,
            adapted_knowledge=adapted,
            confidence=mapping.confidence
        )
        
        self.transfers[transfer_id] = transfer
        logger.info(
            f"Transferred {knowledge_type} from {source_domain_id} to {target_domain_id}"
        )
        
        return transfer
    
    def _adapt_knowledge(
        self,
        knowledge: Any,
        knowledge_type: str,
        mapping: DomainMapping
    ) -> Any:
        """Adapt knowledge to target domain using mapping"""
        
        if knowledge_type == 'rule':
            return self._adapt_rule(knowledge, mapping)
        elif knowledge_type == 'pattern':
            return self._adapt_pattern(knowledge, mapping)
        elif knowledge_type == 'concept':
            return self._adapt_concept(knowledge, mapping)
        else:
            # Generic adaptation: replace concepts
            return self._generic_adaptation(knowledge, mapping)
    
    def _adapt_rule(self, rule: Dict[str, Any], mapping: DomainMapping) -> Dict[str, Any]:
        """Adapt a rule to target domain"""
        adapted = rule.copy()
        
        # Replace antecedents
        if 'antecedents' in adapted:
            adapted['antecedents'] = [
                mapping.concept_mappings.get(ant, ant)
                for ant in adapted['antecedents']
            ]
        
        # Replace consequent
        if 'consequent' in adapted:
            adapted['consequent'] = mapping.concept_mappings.get(
                adapted['consequent'],
                adapted['consequent']
            )
        
        # Reduce confidence due to transfer
        if 'confidence' in adapted:
            adapted['confidence'] *= 0.8  # Transfer penalty
        
        return adapted
    
    def _adapt_pattern(self, pattern: Dict[str, Any], mapping: DomainMapping) -> Dict[str, Any]:
        """Adapt a pattern to target domain"""
        adapted = pattern.copy()
        
        # Replace features using mapping
        if 'features' in adapted:
            adapted_features = {}
            for key, value in adapted['features'].items():
                mapped_key = mapping.concept_mappings.get(key, key)
                adapted_features[mapped_key] = value
            adapted['features'] = adapted_features
        
        return adapted
    
    def _adapt_concept(self, concept: Dict[str, Any], mapping: DomainMapping) -> Dict[str, Any]:
        """Adapt a concept to target domain"""
        adapted = concept.copy()
        
        # Map concept ID
        if 'concept_id' in adapted:
            adapted['concept_id'] = mapping.concept_mappings.get(
                adapted['concept_id'],
                adapted['concept_id']
            )
        
        return adapted
    
    def _generic_adaptation(self, knowledge: Any, mapping: DomainMapping) -> Any:
        """Generic knowledge adaptation"""
        if isinstance(knowledge, dict):
            adapted = {}
            for key, value in knowledge.items():
                # Map keys
                mapped_key = mapping.concept_mappings.get(key, key)
                
                # Recursively adapt values
                if isinstance(value, (dict, list)):
                    adapted[mapped_key] = self._generic_adaptation(value, mapping)
                else:
                    adapted[mapped_key] = value
            return adapted
        
        elif isinstance(knowledge, list):
            return [self._generic_adaptation(item, mapping) for item in knowledge]
        
        else:
            return knowledge
    
    def validate_transfer(
        self,
        transfer_id: str,
        performance_before: float,
        performance_after: float
    ):
        """Validate a knowledge transfer by measuring performance"""
        if transfer_id not in self.transfers:
            return
        
        transfer = self.transfers[transfer_id]
        transfer.validated = True
        transfer.performance_gain = performance_after - performance_before
        
        # If successful, extract universal pattern
        if transfer.performance_gain > 0:
            self._extract_universal_pattern(transfer)
            logger.info(
                f"Transfer validated: {transfer_id} "
                f"(gain: {transfer.performance_gain:+.2%})"
            )
    
    def _extract_universal_pattern(self, transfer: TransferredKnowledge):
        """Extract patterns that work across domains"""
        pattern = {
            'knowledge_type': transfer.knowledge_type,
            'structure': self._extract_structure(transfer.adapted_knowledge),
            'domains': [transfer.source_domain, transfer.target_domain],
            'confidence': transfer.confidence,
            'performance_gain': transfer.performance_gain
        }
        
        self.universal_patterns.append(pattern)
        logger.debug(f"Extracted universal pattern from {transfer.transfer_id}")
    
    def _extract_structure(self, knowledge: Any) -> Dict[str, Any]:
        """Extract abstract structure from knowledge"""
        if isinstance(knowledge, dict):
            return {
                'type': 'dict',
                'keys': list(knowledge.keys()),
                'structure': {
                    k: self._extract_structure(v)
                    for k, v in knowledge.items()
                }
            }
        elif isinstance(knowledge, list):
            return {
                'type': 'list',
                'length': len(knowledge)
            }
        else:
            return {
                'type': type(knowledge).__name__
            }
    
    def suggest_transfer_opportunities(self) -> List[Dict[str, Any]]:
        """Suggest promising knowledge transfer opportunities"""
        suggestions = []
        
        # Find unmapped domain pairs
        domain_ids = list(self.domains.keys())
        
        for i, source_id in enumerate(domain_ids):
            for target_id in domain_ids[i+1:]:
                # Check if mapping exists
                existing = any(
                    m.source_domain == source_id and m.target_domain == target_id
                    for m in self.mappings.values()
                )
                
                if not existing:
                    suggestions.append({
                        'source_domain': source_id,
                        'target_domain': target_id,
                        'reason': 'unmapped_domains',
                        'priority': 'medium'
                    })
        
        # Suggest transfers for validated patterns
        for pattern in self.universal_patterns:
            if pattern['performance_gain'] > 0.1:  # Significant gain
                for domain_id in self.domains.keys():
                    if domain_id not in pattern['domains']:
                        suggestions.append({
                            'source_domain': pattern['domains'][0],
                            'target_domain': domain_id,
                            'reason': 'universal_pattern',
                            'priority': 'high',
                            'expected_gain': pattern['performance_gain']
                        })
        
        return suggestions
    
    def get_insights(self) -> Dict[str, Any]:
        """Get cross-domain engine insights"""
        validated_transfers = [t for t in self.transfers.values() if t.validated]
        successful_transfers = [t for t in validated_transfers if t.performance_gain > 0]
        
        return {
            'total_domains': len(self.domains),
            'total_mappings': len(self.mappings),
            'total_transfers': len(self.transfers),
            'validated_transfers': len(validated_transfers),
            'successful_transfers': len(successful_transfers),
            'success_rate': len(successful_transfers) / max(len(validated_transfers), 1),
            'universal_patterns': len(self.universal_patterns),
            'avg_performance_gain': np.mean([
                t.performance_gain for t in successful_transfers
            ]) if successful_transfers else 0
        }
    
    def save_state(self, filepath: str):
        """Save cross-domain state"""
        state = {
            'domains': {k: v.to_dict() for k, v in self.domains.items()},
            'mappings': {k: v.to_dict() for k, v in self.mappings.items()},
            'transfers': {k: v.to_dict() for k, v in self.transfers.items()},
            'universal_patterns': self.universal_patterns
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Cross-domain state saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo
    engine = CrossDomainEngine()
    
    print("=== Cross-Domain Transfer Demo ===\n")
    
    # Register domains
    physics = engine.register_domain("physics", "Physics")
    economics = engine.register_domain("economics", "Economics")
    
    # Add concepts
    engine.add_concept_to_domain("physics", "force", ["mass", "acceleration"])
    engine.add_concept_to_domain("physics", "mass", ["force"])
    engine.add_concept_to_domain("physics", "acceleration", ["force"])
    
    engine.add_concept_to_domain("economics", "price", ["supply", "demand"])
    engine.add_concept_to_domain("economics", "supply", ["price"])
    engine.add_concept_to_domain("economics", "demand", ["price"])
    
    # Discover mapping
    mapping = engine.discover_domain_mapping("physics", "economics")
    
    if mapping:
        print("Domain Mapping:")
        for src, tgt in mapping.concept_mappings.items():
            print(f"  {src} -> {tgt}")
        print(f"Confidence: {mapping.confidence:.2f}\n")
    
    # Transfer knowledge
    physics_rule = {
        'antecedents': ['mass', 'acceleration'],
        'consequent': 'force',
        'confidence': 0.95,
        'formula': 'F = ma'
    }
    
    transfer = engine.transfer_knowledge(
        knowledge=physics_rule,
        knowledge_type='rule',
        source_domain_id="physics",
        target_domain_id="economics"
    )
    
    if transfer:
        print("Transferred Knowledge:")
        print(f"Source: {transfer.source_knowledge}")
        print(f"Adapted: {transfer.adapted_knowledge}\n")
    
    # Simulate validation
    engine.validate_transfer(transfer.transfer_id, 0.6, 0.75)
    
    # Get suggestions
    print("Transfer Opportunities:")
    suggestions = engine.suggest_transfer_opportunities()
    for sug in suggestions[:3]:
        print(f"  {sug['source_domain']} -> {sug['target_domain']}: {sug['reason']}")
    
    print("\nInsights:")
    insights = engine.get_insights()
    print(json.dumps(insights, indent=2))
