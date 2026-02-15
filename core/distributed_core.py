"""
Distributed Cognitive Core with Full Database Integration
Integrates the Universal Cognitive Core with PostgreSQL, Milvus, and Redis
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger("DistributedCognitiveCore")


class DistributedCognitiveCore:
    """
    Production-grade distributed cognitive core that:
    - Processes observations from multiple data sources
    - Maintains concept formation and rule learning
    - Persists to PostgreSQL, Milvus, and Redis
    - Integrates with AMFG gossip protocol
    """
    
    def __init__(self, node_id: str, postgres_store=None, milvus_store=None, redis_cache=None):
        self.node_id = node_id
        self.postgres = postgres_store
        self.milvus = milvus_store
        self.redis = redis_cache
        
        # Core cognitive state
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.short_term_memory: deque = deque(maxlen=1000)
        self.cross_domain_mappings: Dict[str, set] = {}
        
        # Metrics
        self.metrics = {
            "concepts_formed": 0,
            "concepts_decayed": 0,
            "rules_learned": 0,
            "transfers_made": 0,
            "goals_generated": 0,
            "total_observations": 0,
            "errors": 0,
            "uptime_seconds": 0.0,
            "last_observation_time": None,
            "start_time": time.time()
        }
        
        # Configuration
        self.config = {
            "concept_similarity_threshold": 0.75,
            "min_confidence_threshold": 0.01,
            "concept_half_life_hours": 72.0,
            "decay_check_interval": 3600,
            "checkpoint_interval": 100,
            "goal_generation_interval": 50,
            "max_rules_per_observation": 5
        }
        
        self.iteration = 0
        self._last_decay_time = time.time()
    
    async def ingest(self, observation: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Main ingestion pipeline for observations
        """
        current_time = datetime.now(timezone.utc).timestamp()
        self.iteration += 1
        self.metrics["total_observations"] += 1
        self.metrics["last_observation_time"] = current_time
        self.metrics["uptime_seconds"] = current_time - self.metrics["start_time"]
        
        try:
            # Normalize observation
            observation["domain"] = domain
            observation = self._normalize_observation(observation)
            self.short_term_memory.append(observation)
            
            # Apply concept decay if needed
            if current_time - self._last_decay_time > self.config["decay_check_interval"]:
                await self._apply_concept_decay(current_time)
                self._last_decay_time = current_time
            
            # Form or strengthen concepts
            concept_id = await self._form_concept(observation, domain, current_time)
            
            # Infer rules
            new_rules = await self._infer_rules(observation, current_time)
            await self._process_rules(new_rules, current_time)
            
            # Cross-domain transfer
            if len(self._get_active_domains()) > 1:
                await self._attempt_cross_domain_transfer(domain)
            
            # Generate goals
            if self.iteration % self.config["goal_generation_interval"] == 0:
                await self._generate_autonomous_goals(observation)
            
            # Persist to databases
            if self.postgres:
                await self.postgres.save_observation({
                    "concept_id": concept_id,
                    "symbol": observation.get("symbol"),
                    "price": observation.get("price"),
                    "volume": observation.get("volume"),
                    "metadata": {"domain": domain}
                })
                
                if self.iteration % 10 == 0:
                    await self.postgres.save_metrics(self.metrics)
            
            return {
                "success": True,
                "iteration": self.iteration,
                "concept_id": concept_id,
                "new_rules": len(new_rules),
                "concept_count": len(self.concepts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.exception(f"Error ingesting observation: {e}")
            return {"success": False, "error": str(e)}
    
    def _normalize_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize observation timestamps and data types"""
        current_time = datetime.now(timezone.utc).timestamp()
        ts = obs.get("timestamp")
        
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts = dt.timestamp()
            except:
                ts = current_time
        elif not isinstance(ts, (int, float)):
            ts = current_time
        
        obs["timestamp"] = float(ts)
        return obs
    
    def _create_feature_vector(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric features from observation"""
        features = {}
        for k, v in obs.items():
            if k not in ("timestamp", "symbol", "domain"):
                if isinstance(v, (int, float)):
                    features[k] = float(v)
        return features
    
    async def _apply_concept_decay(self, current_time: float):
        """Apply exponential decay to concepts"""
        to_remove = []
        
        for cid, concept in self.concepts.items():
            hours_since_seen = (current_time - concept.get("last_seen", current_time)) / 3600.0
            decay_factor = (0.5) ** (hours_since_seen / self.config["concept_half_life_hours"])
            
            concept["confidence"] *= decay_factor
            
            if concept["confidence"] < self.config["min_confidence_threshold"]:
                to_remove.append(cid)
        
        for cid in to_remove:
            del self.concepts[cid]
            self.metrics["concepts_decayed"] += 1
        
        if to_remove:
            logger.info(f"Decayed {len(to_remove)} concepts")
    
    async def _form_concept(self, obs: Dict[str, Any], domain: str, current_time: float) -> str:
        """Form or strengthen concepts"""
        fv = self._create_feature_vector(obs)
        
        if not fv:
            return ""
        
        # Look for similar existing concept
        best_concept = None
        best_similarity = 0.0
        
        for cid, concept in self.concepts.items():
            if concept.get("domain") != domain:
                continue
            
            concept_sig = concept.get("signature", {})
            similarity = self._cosine_similarity(fv, concept_sig)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = cid
        
        # Strengthen existing concept
        if best_concept and best_similarity >= self.config["concept_similarity_threshold"]:
            concept = self.concepts[best_concept]
            concept["examples"].append(obs)
            concept["last_seen"] = current_time
            concept["observation_count"] += 1
            concept["confidence"] = min(1.0, concept["confidence"] + 0.05)
            
            logger.debug(f"Strengthened concept {best_concept}")
            return best_concept
        
        # Create new concept
        import uuid
        cid = f"concept_{uuid.uuid4().hex}"
        
        self.concepts[cid] = {
            "id": cid,
            "domain": domain,
            "signature": fv,
            "examples": [obs],
            "first_seen": current_time,
            "last_seen": current_time,
            "confidence": 0.1,
            "observation_count": 1,
            "observation_span_hours": 0.0,
            "distinct_time_windows": 1
        }
        
        self.metrics["concepts_formed"] += 1
        
        # Save to Milvus
        if self.milvus:
            await self.milvus.insert_concept(cid, domain, fv, 0.1, current_time)
        
        # Cache in Redis
        if self.redis:
            await self.redis.cache_concept(cid, self.concepts[cid])
        
        logger.info(f"New concept {cid} in domain '{domain}'")
        return cid
    
    async def _infer_rules(self, obs: Dict[str, Any], current_time: float) -> List[Dict[str, Any]]:
        """Infer rules from observations"""
        keys = [k for k in obs.keys() if k not in ("timestamp", "domain", "symbol")]
        rules = []
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                v1, v2 = obs.get(k1), obs.get(k2)
                
                if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                    continue
                
                # Rule: if k1 is high, k2 is elevated
                if v2 > 0 and v1 > 0.7 * v2:
                    rules.append({
                        "antecedent": f"{k1}_high",
                        "consequent": f"{k2}_elevated",
                        "confidence": 0.7,
                        "created_at": current_time,
                        "last_seen": current_time
                    })
                
                # Rule: if k1 and k2 are similar
                denom = max(abs(v1), abs(v2), 1.0)
                if abs(v1 - v2) / denom < 0.1:
                    rules.append({
                        "antecedent": f"{k1}_similar",
                        "consequent": f"{k2}_similar",
                        "confidence": 0.8,
                        "created_at": current_time,
                        "last_seen": current_time
                    })
        
        return rules[:self.config["max_rules_per_observation"]]
    
    async def _process_rules(self, new_rules: List[Dict[str, Any]], current_time: float):
        """Process and store rules"""
        import hashlib
        
        for rule in new_rules:
            content = f"{rule['antecedent']}→{rule['consequent']}"
            rid = hashlib.md5(content.encode()).hexdigest()
            
            if rid in self.rules:
                self.rules[rid]["support"] += 1
                self.rules[rid]["last_seen"] = current_time
                self.rules[rid]["confidence"] = min(1.0, self.rules[rid]["confidence"] + 0.02)
            else:
                self.rules[rid] = {**rule, "support": 1, "id": rid}
                self.metrics["rules_learned"] += 1
            
            # Save to PostgreSQL
            if self.postgres:
                await self.postgres.save_rule(rid, self.rules[rid])
    
    def _get_active_domains(self) -> set:
        """Get all active domains"""
        return {c.get("domain") for c in self.concepts.values()}
    
    async def _attempt_cross_domain_transfer(self, current_domain: str):
        """Attempt to transfer knowledge across domains"""
        domains = self._get_active_domains()
        
        for other in domains:
            if other == current_domain:
                continue
            
            if other not in self.cross_domain_mappings.get(current_domain, set()):
                if current_domain not in self.cross_domain_mappings:
                    self.cross_domain_mappings[current_domain] = set()
                
                self.cross_domain_mappings[current_domain].add(other)
                self.metrics["transfers_made"] += 1
                logger.info(f"Transfer: {current_domain} → {other}")
    
    async def _generate_autonomous_goals(self, obs: Dict[str, Any]):
        """Generate autonomous goals"""
        self.metrics["goals_generated"] += 1
        logger.debug(f"Goal: analyze covariation in {list(obs.keys())[:3]}")
    
    def _cosine_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Compute cosine similarity between two vectors"""
        import math
        
        keys = set(v1.keys()) | set(v2.keys())
        if not keys:
            return 0.0
        
        dot = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in keys)
        mag1 = math.sqrt(sum(v1.get(k, 0.0) ** 2 for k in keys))
        mag2 = math.sqrt(sum(v2.get(k, 0.0) ** 2 for k in keys))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return self.metrics.copy()
    
    def get_concepts_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current concepts"""
        return {cid: c for cid, c in self.concepts.items()}
    
    def get_rules_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current rules"""
        return {rid: r for rid, r in self.rules.items()}
