"""
Distributed Cognitive Core with Full Database Integration
Integrates the Universal Cognitive Core with PostgreSQL, Milvus, and Redis.
Refined with PHI coherence and noise reduction logic.
"""

import asyncio
import logging
import time
import os
import uuid
import hashlib
import math
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from collections import deque

from config.config import Config

logger = logging.getLogger("DistributedCognitiveCore")

class DistributedCognitiveCore:
    """
    Production-grade distributed cognitive core that:
    - Processes observations from multiple data sources
    - Maintains concept formation and rule learning
    - Calculates Global Coherence (PHI) and Noise Levels (SIGMA)
    - Persists to PostgreSQL, Milvus, and Redis
    """
    
    def __init__(self, node_id: str, postgres=None, milvus=None, redis=None):
        self.node_id = node_id
        self.postgres = postgres
        self.milvus = milvus
        self.redis = redis
        
        # Core cognitive state
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.short_term_memory: deque = deque(maxlen=1000)
        self.cross_domain_mappings: Dict[str, Set[str]] = {}
        
        # Metrics - PHI and SIGMA are central to the consciousness model
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
            "start_time": time.time(),
            "global_coherence_phi": 0.5,
            "noise_level_sigma": 0.1,
            "attention_density": 0.0
        }
        
        self.iteration = 0
        self._last_decay_time = time.time()
    
    async def ingest(self, observation: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Main ingestion pipeline for observations"""
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
            if current_time - self._last_decay_time > Config.DECAY_CHECK_INTERVAL:
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
            if self.iteration % Config.GOAL_GENERATION_INTERVAL == 0:
                await self._generate_autonomous_goals(observation)
            
            # Update system metrics (PHI/SIGMA)
            self._update_system_metrics()
            
            # Persist to databases
            await self._persist_data(concept_id, observation)
            
            return {
                "success": True,
                "iteration": self.iteration,
                "concept_id": concept_id,
                "new_rules": len(new_rules),
                "concept_count": len(self.concepts),
                "phi": self.metrics["global_coherence_phi"],
                "sigma": self.metrics["noise_level_sigma"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.exception(f"Error ingesting observation in domain {domain}: {e}")
            return {"success": False, "error": str(e)}

    def _update_system_metrics(self):
        """Update PHI and SIGMA based on current mesh state"""
        # PHI (Global Coherence): Average confidence of all learned rules
        if not self.rules:
            phi = 0.5
        else:
            phi = sum(r["confidence"] for r in self.rules.values()) / len(self.rules)
        
        # SIGMA (Noise Level): Inverse of PHI weighted by concept volatility
        concept_volatility = 0.0
        if self.concepts:
            # Volatility is higher when concepts have low confidence but high observation counts
            volatility_sum = sum((1.0 - c["confidence"]) for c in self.concepts.values())
            concept_volatility = volatility_sum / len(self.concepts)
            
        sigma = (1.0 - phi) * 0.7 + concept_volatility * 0.3
        
        self.metrics["global_coherence_phi"] = round(phi, 4)
        self.metrics["noise_level_sigma"] = round(sigma, 4)
        self.metrics["attention_density"] = len(self.concepts) / 1000.0 # Normalized density

    async def _persist_data(self, concept_id: str, observation: Dict[str, Any]):
        """Persist data to available stores"""
        try:
            if self.postgres:
                await self.postgres.save_observation({
                    "concept_id": concept_id,
                    "symbol": observation.get("symbol"),
                    "price": observation.get("price"),
                    "volume": observation.get("volume"),
                    "metadata": {"domain": observation.get("domain")}
                })
                
                if self.iteration % 10 == 0:
                    await self.postgres.save_metrics(self.metrics)
        except Exception as e:
            logger.error(f"Postgres persistence error: {e}")

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
            if k not in ("timestamp", "symbol", "domain", "id"):
                if isinstance(v, (int, float)):
                    features[k] = float(v)
        return features
    
    async def _apply_concept_decay(self, current_time: float):
        """Apply exponential decay to concepts"""
        to_remove = []
        
        for cid, concept in self.concepts.items():
            hours_since_seen = (current_time - concept.get("last_seen", current_time)) / 3600.0
            decay_factor = (0.5) ** (hours_since_seen / Config.CONCEPT_HALF_LIFE_HOURS)
            
            concept["confidence"] *= decay_factor
            
            if concept["confidence"] < Config.MIN_CONFIDENCE_THRESHOLD:
                to_remove.append(cid)
        
        for cid in to_remove:
            del self.concepts[cid]
            self.metrics["concepts_decayed"] += 1
        
        if to_remove:
            logger.info(f"Decayed {len(to_remove)} concepts")
    
    async def _form_concept(self, obs: Dict[str, Any], domain: str, current_time: float) -> str:
        """Form or strengthen concepts"""
        fv = self._create_feature_vector(obs)
        if not fv: return ""
        
        # Look for similar existing concept in the same domain
        best_concept = None
        best_similarity = 0.0
        
        domain_concepts = [c for c in self.concepts.values() if c.get("domain") == domain]
        
        for concept in domain_concepts:
            similarity = self._cosine_similarity(fv, concept.get("signature", {}))
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = concept["id"]
        
        # Strengthen existing concept
        if best_concept and best_similarity >= Config.CONCEPT_SIMILARITY_THRESHOLD:
            concept = self.concepts[best_concept]
            concept["last_seen"] = current_time
            concept["observation_count"] += 1
            # Confidence grows as a function of similarity and repetition
            concept["confidence"] = min(1.0, concept["confidence"] + (best_similarity * 0.05))
            
            # Update signature (moving average)
            for k, v in fv.items():
                if k in concept["signature"]:
                    concept["signature"][k] = concept["signature"][k] * 0.9 + v * 0.1
                else:
                    concept["signature"][k] = v
            
            # Keep only last 10 examples to save memory
            concept.setdefault("examples", []).append(obs)
            if len(concept["examples"]) > 10:
                concept["examples"].pop(0)
                
            return best_concept
        
        # Create new concept
        cid = f"concept_{uuid.uuid4().hex}"
        self.concepts[cid] = {
            "id": cid,
            "domain": domain,
            "signature": fv,
            "examples": [obs],
            "first_seen": current_time,
            "last_seen": current_time,
            "confidence": 0.1,
            "observation_count": 1
        }
        
        self.metrics["concepts_formed"] += 1
        
        # Async persistence to vector DB and cache
        asyncio.create_task(self._persist_concept_async(cid, domain, fv))
        
        return cid

    async def _persist_concept_async(self, cid: str, domain: str, signature: Dict[str, float]):
        """Asynchronously persist concept to Milvus and Redis"""
        try:
            if self.milvus:
                await self.milvus.insert_concept(cid, domain, signature, 0.1, time.time())
            if self.redis:
                await self.redis.cache_concept(cid, self.concepts[cid])
        except Exception as e:
            logger.error(f"Async persistence error for concept {cid}: {e}")

    async def _infer_rules(self, obs: Dict[str, Any], current_time: float) -> List[Dict[str, Any]]:
        """Infer rules from observations"""
        keys = [k for k in obs.keys() if k not in ("timestamp", "domain", "symbol", "id")]
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
        
        return rules[:Config.MAX_RULES_PER_OBSERVATION]
    
    async def _process_rules(self, new_rules: List[Dict[str, Any]], current_time: float):
        """Process and store rules"""
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
            
            if self.postgres:
                await self.postgres.save_rule(rid, self.rules[rid])
    
    def _get_active_domains(self) -> Set[str]:
        """Get all active domains"""
        return {c.get("domain") for c in self.concepts.values()}
    
    async def _attempt_cross_domain_transfer(self, current_domain: str):
        """Identify potential transfers between domains"""
        domains = self._get_active_domains()
        for other in domains:
            if other == current_domain: continue
            
            if other not in self.cross_domain_mappings.get(current_domain, set()):
                self.cross_domain_mappings.setdefault(current_domain, set()).add(other)
                self.metrics["transfers_made"] += 1
                logger.info(f"Cross-domain Transfer: {current_domain} → {other}")
    
    async def _generate_autonomous_goals(self, obs: Dict[str, Any]):
        """Generate autonomous goals based on observation patterns"""
        self.metrics["goals_generated"] += 1
        logger.debug(f"Autonomous Goal Generated: Analyze covariation in {list(obs.keys())[:3]}")
    
    def _cosine_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Compute cosine similarity between two feature vectors"""
        keys = set(v1.keys()) | set(v2.keys())
        if not keys: return 0.0
        
        dot = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in keys)
        mag1 = math.sqrt(sum(v1.get(k, 0.0) ** 2 for k in keys))
        mag2 = math.sqrt(sum(v2.get(k, 0.0) ** 2 for k in keys))
        
        if mag1 == 0 or mag2 == 0: return 0.0
        return dot / (mag1 * mag2)
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
    
    def get_concepts_snapshot(self) -> Dict[str, Any]:
        return {cid: c.copy() for cid, c in self.concepts.items()}
    
    def get_rules_snapshot(self) -> Dict[str, Any]:
        return {rid: r.copy() for rid, r in self.rules.items()}
        
    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "phi": self.metrics["global_coherence_phi"],
            "sigma": self.metrics["noise_level_sigma"],
            "concepts_count": len(self.concepts),
            "rules_count": len(self.rules),
            "metrics": self.metrics,
            "timestamp": time.time()
        }
        
    async def process_network_message(self, msg: Any):
        logger.debug(f"Processing network message: {msg}")
        
    async def process_pubsub_message(self, msg: Any):
        logger.debug(f"Processing pubsub message: {msg}")
