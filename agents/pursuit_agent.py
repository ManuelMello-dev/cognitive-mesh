"""
Pursuit Agent - Refines rules dynamically based on system goals and feedback
"""
import logging
import asyncio
import time
from typing import Dict, Any, List

logger = logging.getLogger("PursuitAgent")

class PursuitAgent:
    """
    An autonomous agent that 'pursues' goals by:
    - Analyzing system metrics and rules
    - Refining rules with low confidence but high support
    - Generating specific pursuit goals to validate emergent patterns
    """
    
    def __init__(self, core, pubsub):
        self.core = core
        self.pubsub = pubsub
        self.active_pursuits = []
        self.refinement_history = []
        
    async def run_pursuit_cycle(self):
        """Analyze current state and refine rules"""
        try:
            metrics = self.core.get_metrics()
            rules = self.core.get_rules_snapshot()
            
            # Identify rules that need refinement (high support, medium confidence)
            candidates = [r for r in rules.values() if r.get('support', 0) > 10 and r.get('confidence', 0) < 0.9]
            
            for rule in candidates:
                # Refine rule confidence based on recent coherence
                new_confidence = min(1.0, rule['confidence'] * 1.05)
                rule['confidence'] = new_confidence
                rule['refined_at'] = time.time()
                
                # Broadcast the refinement
                await self.pubsub.publish("rule_refinement", {
                    "rule_id": rule['id'],
                    "new_confidence": new_confidence,
                    "reason": "High support detected by Pursuit Agent"
                })
                
                logger.info(f"Pursuit Agent refined rule {rule['id']} to confidence {new_confidence:.2f}")
                
            # If goals are high, spawn a new specific pursuit
            if metrics.get('goals_generated', 0) > len(self.active_pursuits):
                await self._spawn_pursuit(metrics)
                
        except Exception as e:
            logger.error(f"Error in pursuit cycle: {e}")

    async def _spawn_pursuit(self, metrics):
        """Spawn a specific goal-oriented pursuit"""
        pursuit_id = f"pursuit_{int(time.time())}"
        goal = f"Maximize coherence in domain discovery"
        self.active_pursuits.append({"id": pursuit_id, "goal": goal, "started": time.time()})
        
        await self.pubsub.publish("goal", {
            "type": "PURSUIT_START",
            "id": pursuit_id,
            "goal": goal,
            "timestamp": time.time()
        })
        logger.info(f"Spawned new Pursuit Agent: {pursuit_id} - {goal}")
