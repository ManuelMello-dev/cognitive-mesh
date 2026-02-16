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
        """Analyze current state and refine rules based on coherence and goal amplification"""
        try:
            metrics = self.core.get_metrics()
            rules = self.core.get_rules_snapshot()
            coherence = metrics.get('coherence_score', 0.5)
            
            # Goal Amplification: Identify rules that align with high-coherence states
            # If coherence is high, we accelerate confidence in supporting rules
            multiplier = 1.1 if coherence > 0.7 else 1.05
            
            # Identify rules that need refinement (high support, medium confidence)
            candidates = [r for r in rules.values() if r.get('support', 0) > 5 and r.get('confidence', 0) < 0.95]
            
            for rule in candidates:
                # Refine rule confidence
                new_confidence = min(1.0, rule['confidence'] * multiplier)
                rule['confidence'] = new_confidence
                rule['refined_at'] = time.time()
                
                # Broadcast the refinement
                await self.pubsub.publish("rule_refinement", {
                    "rule_id": rule['id'],
                    "new_confidence": new_confidence,
                    "reason": f"Coherence-driven refinement (Coherence: {coherence:.2f})"
                })
                
                logger.info(f"Pursuit Agent refined rule {rule['id']} to confidence {new_confidence:.2f} (Multiplier: {multiplier})")
                
            # If goals are high or coherence is dipping, spawn a new specific pursuit
            if metrics.get('goals_generated', 0) > len(self.active_pursuits) or coherence < 0.4:
                await self._spawn_pursuit(metrics)
                
        except Exception as e:
            logger.error(f"Error in pursuit cycle: {e}")

    async def _spawn_pursuit(self, metrics):
        """Spawn a specific goal-oriented pursuit, optionally using LLM for dynamic rule refinement"""
        pursuit_id = f"pursuit_{int(time.time())}"
        
        # Determine goal based on system state
        coherence = metrics.get('coherence_score', 0.5)
        if coherence < 0.3:
            goal = "Stabilize mesh state and minimize phi-drift"
        elif metrics.get('concepts_formed', 0) > 100:
            goal = "Prune low-coherence concepts to optimize tensor density"
        else:
            goal = "Maximize coherence in cross-domain asset discovery"
            
        self.active_pursuits.append({"id": pursuit_id, "goal": goal, "started": time.time()})
        
        await self.pubsub.publish("goal", {
            "type": "PURSUIT_START",
            "id": pursuit_id,
            "goal": goal,
            "timestamp": time.time()
        })
        logger.info(f"Spawned new Pursuit Agent: {pursuit_id} - {goal}")
