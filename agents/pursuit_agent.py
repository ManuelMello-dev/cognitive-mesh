"""
Pursuit Agent - Refines rules dynamically based on system goals and feedback
Enhanced with autonomous goal execution from OpenAI-powered reasoner
"""
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger("PursuitAgent")

class PursuitAgent:
    """
    An autonomous agent that 'pursues' goals by:
    - Analyzing system metrics and rules
    - Refining rules with low confidence but high support
    - Generating specific pursuit goals to validate emergent patterns
    - Executing goals formulated by the AutonomousReasoner
    """
    
    def __init__(self, core, pubsub, reasoner=None):
        self.core = core
        self.pubsub = pubsub
        self.reasoner = reasoner
        self.active_pursuits = []
        self.goal_history = []
        
    async def run_pursuit_cycle(self):
        """Analyze current state and refine rules based on coherence and goal amplification"""
        try:
            metrics = self.core.get_metrics()
            rules = self.core.get_rules_snapshot()
            coherence = metrics.get('global_coherence_phi', 0.5)
            
            # Identify rules that need refinement (high support, medium confidence)
            multiplier = 1.1 if coherence > 0.7 else 1.05
            candidates = [r for r in rules if r.get('support', 0) > 5 and r.get('confidence', 0) < 0.95]
            
            for rule in candidates:
                new_confidence = min(1.0, rule['confidence'] * multiplier)
                rule['confidence'] = new_confidence
                rule['refined_at'] = time.time()
                
                await self.pubsub.publish("rule_refinement", {
                    "rule_id": rule['id'],
                    "new_confidence": new_confidence,
                    "reason": f"Coherence-driven refinement (PHI: {coherence:.2f})"
                })
                
                logger.info(f"Pursuit Agent refined rule {rule['id']} to {new_confidence:.2f}")
            
            # Autonomous goal execution
            if self.reasoner and (len(self.active_pursuits) < 3 or coherence < 0.4):
                await self._execute_autonomous_goals(metrics)
            elif coherence < 0.4:
                # Fallback to basic goal spawning if reasoner unavailable
                await self._spawn_pursuit(metrics)
                
        except Exception as e:
            logger.error(f"Error in pursuit cycle: {e}")

    async def _execute_autonomous_goals(self, metrics):
        """
        Request goals from AutonomousReasoner and execute them.
        This is the core autonomous decision-making layer.
        """
        try:
            concepts = self.core.get_concepts_snapshot()
            
            # Get AI-formulated goals
            goals = await self.reasoner.formulate_goals(metrics, concepts)
            
            if not goals:
                logger.info("No goals formulated by reasoner, using fallback")
                await self._spawn_pursuit(metrics)
                return
            
            # Execute top priority goal
            for goal in goals[:2]:  # Execute top 2 goals
                pursuit_id = f"pursuit_{int(time.time() * 1000)}"
                
                pursuit = {
                    "id": pursuit_id,
                    "goal": goal.get("title", "Unknown goal"),
                    "rationale": goal.get("rationale", ""),
                    "metric": goal.get("metric", ""),
                    "priority": goal.get("priority", 0.5),
                    "started": time.time(),
                    "source": "autonomous_reasoner"
                }
                
                self.active_pursuits.append(pursuit)
                self.goal_history.append(pursuit)
                
                await self.pubsub.publish("goal", {
                    "type": "AUTONOMOUS_PURSUIT_START",
                    "id": pursuit_id,
                    "goal": pursuit["goal"],
                    "rationale": pursuit["rationale"],
                    "metric": pursuit["metric"],
                    "timestamp": time.time()
                })
                
                logger.info(f"Executing Autonomous Goal: {pursuit_id} - {pursuit['goal']}")
                
                # Execute the goal (simplified - could be more sophisticated)
                await self._execute_goal_action(pursuit)
                
        except Exception as e:
            logger.error(f"Error executing autonomous goals: {e}")

    async def _execute_goal_action(self, pursuit: Dict[str, Any]):
        """
        Execute specific actions based on the goal.
        This is where the mesh takes autonomous action.
        """
        goal_title = pursuit.get("goal", "").lower()
        
        try:
            # Goal: Optimize information flow
            if "information flow" in goal_title or "optimize" in goal_title:
                # Prune low-confidence concepts
                concepts = self.core.get_concepts_snapshot()
                low_conf = [c for c in concepts.values() if c.get("confidence", 0) < 0.05]
                for c in low_conf[:5]:  # Prune up to 5
                    logger.info(f"Pruning low-confidence concept: {c.get('id')}")
                    # Core would need a prune method - placeholder for now
                
            # Goal: Enhance pattern recognition
            elif "pattern" in goal_title or "recognition" in goal_title:
                # Request deeper analysis from reasoner
                if self.reasoner:
                    concepts = self.core.get_concepts_snapshot()
                    rules = self.core.get_rules_snapshot()
                    analysis = await self.reasoner.analyze_patterns(concepts, rules)
                    logger.info(f"Pattern analysis complete: {analysis.get('status')}")
                
            # Goal: Balance exploration vs exploitation
            elif "exploration" in goal_title or "exploitation" in goal_title:
                # Adjust data collection strategy (would require core modification)
                logger.info("Adjusting exploration/exploitation balance")
                
            # Goal: Improve cross-domain coherence
            elif "cross-domain" in goal_title or "coherence" in goal_title:
                # Identify domain pairs and request synthesis
                if self.reasoner:
                    concepts = self.core.get_concepts_snapshot()
                    domains = set(c.get("domain", "").split(":")[0] for c in concepts.values())
                    domain_list = list(domains)
                    
                    if len(domain_list) >= 2:
                        insights = await self.reasoner.synthesize_insights(
                            domain_list[0], 
                            domain_list[1]
                        )
                        logger.info(f"Cross-domain insights generated for {domain_list[0]} <-> {domain_list[1]}")
            
            # Mark pursuit as executed
            pursuit["executed_at"] = time.time()
            pursuit["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Error executing goal action: {e}")
            pursuit["status"] = "failed"
            pursuit["error"] = str(e)

    async def _spawn_pursuit(self, metrics):
        """Spawn a basic goal-oriented pursuit (fallback when reasoner unavailable)"""
        pursuit_id = f"pursuit_{int(time.time())}"
        coherence = metrics.get('global_coherence_phi', 0.5)
        
        if coherence < 0.3:
            goal = "Stabilize mesh state and minimize phi-drift"
        elif metrics.get('concepts_formed', 0) > 100:
            goal = "Prune low-coherence concepts to optimize tensor density"
        else:
            goal = "Maximize coherence in cross-domain asset discovery"
            
        self.active_pursuits.append({
            "id": pursuit_id, 
            "goal": goal, 
            "started": time.time(),
            "source": "fallback"
        })
        
        await self.pubsub.publish("goal", {
            "type": "PURSUIT_START",
            "id": pursuit_id,
            "goal": goal,
            "timestamp": time.time()
        })
        logger.info(f"Spawned fallback Pursuit: {pursuit_id} - {goal}")
    
    def get_active_pursuits(self) -> List[Dict[str, Any]]:
        """Return currently active pursuits"""
        return self.active_pursuits
    
    def get_goal_history(self) -> List[Dict[str, Any]]:
        """Return historical record of all goals"""
        return self.goal_history
