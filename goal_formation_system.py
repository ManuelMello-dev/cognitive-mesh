"""
Open-Ended Goal Formation System
Autonomous goal generation, evaluation, and pursuit
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Types of goals"""
    EXPLORATION = "exploration"  # Discover new patterns/knowledge
    OPTIMIZATION = "optimization"  # Improve performance
    UNDERSTANDING = "understanding"  # Deepen comprehension
    CAPABILITY = "capability"  # Develop new abilities
    CURIOSITY = "curiosity"  # Investigate anomalies
    META = "meta"  # Goals about goals


class GoalStatus(Enum):
    """Goal lifecycle states"""
    PROPOSED = "proposed"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    PAUSED = "paused"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """Goal representation"""
    goal_id: str
    goal_type: GoalType
    description: str
    success_criteria: Dict[str, Any]
    priority: float  # 0-1
    status: GoalStatus
    progress: float = 0.0  # 0-1
    value_estimate: float = 0.5  # Expected value
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    parent_goal: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    
    # Tracking
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    achieved_at: Optional[datetime] = None
    
    def to_dict(self):
        return {
            'goal_id': self.goal_id,
            'type': self.goal_type.value,
            'description': self.description,
            'priority': self.priority,
            'status': self.status.value,
            'progress': self.progress,
            'value_estimate': self.value_estimate,
            'success_criteria': self.success_criteria,
            'created_at': self.created_at.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'parent_goal': self.parent_goal,
            'sub_goals': self.sub_goals,
            'attempts': self.attempts,
            'achieved_at': self.achieved_at.isoformat() if self.achieved_at else None
        }


@dataclass
class GoalGenerationContext:
    """Context for goal generation"""
    observations: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    capabilities: Set[str]
    constraints: Dict[str, Any]
    current_state: Dict[str, Any]
    performance_metrics: Dict[str, float]


class OpenEndedGoalSystem:
    """
    Generates, evaluates, and pursues goals autonomously
    """
    
    def __init__(
        self,
        max_active_goals: int = 5,
        exploration_rate: float = 0.3,
        enable_meta_goals: bool = True
    ):
        self.max_active_goals = max_active_goals
        self.exploration_rate = exploration_rate
        self.enable_meta_goals = enable_meta_goals
        
        # Goal management
        self.goals: Dict[str, Goal] = {}
        self.active_goals: List[str] = []
        self.goal_counter = 0
        
        # Goal generation strategies
        self.generation_strategies: Dict[str, Callable] = {
            'curiosity_driven': self._generate_curiosity_goals,
            'improvement_driven': self._generate_improvement_goals,
            'exploration_driven': self._generate_exploration_goals,
            'capability_driven': self._generate_capability_goals
        }
        
        # Learning
        self.goal_outcomes: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Intrinsic motivation signals
        self.novelty_scores: deque = deque(maxlen=100)
        self.competence_scores: deque = deque(maxlen=100)
        self.curiosity_triggers: List[Dict[str, Any]] = []
        
        logger.info("Open-Ended Goal System initialized")
    
    def generate_goals(self, context: GoalGenerationContext) -> List[Goal]:
        """
        Generate new goals based on current context
        """
        generated_goals = []
        
        # Select strategies based on performance
        strategies_to_use = self._select_strategies()
        
        for strategy_name in strategies_to_use:
            if strategy_name in self.generation_strategies:
                strategy_fn = self.generation_strategies[strategy_name]
                
                try:
                    new_goals = strategy_fn(context)
                    generated_goals.extend(new_goals)
                except Exception as e:
                    logger.error(f"Error in strategy {strategy_name}: {e}")
        
        # Evaluate and rank generated goals
        for goal in generated_goals:
            goal.value_estimate = self._estimate_goal_value(goal, context)
            goal.priority = self._calculate_priority(goal, context)
        
        # Add to registry
        for goal in generated_goals:
            self.goals[goal.goal_id] = goal
            logger.info(f"Generated goal: {goal.description} (priority: {goal.priority:.2f})")
        
        # Update meta-goals
        if self.enable_meta_goals and len(generated_goals) > 0:
            self._generate_meta_goals(context)
        
        return generated_goals
    
    def _select_strategies(self) -> List[str]:
        """Select which generation strategies to use"""
        strategies = list(self.generation_strategies.keys())
        
        # Epsilon-greedy selection
        if np.random.random() < self.exploration_rate:
            # Explore: random strategy
            return [np.random.choice(strategies)]
        else:
            # Exploit: best performing strategies
            sorted_strategies = sorted(
                strategies,
                key=lambda s: self.strategy_performance[s],
                reverse=True
            )
            return sorted_strategies[:2]  # Top 2
    
    def _generate_curiosity_goals(self, context: GoalGenerationContext) -> List[Goal]:
        """Generate goals driven by curiosity/novelty"""
        goals = []
        
        # Look for anomalies or unexpected patterns
        for pattern in context.patterns:
            if pattern.get('confidence', 1.0) < 0.5:  # Low confidence = interesting
                goal_id = f"goal_{self.goal_counter}"
                self.goal_counter += 1
                
                goal = Goal(
                    goal_id=goal_id,
                    goal_type=GoalType.CURIOSITY,
                    description=f"Investigate low-confidence pattern: {pattern.get('pattern_id')}",
                    success_criteria={
                        'pattern_confidence': 0.8,
                        'observations_collected': 20
                    },
                    priority=0.5,
                    status=GoalStatus.PROPOSED
                )
                
                goals.append(goal)
        
        # Investigate unexplored areas
        if len(context.observations) > 10:
            # Check for feature space coverage
            feature_coverage = self._estimate_coverage(context.observations)
            
            if feature_coverage < 0.7:  # Low coverage
                goal_id = f"goal_{self.goal_counter}"
                self.goal_counter += 1
                
                goal = Goal(
                    goal_id=goal_id,
                    goal_type=GoalType.EXPLORATION,
                    description="Explore undersampled regions of feature space",
                    success_criteria={
                        'coverage_increase': 0.1,
                        'new_observations': 50
                    },
                    priority=0.6,
                    status=GoalStatus.PROPOSED
                )
                
                goals.append(goal)
        
        return goals
    
    def _generate_improvement_goals(self, context: GoalGenerationContext) -> List[Goal]:
        """Generate goals focused on improving performance"""
        goals = []
        
        # Improve on low-performing metrics
        for metric_name, metric_value in context.performance_metrics.items():
            if metric_value < 0.7:  # Room for improvement
                goal_id = f"goal_{self.goal_counter}"
                self.goal_counter += 1
                
                target = min(metric_value + 0.2, 1.0)
                
                goal = Goal(
                    goal_id=goal_id,
                    goal_type=GoalType.OPTIMIZATION,
                    description=f"Improve {metric_name} from {metric_value:.2f} to {target:.2f}",
                    success_criteria={
                        metric_name: target
                    },
                    priority=0.7,
                    status=GoalStatus.PROPOSED
                )
                
                goals.append(goal)
        
        return goals
    
    def _generate_exploration_goals(self, context: GoalGenerationContext) -> List[Goal]:
        """Generate goals for systematic exploration"""
        goals = []
        
        # Explore different parameter combinations
        goal_id = f"goal_{self.goal_counter}"
        self.goal_counter += 1
        
        goal = Goal(
            goal_id=goal_id,
            goal_type=GoalType.EXPLORATION,
            description="Systematically explore parameter space",
            success_criteria={
                'parameter_combinations_tested': 100,
                'optimal_configuration_found': True
            },
            priority=0.5,
            status=GoalStatus.PROPOSED
        )
        
        goals.append(goal)
        
        return goals
    
    def _generate_capability_goals(self, context: GoalGenerationContext) -> List[Goal]:
        """Generate goals for developing new capabilities"""
        goals = []
        
        # Identify missing capabilities
        desired_capabilities = {
            'pattern_recognition',
            'anomaly_detection',
            'prediction',
            'optimization',
            'adaptation'
        }
        
        missing = desired_capabilities - context.capabilities
        
        for capability in missing:
            goal_id = f"goal_{self.goal_counter}"
            self.goal_counter += 1
            
            goal = Goal(
                goal_id=goal_id,
                goal_type=GoalType.CAPABILITY,
                description=f"Develop capability: {capability}",
                success_criteria={
                    'capability_demonstrated': True,
                    'confidence_level': 0.8
                },
                priority=0.6,
                status=GoalStatus.PROPOSED
            )
            
            goals.append(goal)
        
        return goals
    
    def _generate_meta_goals(self, context: GoalGenerationContext):
        """Generate goals about the goal system itself"""
        
        # Improve goal generation
        if len(self.goal_outcomes) > 20:
            success_rate = sum(
                1 for outcome in self.goal_outcomes[-20:]
                if outcome.get('achieved', False)
            ) / 20
            
            if success_rate < 0.5:
                goal_id = f"goal_{self.goal_counter}"
                self.goal_counter += 1
                
                goal = Goal(
                    goal_id=goal_id,
                    goal_type=GoalType.META,
                    description="Improve goal achievement rate",
                    success_criteria={
                        'success_rate': 0.7,
                        'evaluation_period': 50
                    },
                    priority=0.8,
                    status=GoalStatus.PROPOSED
                )
                
                self.goals[goal_id] = goal
    
    def _estimate_coverage(self, observations: List[Dict[str, Any]]) -> float:
        """Estimate feature space coverage"""
        if not observations:
            return 0.0
        
        # Simple heuristic: unique value diversity
        all_values = set()
        total_values = 0
        
        for obs in observations:
            for value in obs.values():
                if isinstance(value, (int, float, str, bool)):
                    all_values.add(str(value))
                    total_values += 1
        
        if total_values == 0:
            return 0.0
        
        return len(all_values) / total_values
    
    def _estimate_goal_value(self, goal: Goal, context: GoalGenerationContext) -> float:
        """Estimate expected value of achieving goal"""
        value = 0.5  # Base value
        
        # Type-based value
        type_values = {
            GoalType.CURIOSITY: 0.6,
            GoalType.OPTIMIZATION: 0.8,
            GoalType.UNDERSTANDING: 0.7,
            GoalType.CAPABILITY: 0.9,
            GoalType.EXPLORATION: 0.5,
            GoalType.META: 0.95
        }
        value += type_values.get(goal.goal_type, 0.5) * 0.3
        
        # Novelty value
        similar_goals = [
            g for g in self.goals.values()
            if g.goal_type == goal.goal_type and g.status == GoalStatus.ACHIEVED
        ]
        novelty = 1.0 / (1.0 + len(similar_goals))
        value += novelty * 0.2
        
        return min(value, 1.0)
    
    def _calculate_priority(self, goal: Goal, context: GoalGenerationContext) -> float:
        """Calculate goal priority"""
        priority = goal.value_estimate
        
        # Deadline urgency
        if goal.deadline:
            time_remaining = (goal.deadline - datetime.now()).total_seconds()
            if time_remaining > 0:
                urgency = 1.0 - (time_remaining / (7 * 24 * 3600))  # Week scale
                priority += urgency * 0.2
        
        # Parent goal priority inheritance
        if goal.parent_goal and goal.parent_goal in self.goals:
            parent = self.goals[goal.parent_goal]
            priority += parent.priority * 0.3
        
        return min(priority, 1.0)
    
    def select_active_goals(self) -> List[str]:
        """Select which goals to actively pursue"""
        
        # Get all proposed or active goals
        candidates = [
            g for g in self.goals.values()
            if g.status in [GoalStatus.PROPOSED, GoalStatus.ACTIVE]
        ]
        
        # Sort by priority
        candidates.sort(key=lambda g: g.priority, reverse=True)
        
        # Select top N
        selected = candidates[:self.max_active_goals]
        
        # Update statuses
        for goal in selected:
            if goal.status == GoalStatus.PROPOSED:
                goal.status = GoalStatus.ACTIVE
                logger.info(f"Activated goal: {goal.description}")
        
        self.active_goals = [g.goal_id for g in selected]
        
        return self.active_goals
    
    def update_goal_progress(
        self,
        goal_id: str,
        metrics: Dict[str, Any]
    ):
        """Update progress on a goal"""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.attempts += 1
        goal.last_attempt = datetime.now()
        
        # Check success criteria
        criteria_met = 0
        for criterion, target in goal.success_criteria.items():
            if criterion in metrics:
                actual = metrics[criterion]
                
                if isinstance(target, bool):
                    if actual == target:
                        criteria_met += 1
                elif isinstance(target, (int, float)):
                    if actual >= target:
                        criteria_met += 1
        
        # Update progress
        goal.progress = criteria_met / max(len(goal.success_criteria), 1)
        
        # Check if achieved
        if goal.progress >= 1.0:
            goal.status = GoalStatus.ACHIEVED
            goal.achieved_at = datetime.now()
            
            # Record outcome
            self._record_goal_outcome(goal, success=True)
            
            logger.info(f"Goal achieved: {goal.description}")
    
    def _record_goal_outcome(self, goal: Goal, success: bool):
        """Record goal outcome for learning"""
        outcome = {
            'goal_id': goal.goal_id,
            'goal_type': goal.goal_type.value,
            'achieved': success,
            'attempts': goal.attempts,
            'duration': (datetime.now() - goal.created_at).total_seconds(),
            'priority': goal.priority,
            'value_estimate': goal.value_estimate
        }
        
        self.goal_outcomes.append(outcome)
        
        # Update strategy performance
        # (In real implementation, track which strategy generated this goal)
        if success:
            # Reward strategies that generated successful goals
            for strategy in self.generation_strategies.keys():
                self.strategy_performance[strategy] = (
                    0.9 * self.strategy_performance[strategy] + 0.1 * 1.0
                )
    
    def abandon_goal(self, goal_id: str, reason: str):
        """Abandon a goal"""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.status = GoalStatus.ABANDONED
        
        self._record_goal_outcome(goal, success=False)
        
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
        
        logger.info(f"Abandoned goal: {goal.description} (reason: {reason})")
    
    def decompose_goal(self, goal_id: str) -> List[Goal]:
        """Decompose complex goal into sub-goals"""
        if goal_id not in self.goals:
            return []
        
        parent_goal = self.goals[goal_id]
        sub_goals = []
        
        # Break down success criteria into individual goals
        for criterion, target in parent_goal.success_criteria.items():
            sub_goal_id = f"goal_{self.goal_counter}"
            self.goal_counter += 1
            
            sub_goal = Goal(
                goal_id=sub_goal_id,
                goal_type=parent_goal.goal_type,
                description=f"Sub-goal: Achieve {criterion} >= {target}",
                success_criteria={criterion: target},
                priority=parent_goal.priority * 0.8,
                status=GoalStatus.PROPOSED,
                parent_goal=goal_id
            )
            
            sub_goals.append(sub_goal)
            self.goals[sub_goal_id] = sub_goal
            parent_goal.sub_goals.append(sub_goal_id)
        
        logger.info(f"Decomposed goal {goal_id} into {len(sub_goals)} sub-goals")
        
        return sub_goals
    
    def get_insights(self) -> Dict[str, Any]:
        """Get goal system insights"""
        active = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        achieved = [g for g in self.goals.values() if g.status == GoalStatus.ACHIEVED]
        
        return {
            'total_goals': len(self.goals),
            'active_goals': len(active),
            'achieved_goals': len(achieved),
            'achievement_rate': len(achieved) / max(len(self.goals), 1),
            'avg_priority': np.mean([g.priority for g in active]) if active else 0,
            'goals_by_type': {
                gt.value: len([g for g in self.goals.values() if g.goal_type == gt])
                for gt in GoalType
            },
            'strategy_performance': dict(self.strategy_performance)
        }
    
    def save_state(self, filepath: str):
        """Save goal system state"""
        state = {
            'goals': {k: v.to_dict() for k, v in self.goals.items()},
            'active_goals': self.active_goals,
            'goal_outcomes': self.goal_outcomes,
            'strategy_performance': dict(self.strategy_performance)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Goal system state saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo
    system = OpenEndedGoalSystem(max_active_goals=3)
    
    print("=== Open-Ended Goal Formation Demo ===\n")
    
    # Create context
    context = GoalGenerationContext(
        observations=[
            {'temp': 20 + i, 'pressure': 1013 + i}
            for i in range(30)
        ],
        patterns=[
            {'pattern_id': 'p1', 'confidence': 0.4},
            {'pattern_id': 'p2', 'confidence': 0.9}
        ],
        capabilities={'pattern_recognition', 'prediction'},
        constraints={},
        current_state={},
        performance_metrics={
            'accuracy': 0.65,
            'coverage': 0.5
        }
    )
    
    # Generate goals
    print("Generating goals...")
    goals = system.generate_goals(context)
    
    print(f"\nGenerated {len(goals)} goals:")
    for goal in goals:
        print(f"  [{goal.goal_type.value}] {goal.description}")
        print(f"    Priority: {goal.priority:.2f}, Value: {goal.value_estimate:.2f}")
    
    # Select active goals
    print("\nSelecting active goals...")
    active = system.select_active_goals()
    print(f"Active goals: {len(active)}")
    
    # Simulate progress
    print("\nSimulating goal progress...")
    for goal_id in active[:2]:
        goal = system.goals[goal_id]
        print(f"\nUpdating: {goal.description}")
        
        # Simulate progress
        metrics = {}
        for criterion in goal.success_criteria.keys():
            if isinstance(goal.success_criteria[criterion], bool):
                metrics[criterion] = True
            else:
                metrics[criterion] = goal.success_criteria[criterion]
        
        system.update_goal_progress(goal_id, metrics)
        print(f"  Progress: {goal.progress:.0%}")
        print(f"  Status: {goal.status.value}")
    
    # Get insights
    print("\nInsights:")
    insights = system.get_insights()
    print(json.dumps(insights, indent=2))
