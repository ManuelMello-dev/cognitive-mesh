"""
Mesh Contracts
==============
Strict structured output schemas for every specialized module.
No module may return natural language as intermediate data.
All inter-module communication must use these dataclasses.

Principle 2: Structured Output, Not Conversation.
Principle 6: Minimal Cross-Contamination.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────
# Constitutional Physics Layer
# ─────────────────────────────────────────────
@dataclass
class ConstitutionalOutput:
    """Structured snapshot of the agent-attractor constitutional layer."""
    agent_id: str
    attractor_id: str
    domain: str
    phi: float
    sigma: float
    coherence: float
    drift: float
    stability: float
    regime: str
    awareness: float
    assignment_distance: float = 0.0
    distance_to_attractor: float = 0.0
    gradient_norm: float = 0.0
    collapse_probability: float = 0.0
    wave_state: Dict[str, Any] = field(default_factory=dict)
    z_state: List[float] = field(default_factory=list)
    z_prime_state: List[float] = field(default_factory=list)
    z_double_prime_state: List[float] = field(default_factory=list)
    checkpoint_state: Dict[str, Any] = field(default_factory=dict)
    interference_state: Dict[str, Any] = field(default_factory=dict)
    logos_state: Dict[str, Any] = field(default_factory=dict)
    z_cubed_state: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Abstraction Engine
# ─────────────────────────────────────────────
@dataclass
class AbstractionOutput:
    """Output from the Abstraction Engine — one per formed concept."""
    concept_id: str
    label: str
    features: Dict[str, Any]
    confidence: float
    domain: str
    support_count: int = 0
    observation_count: int = 0


# ─────────────────────────────────────────────
# Reasoning Engine
# ─────────────────────────────────────────────
@dataclass
class ReasoningOutput:
    """Output from the Reasoning Engine — one per inferred rule."""
    rule_id: str
    antecedents: List[str]
    consequent: str
    confidence: float
    support_count: int
    lift: float = 1.0
    domain: str = ""


# ─────────────────────────────────────────────
# Prediction & Validation Engine
# ─────────────────────────────────────────────
@dataclass
class PredictionOutput:
    """Output from the Prediction Engine — one per active prediction."""
    prediction_id: str
    symbol: str
    domain: str
    direction: str          # 'up', 'down', 'critical'
    confidence: float
    basis: str              # rule_id or pattern label that triggered this
    horizon: int            # ticks until validation
    ticks_remaining: int
    predicted_price: float
    dead_zone_pct: float
    rule_source: Optional[str] = None


# ─────────────────────────────────────────────
# Market EEG Monitor
# ─────────────────────────────────────────────
@dataclass
class EEGOutput:
    """Output from the Market EEG Monitor — coherence snapshot."""
    phi: float                          # Global coherence (0-1)
    sigma: float                        # Noise level (0-1)
    band_power: Dict[str, float]        # {'delta':…, 'theta':…, 'alpha':…, 'beta':…, 'gamma':…}
    phase_lock_pairs: List[tuple]       # [(symbol_a, symbol_b, strength), …]
    dominant_frequency: str = "unknown"
    attention_score: float = 0.0


# ─────────────────────────────────────────────
# Cross-Domain Engine
# ─────────────────────────────────────────────
@dataclass
class CrossDomainOutput:
    """Output from the Cross-Domain Engine — one per validated transfer."""
    transfer_id: str
    source_domain: str
    target_domain: str
    knowledge_type: str     # 'rule', 'pattern', 'concept'
    confidence: float
    performance_gain: float
    mappings: Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Goal Formation System
# ─────────────────────────────────────────────
@dataclass
class GoalOutput:
    """Output from the Goal Formation System — one per active goal."""
    goal_id: str
    type: str               # GoalType.value string
    description: str
    priority: float
    success_criteria: Dict[str, Any]
    status: str = "active"
    progress: float = 0.0


# ─────────────────────────────────────────────
# Continuous Learning Engine
# ─────────────────────────────────────────────
@dataclass
class LearningOutput:
    """Output from the Continuous Learning Engine — one per detected pattern."""
    pattern_id: str
    features: List[str]
    strength: float
    hit_count: int
    domain: str = ""
    drift_detected: bool = False
    adaptation_applied: str = ""


# ─────────────────────────────────────────────
# Memory Module
# ─────────────────────────────────────────────
@dataclass
class MemoryOutput:
    """Output from the Memory System — one per recalled item."""
    key: str
    value: Any
    timestamp: float
    decay_weight: float
    emotion_weight: float = 0.0


# ─────────────────────────────────────────────
# Coordinator State (Z3 Anchor)
# ─────────────────────────────────────────────
@dataclass
class CoordinatorState:
    """
    The central state maintained by the Coordinator.
    This is the only object that crosses module boundaries.
    Modules write to it; the coordinator reads from it.
    Principle 3: Central Coordination Layer.
    Principle 4: Identity Continuity (Z3 Anchor).
    """
    iteration: int = 0

    # Constitutional layer — populated before all higher cognition
    constitutional: Optional[ConstitutionalOutput] = None

    # Module outputs — populated each cycle
    abstractions: List[AbstractionOutput] = field(default_factory=list)
    rules: List[ReasoningOutput] = field(default_factory=list)
    predictions: List[PredictionOutput] = field(default_factory=list)
    eeg: Optional[EEGOutput] = None
    cross_domain_transfers: List[CrossDomainOutput] = field(default_factory=list)
    active_goals: List[GoalOutput] = field(default_factory=list)
    learning_patterns: List[LearningOutput] = field(default_factory=list)

    # Coordinator-resolved signals
    weighted_signals: Dict[str, float] = field(default_factory=dict)
    resolved_conflicts: List[str] = field(default_factory=list)
    phi: float = 0.5
    sigma: float = 0.5

    # Z3 identity drift tracking
    z_cubed_state: Dict[str, Any] = field(default_factory=dict)
    drift_vector: float = 0.0
    cycle_timestamp: float = 0.0
