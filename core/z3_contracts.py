"""
Public Z3 Contracts
===================
Typed public membrane for the organism-level Z3 surface.

These contracts intentionally do not mirror every internal mesh module. They
compress coordinator, world-model, learning, memory, and prediction signals into
objects suitable for user/API/UI consumption while keeping Z-prime machinery as
infrastructure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NoveltyEvent:
    """Compressed public event emitted when the mesh detects deviation from baseline."""

    event_id: str
    source: str
    signal_type: str
    novelty_score: float
    severity: str
    summary: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    baseline_version: int = 1
    observed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineState:
    """Current Z3 baseline: the organism-level definition of normal."""

    baseline_id: str
    version: int
    iteration: int
    phi: float
    sigma: float
    drift_vector: float
    regime: str
    coherence: float
    stability: float
    watch_targets: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Z3Decision:
    """Z3 adjudication result for baseline changes and novelty handling."""

    decision_id: str
    action: str
    reason: str
    confidence: float
    baseline_version: int
    linked_event_id: Optional[str] = None
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Z3State:
    """Public state contract for the Z3 console and API endpoints."""

    identity: str
    interface_version: str
    baseline: BaselineState
    novelty_events: List[NoveltyEvent] = field(default_factory=list)
    last_decision: Optional[Z3Decision] = None
    organism_state: Dict[str, Any] = field(default_factory=dict)
    public_metrics: Dict[str, Any] = field(default_factory=dict)
    next_watch_target: Optional[str] = None
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "interface_version": self.interface_version,
            "baseline": self.baseline.to_dict(),
            "novelty_events": [event.to_dict() for event in self.novelty_events],
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
            "organism_state": dict(self.organism_state),
            "public_metrics": dict(self.public_metrics),
            "next_watch_target": self.next_watch_target,
            "timestamp": self.timestamp,
        }
