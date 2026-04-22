"""
Identity Checkpoint Recursion
============================
Implements constitutional checkpoint evolution:

    C_{n+1} = f(C_n) + I_n · δ

A checkpoint is not merely a saved state. It is a recursive identity-memory trace
that preserves becoming, stabilization, salience, and continuity pressure across time.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

import numpy as np



def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))



@dataclass
class IdentityCheckpoint:
    checkpoint_id: str
    iteration: int
    domain: str
    realized_state: np.ndarray
    becoming_state: np.ndarray
    curvature_state: np.ndarray
    awareness: float
    salience: float
    coherence: float
    delta_norm: float
    continuity: float

    def to_dict(self, dims: int = 8) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "iteration": self.iteration,
            "domain": self.domain,
            "awareness": round(float(self.awareness), 6),
            "salience": round(float(self.salience), 6),
            "coherence": round(float(self.coherence), 6),
            "delta_norm": round(float(self.delta_norm), 6),
            "continuity": round(float(self.continuity), 6),
            "realized_state": [round(float(v), 6) for v in self.realized_state[:dims]],
            "becoming_state": [round(float(v), 6) for v in self.becoming_state[:dims]],
            "curvature_state": [round(float(v), 6) for v in self.curvature_state[:dims]],
        }



@dataclass
class CheckpointLedger:
    domain: str
    checkpoints: Deque[IdentityCheckpoint] = field(default_factory=lambda: deque(maxlen=128))
    recursive_center: np.ndarray | None = None
    continuity_score: float = 0.5
    amplification_score: float = 0.0



class IdentityCheckpointRecursion:
    """Maintains recursive checkpoint evolution for constitutional identity."""

    def __init__(self, dimension: int, retention: int = 128, recursive_gain: float = 0.18) -> None:
        self.dimension = int(dimension)
        self.retention = int(retention)
        self.recursive_gain = float(recursive_gain)
        self.ledgers: Dict[str, CheckpointLedger] = {}
        self.counter = 0

    def _ledger(self, domain: str) -> CheckpointLedger:
        ledger = self.ledgers.get(domain)
        if ledger is None:
            ledger = CheckpointLedger(domain=domain, checkpoints=deque(maxlen=self.retention))
            self.ledgers[domain] = ledger
        return ledger

    def update(
        self,
        *,
        domain: str,
        iteration: int,
        realized_state: np.ndarray,
        becoming_state: np.ndarray,
        curvature_state: np.ndarray,
        awareness: float,
        input_intensity: float,
        coherence: float,
    ) -> Dict[str, Any]:
        ledger = self._ledger(domain)
        if ledger.recursive_center is None:
            ledger.recursive_center = realized_state.copy()

        previous_center = ledger.recursive_center.copy()
        delta = realized_state - previous_center
        delta_norm = float(np.linalg.norm(delta))
        becoming_norm = float(np.linalg.norm(becoming_state))
        curvature_norm = float(np.linalg.norm(curvature_state))

        salience = _clamp(
            0.35 * _clamp(awareness)
            + 0.30 * min(1.0, input_intensity)
            + 0.20 * _clamp(0.5 + coherence)
            + 0.15 * min(1.0, delta_norm)
        )

        recursive_input = realized_state + (0.45 * becoming_state) + (0.20 * curvature_state)
        ledger.recursive_center = ((1.0 - self.recursive_gain) * ledger.recursive_center) + (self.recursive_gain * recursive_input)

        continuity = _clamp(1.0 - min(1.0, float(np.linalg.norm(realized_state - previous_center))))
        ledger.continuity_score = 0.70 * ledger.continuity_score + 0.30 * continuity

        amplification = salience * math.exp(-0.35 * max(0.0, len(ledger.checkpoints) / max(1, self.retention)))
        amplification *= (0.70 + 0.30 * _clamp(awareness))
        ledger.amplification_score = max(float(amplification), 0.0)

        checkpoint = IdentityCheckpoint(
            checkpoint_id=f"checkpoint_{self.counter:06d}",
            iteration=int(iteration),
            domain=domain,
            realized_state=realized_state.copy(),
            becoming_state=becoming_state.copy(),
            curvature_state=curvature_state.copy(),
            awareness=_clamp(awareness),
            salience=salience,
            coherence=coherence,
            delta_norm=delta_norm,
            continuity=ledger.continuity_score,
        )
        self.counter += 1
        ledger.checkpoints.append(checkpoint)

        checkpoint_dict = checkpoint.to_dict()
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "continuity": round(float(ledger.continuity_score), 6),
            "amplification": round(float(ledger.amplification_score), 6),
            "recursive_center": [round(float(v), 6) for v in ledger.recursive_center[:8]],
            "becoming_norm": round(becoming_norm, 6),
            "curvature_norm": round(curvature_norm, 6),
            "checkpoint": checkpoint_dict,
            "recent_checkpoints": [cp.to_dict() for cp in list(ledger.checkpoints)[-3:]],
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            domain: {
                "continuity_score": round(float(ledger.continuity_score), 6),
                "amplification_score": round(float(ledger.amplification_score), 6),
                "recursive_center": [round(float(v), 6) for v in (ledger.recursive_center[:8] if ledger.recursive_center is not None else np.zeros(8))],
                "checkpoint_count": len(ledger.checkpoints),
                "recent_checkpoints": [cp.to_dict() for cp in list(ledger.checkpoints)[-5:]],
            }
            for domain, ledger in self.ledgers.items()
        }
