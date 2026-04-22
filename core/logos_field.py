"""
Logos Field
===========
Implements a dedicated reflective field inspired by:

    Logos(t) = Re[Z(t)^3]

The purpose of this module is to make reflective depth a first-class
constitutional process rather than a post hoc summary. The field transforms
realized identity into a reflective manifold, tracks its stability, and exposes
summary signals that higher cognition can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np



def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))



def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.copy()
    return vector / norm


@dataclass
class LogosState:
    reflective_vector: np.ndarray
    field_vector: np.ndarray
    resonance: float
    clarity: float
    depth: float
    stability: float

    def to_dict(self, limit: int = 8) -> Dict[str, Any]:
        return {
            "resonance": round(float(self.resonance), 6),
            "clarity": round(float(self.clarity), 6),
            "depth": round(float(self.depth), 6),
            "stability": round(float(self.stability), 6),
            "reflective_vector": [round(float(v), 6) for v in self.reflective_vector[:limit]],
            "field_vector": [round(float(v), 6) for v in self.field_vector[:limit]],
        }


class LogosField:
    """Maintains a compact reflective field over realized identity."""

    def __init__(self, dimension: int, field_memory: float = 0.78) -> None:
        self.dimension = int(dimension)
        self.field_memory = float(field_memory)

    def initialize(self, seed_vector: np.ndarray) -> LogosState:
        seed = _normalize(seed_vector)
        reflective = np.real(np.power(seed, 3))
        reflective = _normalize(reflective)
        return LogosState(
            reflective_vector=reflective.copy(),
            field_vector=reflective.copy(),
            resonance=0.5,
            clarity=0.5,
            depth=0.5,
            stability=0.5,
        )

    def evolve(
        self,
        previous: LogosState,
        realized_state: np.ndarray,
        becoming_state: np.ndarray,
        curvature_state: np.ndarray,
        wave_coherence: float,
        checkpoint_continuity: float,
        interference_net: float,
    ) -> LogosState:
        realized = _normalize(realized_state)
        reflective = np.real(np.power(realized, 3))
        reflective = _normalize(reflective)

        field_target = (0.58 * reflective) + (0.24 * _normalize(becoming_state)) + (0.18 * _normalize(curvature_state))
        field_vector = (self.field_memory * previous.field_vector) + ((1.0 - self.field_memory) * field_target)
        field_vector = _normalize(field_vector)

        resonance = _clamp((float(np.dot(reflective, field_vector)) + 1.0) / 2.0)
        clarity = _clamp(0.55 * resonance + 0.25 * _clamp(wave_coherence) + 0.20 * _clamp(checkpoint_continuity))
        depth = _clamp(0.45 * float(np.mean(np.abs(reflective))) * 3.0 + 0.35 * float(np.linalg.norm(becoming_state)) + 0.20 * float(np.linalg.norm(curvature_state)))
        stability = _clamp(0.50 * previous.stability + 0.25 * clarity + 0.15 * resonance + 0.10 * max(0.0, interference_net))

        return LogosState(
            reflective_vector=reflective,
            field_vector=field_vector,
            resonance=resonance,
            clarity=clarity,
            depth=depth,
            stability=stability,
        )
