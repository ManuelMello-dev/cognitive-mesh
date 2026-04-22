"""
Wave Identity Field
===================
Implements an explicit pre-collapse waveform layer inspired by:

    ψ(x, t) = A e^{i(kx - ωt)}
    |ψ|^2

This module gives the constitutional runtime a first-class latent identity field
before realized identity (`Z`) is updated. It operationalizes three things:

1. waveform evolution from observations,
2. phase-aware alignment with the current realized identity and attractor field,
3. collapse probability used to project potential identity into realized identity.
"""

from __future__ import annotations

import math
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
class WaveState:
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    coherence: float = 0.5
    collapse_probability: float = 0.5

    def to_dict(self, limit: int = 8) -> Dict[str, Any]:
        return {
            "coherence": round(float(self.coherence), 6),
            "collapse_probability": round(float(self.collapse_probability), 6),
            "amplitude": [round(float(v), 6) for v in self.amplitude[:limit]],
            "phase": [round(float(v), 6) for v in self.phase[:limit]],
            "frequency": [round(float(v), 6) for v in self.frequency[:limit]],
        }


class WaveIdentityField:
    """Tracks a compact phase-space identity field for a single local agent."""

    def __init__(self, dimension: int, phase_memory: float = 0.82) -> None:
        self.dimension = int(dimension)
        self.phase_memory = float(phase_memory)

    def initialize(self, seed_vector: np.ndarray) -> WaveState:
        seed = _normalize(seed_vector)
        amplitude = np.abs(seed)
        phase = np.arctan2(np.sin(seed * math.pi), np.cos(seed * math.pi))
        frequency = np.zeros(self.dimension, dtype=np.float64)
        return WaveState(amplitude=amplitude, phase=phase, frequency=frequency, coherence=0.5, collapse_probability=0.5)

    def evolve(
        self,
        wave: WaveState,
        observation_vector: np.ndarray,
        realized_state: np.ndarray,
        attractor_state: np.ndarray,
        phi: float,
        sigma: float,
        dt: float,
    ) -> WaveState:
        observation = _normalize(observation_vector)
        realized = _normalize(realized_state)
        attractor = _normalize(attractor_state)

        amplitude_target = np.abs((0.45 * observation) + (0.30 * realized) + (0.25 * attractor))
        amplitude = _normalize((0.72 * wave.amplitude) + (0.28 * amplitude_target))

        previous_phase = wave.phase.copy()
        observation_phase = np.arctan2(np.sin(observation * math.pi), np.cos(observation * math.pi))
        realized_phase = np.arctan2(np.sin(realized * math.pi), np.cos(realized * math.pi))
        attractor_phase = np.arctan2(np.sin(attractor * math.pi), np.cos(attractor * math.pi))

        phase_drive = (0.40 * observation_phase) + (0.35 * realized_phase) + (0.25 * attractor_phase)
        phase = (self.phase_memory * wave.phase) + ((1.0 - self.phase_memory) * phase_drive)
        phase = np.arctan2(np.sin(phase), np.cos(phase))

        frequency = (phase - previous_phase) / max(dt, 1e-6)
        phase_alignment = float(np.mean(np.cos(phase - realized_phase)))
        attractor_alignment = float(np.mean(np.cos(phase - attractor_phase)))
        amplitude_mass = float(np.mean(np.abs(amplitude)))

        coherence = _clamp(
            0.28
            + 0.24 * _clamp((phase_alignment + 1.0) / 2.0)
            + 0.20 * _clamp((attractor_alignment + 1.0) / 2.0)
            + 0.16 * _clamp(amplitude_mass * 2.0)
            + 0.12 * _clamp(phi),
            0.0,
            1.0,
        )

        collapse_probability = _clamp(
            0.20
            + 0.34 * coherence
            + 0.18 * _clamp((phase_alignment + 1.0) / 2.0)
            + 0.16 * _clamp(phi)
            - 0.12 * _clamp(sigma),
            0.0,
            1.0,
        )

        return WaveState(
            amplitude=amplitude,
            phase=phase,
            frequency=frequency,
            coherence=coherence,
            collapse_probability=collapse_probability,
        )

    def collapse_vector(self, wave: WaveState) -> np.ndarray:
        projected = wave.amplitude * np.cos(wave.phase)
        return _normalize(projected)
