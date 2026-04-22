"""
Constitutional Mesh Physics
==========================
The mesh's constitutional layer. Every observation first becomes a local
identity process that evolves through:

- potential identity (`psi`-like latent field)
- realized identity (`Z`)
- becoming (`Z′`)
- curvature / acceleration (`Z''`)
- recursive checkpoint memory (`C_n`)
- global attractor reflection (`Z³`)

Higher cognitive modules should build on top of this layer rather than replace it.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from core.identity_checkpoint import IdentityCheckpointRecursion
from core.interference_field import InterferenceField
from core.logos_field import LogosField, LogosState
from core.wave_identity import WaveIdentityField, WaveState



def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default



def _stable_bucket(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


@dataclass
class ConstitutionalAgent:
    agent_id: str
    entity_key: str
    domain: str
    potential_state: np.ndarray
    realized_state: np.ndarray
    becoming_state: np.ndarray
    curvature_state: np.ndarray
    awareness: float = 0.5
    wave_state: Optional[WaveState] = None
    logos_state: Optional[LogosState] = None
    trajectory: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=128))
    coherence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=128))
    checkpoint_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=64))
    noise_scale: float = 0.12
    updates: int = 0
    last_attractor_id: str = ""


@dataclass
class ConstitutionalAttractor:
    attractor_id: str
    domain: str
    center: np.ndarray
    level: int = 0
    coherent_agents: Set[str] = field(default_factory=set)
    trajectory: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=256))
    gradient_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=256))
    progress_buffer: Deque[Tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=128))
    stability_score: float = 0.5
    total_coherence: float = 0.0
    average_awareness: float = 0.5
    regime: str = "critical"
    updates: int = 0


class ConstitutionalMeshPhysics:
    """
    A richer operational implementation of the user's constitutional law.

    The runtime now tracks:
        - Z   : realized identity state
        - Z′  : becoming vector
        - Z'' : curvature / acceleration of becoming
        - C_n : recursive checkpoints
        - Z³  : global attractor / reflective field summary
    """

    def __init__(
        self,
        dimension: int = 32,
        learning_rate: float = 0.10,
        base_noise: float = 0.01,
        step_scale: float = 1.00,
        max_step: float = 0.40,
        assignment_threshold: float = 0.42,
        fixed_phi: float = 0.50,
        dt: float = 0.05,
    ) -> None:
        self.dimension = int(dimension)
        self.alpha = float(learning_rate)
        self.base_noise = float(base_noise)
        self.step_scale = float(step_scale)
        self.max_step = float(max_step)
        self.assignment_threshold = float(assignment_threshold)
        self.fixed_phi = float(fixed_phi)
        self.dt = float(dt)
        self.noise_kappa = 0.6
        self.gradient_norm_min = 1e-4

        self.agents: Dict[str, ConstitutionalAgent] = {}
        self.attractors: Dict[str, ConstitutionalAttractor] = {}
        self.attractors_by_domain: Dict[str, Set[str]] = defaultdict(set)
        self.checkpoints = IdentityCheckpointRecursion(self.dimension)
        self.interference_field = InterferenceField(neighborhood_limit=6)
        self.wave_identity = WaveIdentityField(self.dimension)
        self.logos_field = LogosField(self.dimension)

        self.agent_counter = 0
        self.attractor_counter = 0
        self.iteration = 0
        self.last_snapshot: Dict[str, Any] = {}

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector.copy()
        return vector / norm

    def _vectorize_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        numeric_items = []
        symbolic_items = []

        for key, value in sorted(observation.items()):
            if isinstance(value, bool):
                numeric_items.append((key, 1.0 if value else 0.0))
            elif isinstance(value, (int, float)):
                numeric_items.append((key, float(value)))
            elif value is not None:
                symbolic_items.append((key, str(value)))

        for key, value in numeric_items:
            idx = _stable_bucket(f"num::{key}", self.dimension)
            scale = math.tanh(value / 10.0)
            vector[idx] += scale

        for key, value in symbolic_items:
            idx = _stable_bucket(f"sym::{key}::{value}", self.dimension)
            vector[idx] += 0.35

        return self._normalize(vector)

    def _entity_key(self, observation: Dict[str, Any], domain: str) -> str:
        entity = str(
            observation.get("entity_id")
            or observation.get("symbol")
            or observation.get("stream_id")
            or observation.get("concept_id")
            or "anonymous"
        ).strip()
        return f"{domain}:{entity or 'anonymous'}"

    def _create_agent(self, entity_key: str, domain: str, vector: np.ndarray) -> ConstitutionalAgent:
        agent_id = f"agent_{self.agent_counter:06d}"
        self.agent_counter += 1
        zeros = np.zeros(self.dimension, dtype=np.float64)
        agent = ConstitutionalAgent(
            agent_id=agent_id,
            entity_key=entity_key,
            domain=domain,
            potential_state=vector.copy(),
            realized_state=vector.copy(),
            becoming_state=zeros.copy(),
            curvature_state=zeros.copy(),
            awareness=0.5,
            wave_state=self.wave_identity.initialize(vector.copy()),
            logos_state=self.logos_field.initialize(vector.copy()),
        )
        agent.trajectory.append(agent.realized_state.copy())
        self.agents[entity_key] = agent
        return agent

    def _create_attractor(self, domain: str, vector: np.ndarray) -> ConstitutionalAttractor:
        attractor_id = f"attractor_{self.attractor_counter:06d}"
        self.attractor_counter += 1
        attractor = ConstitutionalAttractor(
            attractor_id=attractor_id,
            domain=domain,
            center=vector.copy(),
            regime="critical",
        )
        attractor.trajectory.append(attractor.center.copy())
        self.attractors[attractor_id] = attractor
        self.attractors_by_domain[domain].add(attractor_id)
        return attractor

    def _find_or_create_attractor(self, domain: str, vector: np.ndarray) -> Tuple[ConstitutionalAttractor, float, bool]:
        domain_ids = list(self.attractors_by_domain.get(domain, set()))
        if not domain_ids:
            attractor = self._create_attractor(domain, vector)
            return attractor, 0.0, True

        nearest: Optional[ConstitutionalAttractor] = None
        min_distance = float("inf")
        for attractor_id in domain_ids:
            attractor = self.attractors[attractor_id]
            distance = float(np.linalg.norm(vector - attractor.center))
            if distance < min_distance:
                min_distance = distance
                nearest = attractor

        if nearest is None or min_distance > self.assignment_threshold:
            attractor = self._create_attractor(domain, vector)
            return attractor, min_distance if math.isfinite(min_distance) else 0.0, True
        return nearest, min_distance, False

    def _center_variance(self, attractor: ConstitutionalAttractor, window: int = 12) -> float:
        recent = list(attractor.trajectory)[-window:]
        if len(recent) < 2:
            return 0.0
        arr = np.stack(recent, axis=0)
        return float(np.mean(np.var(arr, axis=0)))

    def _gradient_norm(self, attractor: ConstitutionalAttractor) -> float:
        recent = list(attractor.gradient_history)[-20:]
        if not recent:
            return 0.0
        if len(recent) == 1:
            return float(np.linalg.norm(recent[-1]))
        grads = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        avg = np.mean(grads, axis=0)
        return float(np.linalg.norm(avg))

    def _update_attractor_regime(self, phi: float, sigma: float) -> str:
        if phi >= 0.72 and sigma <= 0.35:
            return "ordered"
        if sigma >= 0.68:
            return "chaos"
        return "critical"

    def _logos_state(
        self,
        logos_state: LogosState,
        becoming_state: np.ndarray,
        curvature_state: np.ndarray,
    ) -> Dict[str, Any]:
        reflective_energy = float(np.mean(np.abs(logos_state.reflective_vector)))
        becoming_energy = float(np.linalg.norm(becoming_state))
        curvature_energy = float(np.linalg.norm(curvature_state))
        payload = logos_state.to_dict()
        payload.update({
            "reflective_energy": round(reflective_energy, 6),
            "becoming_energy": round(becoming_energy, 6),
            "curvature_energy": round(curvature_energy, 6),
            "logos_vector": payload.get("reflective_vector", []),
        })
        return payload

    def observe(
        self,
        observation: Dict[str, Any],
        domain: str,
        phi_hint: Optional[float] = None,
        sigma_hint: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.iteration += 1
        vector = self._vectorize_observation(observation)
        entity_key = self._entity_key(observation, domain)

        agent = self.agents.get(entity_key)
        if agent is None:
            agent = self._create_agent(entity_key, domain, vector)

        attractor, assignment_distance, created = self._find_or_create_attractor(domain, vector)

        z_old = agent.realized_state.copy()
        z3_old = attractor.center.copy()
        becoming_old = agent.becoming_state.copy()

        phi_base = _clamp((0.80 * self.fixed_phi) + (0.20 * float(phi_hint)), 0.12, 1.0) if phi_hint is not None else _clamp(self.fixed_phi, 0.12, 1.0)
        sigma_base = _clamp(sigma_hint if sigma_hint is not None else self.base_noise)

        agent.potential_state = self._normalize((0.72 * agent.potential_state) + (0.28 * vector))

        potential_grad = z_old - z3_old
        grad_norm = float(np.linalg.norm(potential_grad))
        grad_direction = potential_grad / max(grad_norm, 1e-8)
        attractor_force = phi_base * grad_direction

        if agent.wave_state is None:
            agent.wave_state = self.wave_identity.initialize(agent.potential_state)
        agent.wave_state = self.wave_identity.evolve(
            agent.wave_state,
            observation_vector=agent.potential_state,
            realized_state=z_old,
            attractor_state=z3_old,
            phi=phi_base,
            sigma=sigma_base,
            dt=self.dt,
        )
        wave_projection = self.wave_identity.collapse_vector(agent.wave_state)
        collapse_alignment = float(np.dot(wave_projection, z_old))
        collapse_probability = agent.wave_state.collapse_probability

        noise = np.random.randn(self.dimension) * max(self.base_noise, sigma_base)
        noise_term = (1.0 / max(phi_base, 1e-6)) * noise

        checkpoint_pull = np.zeros(self.dimension, dtype=np.float64)
        checkpoint_domain = self.checkpoints.export_state().get(domain, {})
        checkpoint_center = checkpoint_domain.get("recursive_center", [])
        if checkpoint_center:
            checkpoint_pull[: min(len(checkpoint_center), self.dimension)] = np.asarray(checkpoint_center[: self.dimension], dtype=np.float64)
            checkpoint_pull = self._normalize(checkpoint_pull)

        domain_neighbors = [
            {
                "agent_id": other.agent_id,
                "state": other.realized_state,
            }
            for key, other in self.agents.items()
            if other.domain == domain and key != entity_key
        ]
        interference = self.interference_field.summarize(
            agent.agent_id,
            z_old,
            becoming_old,
            domain_neighbors,
        )

        z_candidate = (0.78 * wave_projection) + (0.22 * z_old) - (attractor_force * self.dt) + noise_term
        if np.linalg.norm(checkpoint_pull) > 1e-8:
            z_candidate = (0.84 * z_candidate) + (0.16 * checkpoint_pull)
        if interference.get("partner_count", 0) > 0:
            constructive = float(interference.get("constructive", 0.0))
            destructive = float(interference.get("destructive", 0.0))
            z_candidate = ((0.88 + (0.08 * constructive)) * z_candidate) - ((0.04 + (0.08 * destructive)) * attractor_force)

        delta = z_candidate - z_old
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.max_step:
            z_candidate = z_old + (delta / delta_norm) * self.max_step

        z_new = self._normalize((collapse_probability * z_candidate) + ((1.0 - collapse_probability) * agent.potential_state))
        becoming_new = z_new - z_old
        curvature_new = becoming_new - becoming_old

        agent.realized_state = z_new
        agent.becoming_state = becoming_new
        agent.curvature_state = curvature_new
        agent.trajectory.append(z_new.copy())
        agent.last_attractor_id = attractor.attractor_id
        agent.updates += 1

        distance_old = float(np.linalg.norm(z_old - z3_old))
        distance_new = float(np.linalg.norm(z_new - z3_old))
        delta_s = distance_old - distance_new
        agent.coherence_history.append(delta_s)

        agent.awareness = _clamp(
            0.52 * agent.awareness
            + 0.20 * phi_base
            + 0.14 * _clamp(0.5 + delta_s)
            + 0.14 * collapse_probability,
            0.12,
            1.0,
        )

        checkpoint_summary = self.checkpoints.update(
            domain=domain,
            iteration=self.iteration,
            realized_state=agent.realized_state,
            becoming_state=agent.becoming_state,
            curvature_state=agent.curvature_state,
            awareness=agent.awareness,
            input_intensity=float(np.linalg.norm(vector - z_old)),
            coherence=delta_s,
        )

        if agent.logos_state is None:
            agent.logos_state = self.logos_field.initialize(agent.realized_state)
        agent.logos_state = self.logos_field.evolve(
            previous=agent.logos_state,
            realized_state=agent.realized_state,
            becoming_state=agent.becoming_state,
            curvature_state=agent.curvature_state,
            wave_coherence=float(agent.wave_state.coherence if agent.wave_state is not None else 0.0),
            checkpoint_continuity=float(checkpoint_summary.get("continuity", 0.0)),
            interference_net=float(interference.get("net", 0.0)),
        )
        agent.checkpoint_ids.append(checkpoint_summary["checkpoint_id"])

        contribution = max(0.0, delta_s) * (0.70 + 0.30 * checkpoint_summary.get("continuity", 0.5))
        if contribution > 0.0:
            attractor.progress_buffer.append((contribution, z_new.copy()))

        if created:
            attractor.center = z_new.copy()
        elif attractor.progress_buffer:
            weights = np.array([item[0] for item in attractor.progress_buffer], dtype=np.float64)
            states = np.stack([item[1] for item in attractor.progress_buffer], axis=0)
            total_weight = float(np.sum(weights)) + 1e-8
            target = np.einsum('i,ij->j', weights / total_weight, states)
            attractor.center = ((1.0 - self.alpha) * attractor.center) + (self.alpha * target)
            attractor.center = self._normalize(attractor.center)

        attractor.trajectory.append(attractor.center.copy())
        attractor.gradient_history.append(z_new - z3_old)
        attractor.coherent_agents.add(agent.agent_id)
        attractor.total_coherence += delta_s
        attractor.updates += 1

        awarenesses = [self.agents[key].awareness for key in self.agents if self.agents[key].domain == domain]
        if awarenesses:
            attractor.average_awareness = float(np.mean(awarenesses))

        variance = self._center_variance(attractor)
        critical_awareness = (max(self.base_noise, sigma_base) ** 2) / ((grad_norm ** 2) + 1e-8)
        stable = (phi_base ** 2) > critical_awareness
        stability = _clamp((1.0 - min(1.0, variance * 25.0)) * (1.0 if stable else 0.6))
        attractor.stability_score = 0.60 * attractor.stability_score + 0.25 * stability + 0.15 * checkpoint_summary.get("continuity", 0.5)

        sigma = _clamp(
            0.42 * (1.0 - attractor.stability_score)
            + 0.20 * max(0.0, -delta_s)
            + 0.20 * sigma_base
            + 0.18 * min(1.0, float(np.linalg.norm(curvature_new))),
        )
        phi = _clamp(
            0.45 * phi_base
            + 0.25 * attractor.stability_score
            + 0.15 * _clamp(1.0 - assignment_distance)
            + 0.15 * checkpoint_summary.get("continuity", 0.5),
        )
        drift = float(np.linalg.norm(attractor.center - z3_old))
        attractor.regime = self._update_attractor_regime(phi, sigma)
        logos_state = self._logos_state(agent.logos_state, agent.becoming_state, agent.curvature_state)
        wave_summary = agent.wave_state.to_dict() if agent.wave_state is not None else {}
        logos_state["wave_coherence"] = round(float(wave_summary.get("coherence", 0.0)), 6)
        logos_state["interference_net"] = round(float(interference.get("net", 0.0)), 6)
        logos_state["interference_constructive"] = round(float(interference.get("constructive", 0.0)), 6)
        logos_state["interference_destructive"] = round(float(interference.get("destructive", 0.0)), 6)

        snapshot = {
            "iteration": self.iteration,
            "agent_id": agent.agent_id,
            "entity_key": agent.entity_key,
            "domain": domain,
            "attractor_id": attractor.attractor_id,
            "new_attractor": created,
            "regime": attractor.regime,
            "phi": round(phi, 6),
            "sigma": round(sigma, 6),
            "coherence": round(delta_s, 6),
            "drift": round(drift, 6),
            "assignment_distance": round(assignment_distance, 6),
            "distance_to_attractor": round(float(np.linalg.norm(z_new - attractor.center)), 6),
            "gradient_norm": round(max(grad_norm, self.gradient_norm_min), 6),
            "stability": round(attractor.stability_score, 6),
            "awareness": round(agent.awareness, 6),
            "positive_progress_count": len(attractor.progress_buffer),
            "critical_awareness": round(float(critical_awareness), 6),
            "collapse_probability": round(collapse_probability, 6),
            "wave_state": wave_summary,
            "z_state": [round(float(v), 6) for v in agent.realized_state[:8]],
            "z_prime_state": [round(float(v), 6) for v in agent.becoming_state[:8]],
            "z_double_prime_state": [round(float(v), 6) for v in agent.curvature_state[:8]],
            "checkpoint_state": checkpoint_summary,
            "interference_state": interference,
            "logos_state": logos_state,
            "z_cubed_state": {
                "attractor_id": attractor.attractor_id,
                "domain": domain,
                "phi": round(phi, 6),
                "sigma": round(sigma, 6),
                "stability": round(attractor.stability_score, 6),
                "regime": attractor.regime,
                "coherent_agents": len(attractor.coherent_agents),
                "active_attractors": len(self.attractors_by_domain.get(domain, set())),
                "center": [round(float(v), 6) for v in attractor.center[:8]],
                "gradient_norm": round(self._gradient_norm(attractor), 6),
                "wave_state": wave_summary,
                "logos_state": logos_state,
                "interference_state": interference,
                "checkpoint_continuity": checkpoint_summary.get("continuity", 0.0),
                "checkpoint_amplification": checkpoint_summary.get("amplification", 0.0),
            },
        }
        self.last_snapshot = snapshot
        return snapshot

    def export_state(self) -> Dict[str, Any]:
        by_domain = {
            domain: sorted(list(attractor_ids))
            for domain, attractor_ids in self.attractors_by_domain.items()
        }
        return {
            "iteration": self.iteration,
            "total_agents": len(self.agents),
            "total_attractors": len(self.attractors),
            "domains": by_domain,
            "checkpoints": self.checkpoints.export_state(),
            "last_snapshot": dict(self.last_snapshot),
        }
