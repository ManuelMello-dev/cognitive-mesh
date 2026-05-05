"""
Z3 Public Interface
===================
Projection/adjudication layer that turns raw mesh internals into the public Z3
console contract.

Z3 is the persistent organism-level state. Z-prime agents, constitutional
snapshots, coordinator payloads, learning drift, and world-model losses remain
internal machinery. This layer exposes only the compressed Z3 baseline, novelty
feed, adjudication decision, and watch target.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

from z3_adjudicator import Z3Adjudicator
from z3_contracts import BaselineState, NoveltyEvent, Z3Decision, Z3EvidenceScore, Z3State, Z3Transition


class Z3Interface:
    """Public membrane for Z3 state projection and novelty adjudication."""

    def __init__(self, novelty_threshold: float = 0.35, severe_threshold: float = 0.72) -> None:
        self.novelty_threshold = float(novelty_threshold)
        self.severe_threshold = float(severe_threshold)
        self.adjudicator = Z3Adjudicator(
            novelty_threshold=self.novelty_threshold,
            update_threshold=self.severe_threshold,
        )
        self._baseline_version = 1
        self._baseline_id = "z3-baseline-1"
        self._last_signature: Optional[str] = None
        self._events: deque[NoveltyEvent] = deque(maxlen=100)
        self._event_ids: set[str] = set()
        self._decisions: deque[Z3Decision] = deque(maxlen=100)
        self._decision_ids: set[str] = set()
        self._transitions: deque = deque(maxlen=100)

    def project(
        self,
        coordinator_state: Dict[str, Any],
        world_model: Optional[Dict[str, Any]] = None,
        resonant_memory: Optional[Dict[str, Any]] = None,
        learning: Optional[Dict[str, Any]] = None,
        predictions: Optional[Iterable[Dict[str, Any]]] = None,
        recursive_state: Optional[Dict[str, Any]] = None,
    ) -> Z3State:
        """Return the public Z3 projection for the current runtime cache."""
        now = time.time()
        metrics = self._dict(coordinator_state.get("metrics"))
        z_cubed = self._dict(coordinator_state.get("z_cubed_state"))
        constitutional = self._dict(coordinator_state.get("constitutional"))
        recursive_state = self._dict(recursive_state or coordinator_state.get("recursive_state"))
        world_model = self._dict(world_model or coordinator_state.get("world_model"))
        resonant_memory = self._dict(resonant_memory or coordinator_state.get("resonant_memory"))
        learning = self._dict(learning or coordinator_state.get("learning"))
        predictions_list = list(predictions if predictions is not None else coordinator_state.get("predictions", []) or [])

        phi = self._float(coordinator_state.get("phi", metrics.get("global_coherence_phi", 0.5)), 0.5)
        sigma = self._float(coordinator_state.get("sigma", metrics.get("noise_level_sigma", 0.5)), 0.5)
        drift = self._float(coordinator_state.get("drift_vector", recursive_state.get("drift", 0.0)), 0.0)
        iteration = int(self._float(coordinator_state.get("iteration", recursive_state.get("iteration", 0)), 0.0))
        coherence = self._float(
            z_cubed.get("coherence", constitutional.get("coherence", metrics.get("global_coherence_phi", phi))),
            phi,
        )
        stability = self._float(z_cubed.get("stability", constitutional.get("stability", 1.0 - sigma)), 1.0 - sigma)
        regime = str(z_cubed.get("regime", constitutional.get("regime", self._regime(phi, sigma))))

        watch_targets = self._watch_targets(metrics, world_model, learning, predictions_list, recursive_state)
        signature = self._baseline_signature(phi, sigma, drift, regime, watch_targets)
        novelty_events = self._extract_novelty_events(world_model, learning, recursive_state, self._baseline_version, now)
        for event in novelty_events:
            if event.event_id not in self._event_ids:
                self._events.appendleft(event)
                self._event_ids.add(event.event_id)

        signature_changed = self._last_signature is not None and signature != self._last_signature
        decision, transition, resulting_version = self.adjudicator.adjudicate(
            events=list(self._events),
            phi=phi,
            sigma=sigma,
            drift=drift,
            baseline_version=self._baseline_version,
            signature_changed=signature_changed,
            now=now,
        )
        if resulting_version != self._baseline_version:
            self._baseline_version = resulting_version
            self._baseline_id = f"z3-baseline-{self._baseline_version}"
        if decision.decision_id not in self._decision_ids:
            self._decisions.appendleft(decision)
            self._decision_ids.add(decision.decision_id)
        if transition:
            self._transitions.appendleft(transition)
        self._last_signature = signature

        baseline = BaselineState(
            baseline_id=self._baseline_id,
            version=self._baseline_version,
            iteration=iteration,
            phi=round(phi, 6),
            sigma=round(sigma, 6),
            drift_vector=round(drift, 6),
            regime=regime,
            coherence=round(coherence, 6),
            stability=round(stability, 6),
            watch_targets=watch_targets,
            metrics=self._public_baseline_metrics(metrics, world_model, resonant_memory, recursive_state),
            summary=self._baseline_summary(phi, sigma, drift, regime),
            updated_at=now,
        )

        organism_state = {
            "cycle": iteration,
            "state": self._coherence_label(phi),
            "coherence": round(phi, 6),
            "noise": round(sigma, 6),
            "drift": round(drift, 6),
            "novelty_pressure": self._current_novelty_pressure(world_model, learning, recursive_state),
            "active_predictions": len(predictions_list),
            "active_goals": int(self._float(metrics.get("active_goals", metrics.get("total_goals", 0)), 0.0)),
        }

        public_metrics = {
            "observations": int(self._float(metrics.get("total_observations", 0), 0.0)),
            "concepts": int(self._float(metrics.get("total_concepts", 0), 0.0)),
            "rules": int(self._float(metrics.get("total_rules", 0), 0.0)),
            "prediction_accuracy": round(self._float(metrics.get("prediction_accuracy", 0.0), 0.0), 6),
            "world_model_loss": round(self._float(recursive_state.get("world_model_loss", metrics.get("world_model_loss", 0.0)), 0.0), 6),
            "memory_reconstruction_confidence": round(
                self._float(metrics.get("memory_reconstruction_confidence", 0.0), 0.0), 6
            ),
        }

        return Z3State(
            identity="Z3",
            interface_version="0.1",
            baseline=baseline,
            novelty_events=list(self._events)[:25],
            last_decision=decision,
            organism_state=organism_state,
            public_metrics=public_metrics,
            next_watch_target=watch_targets[0] if watch_targets else None,
            transitions=list(self._transitions)[:25],
            timestamp=now,
        )

    def restore_from_public_state(self, z3_state: Dict[str, Any]) -> None:
        """Restore baseline/events/decisions from a persisted public Z3 snapshot."""
        if not isinstance(z3_state, dict):
            return
        baseline = self._dict(z3_state.get("baseline"))
        version = int(self._float(baseline.get("version", self._baseline_version), self._baseline_version))
        self._baseline_version = max(1, version)
        self._baseline_id = str(baseline.get("baseline_id", f"z3-baseline-{self._baseline_version}"))
        self._events.clear()
        self._event_ids.clear()
        for raw_event in z3_state.get("novelty_events", []) or []:
            if not isinstance(raw_event, dict) or not raw_event.get("event_id"):
                continue
            event = NoveltyEvent(
                event_id=str(raw_event.get("event_id")),
                source=str(raw_event.get("source", "unknown")),
                signal_type=str(raw_event.get("signal_type", "unknown")),
                novelty_score=self._float(raw_event.get("novelty_score", 0.0), 0.0),
                severity=str(raw_event.get("severity", "low")),
                summary=str(raw_event.get("summary", "Persisted Z3 novelty event.")),
                evidence=self._dict(raw_event.get("evidence")),
                baseline_version=int(self._float(raw_event.get("baseline_version", self._baseline_version), self._baseline_version)),
                observed_at=self._float(raw_event.get("observed_at", 0.0), 0.0),
            )
            self._events.append(event)
            self._event_ids.add(event.event_id)

        self._decisions.clear()
        self._decision_ids.clear()
        raw_decision = self._dict(z3_state.get("last_decision"))
        if raw_decision.get("decision_id"):
            raw_score = self._dict(raw_decision.get("evidence_score"))
            score = None
            if raw_score:
                score = Z3EvidenceScore(
                    event_id=str(raw_score.get("event_id", "none")),
                    novelty=self._float(raw_score.get("novelty", 0.0), 0.0),
                    coherence=self._float(raw_score.get("coherence", 0.0), 0.0),
                    stability=self._float(raw_score.get("stability", 0.0), 0.0),
                    noise=self._float(raw_score.get("noise", 0.0), 0.0),
                    drift_pressure=self._float(raw_score.get("drift_pressure", 0.0), 0.0),
                    trust=self._float(raw_score.get("trust", 0.0), 0.0),
                    recommendation=str(raw_score.get("recommendation", "observe")),
                    rationale=str(raw_score.get("rationale", "Persisted evidence score.")),
                )
            decision = Z3Decision(
                decision_id=str(raw_decision.get("decision_id")),
                action=str(raw_decision.get("action", "observe")),
                reason=str(raw_decision.get("reason", "Persisted Z3 decision.")),
                confidence=self._float(raw_decision.get("confidence", 0.0), 0.0),
                baseline_version=int(self._float(raw_decision.get("baseline_version", self._baseline_version), self._baseline_version)),
                linked_event_id=raw_decision.get("linked_event_id"),
                evidence_score=score,
                created_at=self._float(raw_decision.get("created_at", 0.0), 0.0),
            )
            self._decisions.append(decision)
            self._decision_ids.add(decision.decision_id)

        self._transitions.clear()
        for raw_transition in z3_state.get("transitions", []) or []:
            if not isinstance(raw_transition, dict) or not raw_transition.get("transition_id"):
                continue
            self._transitions.append(Z3Transition(
                transition_id=str(raw_transition.get("transition_id")),
                from_baseline_version=int(self._float(raw_transition.get("from_baseline_version", self._baseline_version), self._baseline_version)),
                to_baseline_version=int(self._float(raw_transition.get("to_baseline_version", self._baseline_version), self._baseline_version)),
                action=str(raw_transition.get("action", "observe")),
                decision_id=str(raw_transition.get("decision_id", "")),
                linked_event_id=raw_transition.get("linked_event_id"),
                created_at=self._float(raw_transition.get("created_at", 0.0), 0.0),
            ))

    def _extract_novelty_events(
        self,
        world_model: Dict[str, Any],
        learning: Dict[str, Any],
        recursive_state: Dict[str, Any],
        baseline_version: int,
        now: float,
    ) -> List[NoveltyEvent]:
        events: List[NoveltyEvent] = []
        latest = self._dict(world_model.get("latest"))
        novelty = self._float(latest.get("novelty", latest.get("memory_loss", 0.0)), 0.0)
        if latest and novelty >= self.novelty_threshold:
            iteration = int(self._float(latest.get("iteration", world_model.get("iteration", 0)), 0.0))
            observed_at = self._float(latest.get("timestamp", now), now)
            severity = self._severity(novelty)
            events.append(
                NoveltyEvent(
                    event_id=f"world-model-{iteration}-{int(observed_at)}",
                    source="world_model",
                    signal_type="memory_distance",
                    novelty_score=round(novelty, 6),
                    severity=severity,
                    summary=f"World model detected {severity} deviation from learned memory baseline.",
                    evidence={
                        "memory_loss": latest.get("memory_loss"),
                        "nearest_memory_distance": latest.get("nearest_memory_distance"),
                        "prediction_loss": latest.get("prediction_loss"),
                        "reconstruction_loss": latest.get("reconstruction_loss"),
                        "latent_norm": latest.get("latent_norm"),
                    },
                    baseline_version=baseline_version,
                    observed_at=observed_at,
                )
            )

        drift_events = self._dict(learning.get("metrics")).get("drift_events", learning.get("drift_events", []))
        if isinstance(drift_events, list):
            for index, drift_event in enumerate(drift_events[-5:]):
                if not isinstance(drift_event, dict):
                    continue
                timestamp = self._float(drift_event.get("timestamp", now), now)
                score = min(1.0, abs(self._float(recursive_state.get("drift", recursive_state.get("loss_delta", 0.0)), 0.0)) + 0.35)
                events.append(
                    NoveltyEvent(
                        event_id=f"learning-drift-{int(timestamp)}-{index}",
                        source="learning_engine",
                        signal_type=str(drift_event.get("type", "drift")),
                        novelty_score=round(score, 6),
                        severity=self._severity(score),
                        summary="Learning engine reported distribution drift against the active baseline.",
                        evidence={k: v for k, v in drift_event.items() if k != "raw"},
                        baseline_version=baseline_version,
                        observed_at=timestamp,
                    )
                )
        return events

    def _adjudicate(
        self,
        signature: str,
        events: List[NoveltyEvent],
        phi: float,
        sigma: float,
        drift: float,
        now: float,
    ) -> Optional[Z3Decision]:
        latest_event = events[0] if events else None
        severe_event = latest_event and latest_event.novelty_score >= self.severe_threshold
        baseline_changed = self._last_signature is not None and signature != self._last_signature
        coherence_loss = phi < 0.42 or sigma > 0.72 or abs(drift) > 0.18

        if severe_event or (baseline_changed and coherence_loss):
            self._baseline_version += 1
            self._baseline_id = f"z3-baseline-{self._baseline_version}"
            reason = "Novelty exceeded adjudication threshold; baseline advanced for future Z-prime boot state."
            if not severe_event:
                reason = "Coherence drift changed the baseline signature; Z3 advanced the public baseline."
            return Z3Decision(
                decision_id=f"z3-decision-{int(now)}-update",
                action="update_baseline",
                reason=reason,
                confidence=0.86 if severe_event else 0.72,
                baseline_version=self._baseline_version,
                linked_event_id=latest_event.event_id if latest_event else None,
                created_at=now,
            )

        if latest_event and latest_event.novelty_score >= self.novelty_threshold:
            return Z3Decision(
                decision_id=f"z3-decision-{int(now)}-hold",
                action="hold_baseline",
                reason="Novelty is visible but below baseline-update threshold; Z3 is watching without rewriting normal.",
                confidence=0.68,
                baseline_version=self._baseline_version,
                linked_event_id=latest_event.event_id,
                created_at=now,
            )
        return None

    @staticmethod
    def _dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _severity(score: float) -> str:
        if score >= 0.85:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _coherence_label(phi: float) -> str:
        if phi < 0.3:
            return "critical"
        if phi < 0.5:
            return "strained"
        if phi < 0.7:
            return "stable"
        return "coherent"

    @staticmethod
    def _regime(phi: float, sigma: float) -> str:
        if phi < 0.35 or sigma > 0.75:
            return "fragmented"
        if phi > 0.7 and sigma < 0.4:
            return "coherent"
        return "adaptive"

    def _current_novelty_pressure(
        self, world_model: Dict[str, Any], learning: Dict[str, Any], recursive_state: Dict[str, Any]
    ) -> float:
        latest = self._dict(world_model.get("latest"))
        signals = [
            self._float(latest.get("novelty", latest.get("memory_loss", 0.0)), 0.0),
            abs(self._float(recursive_state.get("loss_delta", 0.0), 0.0)),
            self._float(recursive_state.get("world_model_memory_loss", 0.0), 0.0),
        ]
        if learning.get("drift_events"):
            signals.append(0.45)
        return round(min(1.0, max(signals or [0.0])), 6)

    def _watch_targets(
        self,
        metrics: Dict[str, Any],
        world_model: Dict[str, Any],
        learning: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        recursive_state: Dict[str, Any],
    ) -> List[str]:
        targets: List[str] = []
        latest = self._dict(world_model.get("latest"))
        if self._float(latest.get("novelty", 0.0), 0.0) >= self.novelty_threshold:
            targets.append("world_model_memory_distance")
        if abs(self._float(recursive_state.get("loss_delta", 0.0), 0.0)) > 0.05:
            targets.append("recursive_coherence_loss_delta")
        if learning.get("drift_events"):
            targets.append("learning_distribution_drift")
        if self._float(metrics.get("prediction_accuracy", 0.0), 0.0) < 0.45 and predictions:
            targets.append("prediction_accuracy")
        if not targets:
            targets.append("coherence_baseline")
        return targets[:5]

    @staticmethod
    def _baseline_signature(phi: float, sigma: float, drift: float, regime: str, watch_targets: List[str]) -> str:
        return f"{round(phi, 2)}:{round(sigma, 2)}:{round(drift, 2)}:{regime}:{','.join(watch_targets[:2])}"

    @staticmethod
    def _baseline_summary(phi: float, sigma: float, drift: float, regime: str) -> str:
        return (
            f"Z3 baseline is {regime} with coherence {phi:.3f}, noise {sigma:.3f}, "
            f"and drift {drift:+.3f}."
        )

    def _public_baseline_metrics(
        self,
        metrics: Dict[str, Any],
        world_model: Dict[str, Any],
        resonant_memory: Dict[str, Any],
        recursive_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        resonance_metrics = self._dict(resonant_memory.get("metrics"))
        return {
            "total_observations": int(self._float(metrics.get("total_observations", 0), 0.0)),
            "world_model_iteration": int(self._float(world_model.get("iteration", 0), 0.0)),
            "average_recent_loss": round(self._float(world_model.get("average_recent_loss", 0.0), 0.0), 6),
            "recursive_loss": round(self._float(recursive_state.get("coherence_loss", 0.0), 0.0), 6),
            "resonant_memory_rings": int(self._float(resonance_metrics.get("rings", 0), 0.0)),
        }
