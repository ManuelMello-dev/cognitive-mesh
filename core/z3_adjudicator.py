"""
Z3 Adjudicator
==============
Explicit update/hold/reject gate for the organism-level baseline.

The adjudicator is the immune boundary of Z3: novelty alone cannot rewrite the
organism. A signal must carry enough coherence, stability, drift control, and
salience memory to be trusted.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from z3_contracts import NoveltyEvent, Z3Decision, Z3EvidenceScore, Z3Transition


class Z3Adjudicator:
    """Scores novelty evidence and returns a Z3 baseline decision."""

    def __init__(
        self,
        novelty_threshold: float = 0.35,
        update_threshold: float = 0.68,
        reject_noise_threshold: float = 0.78,
        coherence_gate_threshold: float = 0.35,
    ) -> None:
        self.novelty_threshold = float(novelty_threshold)
        self.update_threshold = float(update_threshold)
        self.reject_noise_threshold = float(reject_noise_threshold)
        self.coherence_gate_threshold = float(coherence_gate_threshold)

    def adjudicate(
        self,
        *,
        events: List[NoveltyEvent],
        phi: float,
        sigma: float,
        drift: float,
        baseline_version: int,
        signature_changed: bool,
        now: float,
        evidence_frame: Optional[Dict[str, Any]] = None,
    ) -> tuple[Z3Decision, Optional[Z3Transition], int]:
        """Return decision, optional transition record, and resulting baseline version."""
        latest_event = events[0] if events else None
        score = self.score_event(latest_event, phi=phi, sigma=sigma, drift=drift, evidence_frame=evidence_frame)

        action = "observe"
        reason = "No trusted novelty above threshold; Z3 keeps the current baseline and continues watching."
        confidence = max(0.55, score.trust)
        new_version = baseline_version

        if latest_event is None or latest_event.novelty_score < self.novelty_threshold:
            action = "observe"
        elif score.noise >= self.reject_noise_threshold and score.coherence < 0.42:
            action = "reject_noise"
            reason = "Novelty is present, but coherence is too low and noise is too high; Z3 refuses baseline mutation."
            confidence = max(0.7, min(0.95, score.noise))
        elif not score.gate_open:
            action = "hold_baseline"
            reason = "Novelty is visible, but the coherence gate is closed; Z3 compresses the event into memory without rewriting normal."
            confidence = min(0.88, max(0.62, score.coherence))
        elif score.trust >= self.update_threshold and (signature_changed or latest_event.novelty_score >= self.update_threshold):
            action = "update_baseline"
            new_version = baseline_version + 1
            reason = "Novelty carried enough coherence, stability, and salience memory to become the next Z3 baseline."
            confidence = min(0.98, max(0.75, score.trust))
        elif latest_event.novelty_score >= self.novelty_threshold:
            action = "hold_baseline"
            reason = "Novelty is coherent enough to watch but not trusted enough to rewrite normal; Z3 holds baseline and accumulates evidence."
            confidence = min(0.9, max(0.6, 1.0 - abs(score.trust - self.update_threshold)))

        decision = Z3Decision(
            decision_id=f"z3-decision-{int(now)}-{action}",
            action=action,
            reason=reason,
            confidence=round(confidence, 6),
            baseline_version=new_version,
            linked_event_id=latest_event.event_id if latest_event else None,
            evidence_score=score,
            created_at=now,
        )

        transition = None
        if action in {"update_baseline", "hold_baseline", "reject_noise"}:
            transition = Z3Transition(
                transition_id=f"z3-transition-{int(now)}-{baseline_version}-to-{new_version}",
                from_baseline_version=baseline_version,
                to_baseline_version=new_version,
                action=action,
                decision_id=decision.decision_id,
                linked_event_id=decision.linked_event_id,
                created_at=now,
            )
        return decision, transition, new_version

    def score_event(
        self,
        event: Optional[NoveltyEvent],
        *,
        phi: float,
        sigma: float,
        drift: float,
        evidence_frame: Optional[Dict[str, Any]] = None,
    ) -> Z3EvidenceScore:
        frame = evidence_frame or {}
        latest = frame.get("latest") if isinstance(frame.get("latest"), dict) else {}
        memory = frame.get("memory") if isinstance(frame.get("memory"), dict) else {}

        novelty = self._clamp(latest.get("novelty", event.novelty_score if event else 0.0))
        coherence = self._clamp(latest.get("coherence", phi))
        local_coherence = self._clamp(latest.get("local_coherence", coherence))
        noise = self._clamp(latest.get("noise", sigma))
        drift_pressure = self._clamp(latest.get("drift_pressure", abs(drift)))
        gate_open = bool(latest.get("trusted_gate_open", latest.get("gate_open", False)))
        memory_salience = self._clamp(memory.get("salience_peak", latest.get("salience", 0.0)))

        stability = self._clamp(
            (0.46 * coherence)
            + (0.26 * local_coherence)
            + (0.18 * (1.0 - noise))
            + (0.10 * (1.0 - drift_pressure))
        )
        trust = self._clamp(
            (0.42 * novelty)
            + (0.26 * coherence)
            + (0.16 * stability)
            + (0.12 * memory_salience)
            + (0.12 if gate_open else 0.0)
            - (0.18 * noise)
            - (0.05 * drift_pressure)
        )

        if novelty < self.novelty_threshold:
            recommendation = "observe"
            rationale = "Novelty is below the public Z3 adjudication threshold."
        elif noise >= self.reject_noise_threshold and coherence < 0.42:
            recommendation = "reject_noise"
            rationale = "The signal looks more like incoherent noise than trustworthy novelty."
        elif not gate_open:
            recommendation = "hold_baseline"
            rationale = "The novelty gate is closed because coherence, stability, or noise control is insufficient."
        elif trust >= self.update_threshold:
            recommendation = "update_baseline"
            rationale = "Novelty, coherence, stability, and salience memory are jointly strong enough for baseline mutation."
        else:
            recommendation = "hold_baseline"
            rationale = "Novelty is real and coherent, but trust is below update threshold."

        return Z3EvidenceScore(
            event_id=event.event_id if event else "none",
            novelty=round(novelty, 6),
            coherence=round(coherence, 6),
            stability=round(stability, 6),
            noise=round(noise, 6),
            drift_pressure=round(drift_pressure, 6),
            trust=round(trust, 6),
            recommendation=recommendation,
            rationale=rationale,
            gate_open=gate_open,
            local_coherence=round(local_coherence, 6),
            memory_salience=round(memory_salience, 6),
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
