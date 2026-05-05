"""
Z3 Adjudicator
==============
Explicit update/hold/reject gate for the organism-level baseline.

The adjudicator is the immune boundary of Z3: novelty alone cannot rewrite the
organism. A signal must carry enough coherence and stability to be trusted.
"""
from __future__ import annotations

from typing import List, Optional

from z3_contracts import NoveltyEvent, Z3Decision, Z3EvidenceScore, Z3Transition


class Z3Adjudicator:
    """Scores novelty evidence and returns a Z3 baseline decision."""

    def __init__(
        self,
        novelty_threshold: float = 0.35,
        update_threshold: float = 0.68,
        reject_noise_threshold: float = 0.78,
    ) -> None:
        self.novelty_threshold = float(novelty_threshold)
        self.update_threshold = float(update_threshold)
        self.reject_noise_threshold = float(reject_noise_threshold)

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
    ) -> tuple[Z3Decision, Optional[Z3Transition], int]:
        """Return decision, optional transition record, and resulting baseline version."""
        latest_event = events[0] if events else None
        score = self.score_event(latest_event, phi=phi, sigma=sigma, drift=drift)

        action = "observe"
        reason = "No novelty above threshold; Z3 keeps the current baseline and continues watching."
        confidence = max(0.55, score.trust)
        new_version = baseline_version

        if latest_event is None or latest_event.novelty_score < self.novelty_threshold:
            action = "observe"
        elif score.noise >= self.reject_noise_threshold and score.coherence < 0.42:
            action = "reject_noise"
            reason = "Novelty is present, but coherence is too low and noise is too high; Z3 refuses baseline mutation."
            confidence = max(0.7, min(0.95, score.noise))
        elif score.trust >= self.update_threshold and (signature_changed or latest_event.novelty_score >= self.update_threshold):
            action = "update_baseline"
            new_version = baseline_version + 1
            reason = "Novelty carried enough coherence and stability to become the next Z3 baseline."
            confidence = min(0.98, max(0.75, score.trust))
        elif latest_event.novelty_score >= self.novelty_threshold:
            action = "hold_baseline"
            reason = "Novelty is visible but not trusted enough to rewrite normal; Z3 holds baseline and watches."
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

    def score_event(self, event: Optional[NoveltyEvent], *, phi: float, sigma: float, drift: float) -> Z3EvidenceScore:
        novelty = float(event.novelty_score) if event else 0.0
        coherence = self._clamp(phi)
        noise = self._clamp(sigma)
        drift_pressure = self._clamp(abs(drift))
        stability = self._clamp((0.55 * coherence) + (0.35 * (1.0 - noise)) + (0.10 * (1.0 - drift_pressure)))
        trust = self._clamp((0.45 * novelty) + (0.35 * coherence) + (0.20 * stability) - (0.25 * noise))

        if novelty < self.novelty_threshold:
            recommendation = "observe"
            rationale = "Novelty is below the public Z3 adjudication threshold."
        elif noise >= self.reject_noise_threshold and coherence < 0.42:
            recommendation = "reject_noise"
            rationale = "The signal looks more like incoherent noise than trustworthy novelty."
        elif trust >= self.update_threshold:
            recommendation = "update_baseline"
            rationale = "Novelty, coherence, and stability are jointly strong enough for baseline mutation."
        else:
            recommendation = "hold_baseline"
            rationale = "Novelty is real but trust is below update threshold."

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
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
