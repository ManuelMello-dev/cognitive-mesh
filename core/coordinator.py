"""
Mesh Coordinator
================
The central coordination layer (Principle 3 & 4).

Responsibilities:
- Receives structured outputs from all specialized modules
- Resolves signal conflicts using weighted voting
- Maintains CoordinatorState as the Z³ identity anchor
- Tracks drift and applies stability corrections
- Does NOT produce natural language (Principle 2)
- Does NOT perform any module-specific cognitive work (Principle 1)

This replaces the monolithic CognitiveIntelligentSystem.process_observation().
"""
import logging
import time
import math
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from contracts import (
    AbstractionOutput,
    ReasoningOutput,
    PredictionOutput,
    EEGOutput,
    CrossDomainOutput,
    GoalOutput,
    LearningOutput,
    CoordinatorState,
)

logger = logging.getLogger(__name__)


class MeshCoordinator:
    """
    The central coordination node.

    Receives outputs from all modules, resolves conflicts,
    applies weighting, and maintains the Z³ CoordinatorState.

    Principle 3: Central Coordination Layer.
    Principle 4: Identity Continuity (Z³ Anchor).
    Principle 6: Minimal Cross-Contamination.
    """

    # Signal weights — coordinator decides how much each module influences PHI
    _MODULE_WEIGHTS: Dict[str, float] = {
        "abstraction": 0.20,
        "reasoning":   0.20,
        "prediction":  0.25,
        "eeg":         0.15,
        "cross_domain":0.10,
        "goals":       0.05,
        "learning":    0.05,
    }

    # Drift threshold — if drift_vector exceeds this, apply identity correction
    _DRIFT_THRESHOLD: float = 0.15

    def __init__(self) -> None:
        self.state = CoordinatorState()
        self._phi_history: List[float] = []
        self._sigma_history: List[float] = []
        logger.info("MeshCoordinator initialized — Z³ anchor ready")

    # ─────────────────────────────────────────────
    # Primary entry point — called once per cognitive cycle
    # ─────────────────────────────────────────────

    def coordinate(
        self,
        abstractions: List[AbstractionOutput],
        rules: List[ReasoningOutput],
        predictions: List[PredictionOutput],
        eeg: Optional[EEGOutput],
        cross_domain_transfers: List[CrossDomainOutput],
        active_goals: List[GoalOutput],
        learning_patterns: List[LearningOutput],
    ) -> CoordinatorState:
        """
        Accept structured outputs from all modules, resolve conflicts,
        update the Z³ state, and return the new CoordinatorState.

        This is the only method external callers should invoke.
        """
        self.state.iteration += 1
        self.state.cycle_timestamp = time.time()

        # 1. Store raw module outputs
        self.state.abstractions = abstractions
        self.state.rules = rules
        self.state.predictions = predictions
        self.state.eeg = eeg
        self.state.cross_domain_transfers = cross_domain_transfers
        self.state.active_goals = active_goals
        self.state.learning_patterns = learning_patterns

        # 2. Compute weighted signals from each module
        self.state.weighted_signals = self._compute_weighted_signals()

        # 3. Resolve conflicts between contradictory signals
        self.state.resolved_conflicts = self._resolve_conflicts()

        # 4. Update PHI and SIGMA from EEG + prediction accuracy
        self._update_coherence(eeg, predictions)

        # 5. Update Z³ identity state
        self._update_z3_state()

        # 6. Check and correct drift
        self._check_drift()

        return self.state

    # ─────────────────────────────────────────────
    # Internal coordination methods
    # ─────────────────────────────────────────────

    def _compute_weighted_signals(self) -> Dict[str, float]:
        """
        Compute a scalar signal strength for each module's output.
        Returns a dict of {module_name: weighted_signal_strength}.
        """
        signals: Dict[str, float] = {}

        # Abstraction signal: mean confidence of formed concepts
        if self.state.abstractions:
            raw = sum(a.confidence for a in self.state.abstractions) / len(self.state.abstractions)
            signals["abstraction"] = raw * self._MODULE_WEIGHTS["abstraction"]
        else:
            signals["abstraction"] = 0.0

        # Reasoning signal: mean confidence of active rules
        if self.state.rules:
            raw = sum(r.confidence for r in self.state.rules) / len(self.state.rules)
            signals["reasoning"] = raw * self._MODULE_WEIGHTS["reasoning"]
        else:
            signals["reasoning"] = 0.0

        # Prediction signal: mean confidence of active predictions
        if self.state.predictions:
            raw = sum(p.confidence for p in self.state.predictions) / len(self.state.predictions)
            signals["prediction"] = raw * self._MODULE_WEIGHTS["prediction"]
        else:
            signals["prediction"] = 0.0

        # EEG signal: PHI directly
        if self.state.eeg:
            signals["eeg"] = self.state.eeg.phi * self._MODULE_WEIGHTS["eeg"]
        else:
            signals["eeg"] = 0.0

        # Cross-domain signal: mean confidence of transfers
        if self.state.cross_domain_transfers:
            raw = sum(t.confidence for t in self.state.cross_domain_transfers) / len(self.state.cross_domain_transfers)
            signals["cross_domain"] = raw * self._MODULE_WEIGHTS["cross_domain"]
        else:
            signals["cross_domain"] = 0.0

        # Goal signal: mean priority of active goals
        if self.state.active_goals:
            raw = sum(g.priority for g in self.state.active_goals) / len(self.state.active_goals)
            signals["goals"] = raw * self._MODULE_WEIGHTS["goals"]
        else:
            signals["goals"] = 0.0

        # Learning signal: mean pattern strength
        if self.state.learning_patterns:
            raw = sum(p.strength for p in self.state.learning_patterns) / len(self.state.learning_patterns)
            signals["learning"] = raw * self._MODULE_WEIGHTS["learning"]
        else:
            signals["learning"] = 0.0

        return signals

    def _resolve_conflicts(self) -> List[str]:
        """
        Detect and resolve contradictory signals between modules.
        Returns a list of conflict resolution descriptions.
        Principle 6: Emotion does not override logic. Memory does not generate language.
        """
        conflicts: List[str] = []

        # Conflict: prediction says UP but EEG coherence is critically low
        if self.state.eeg and self.state.eeg.phi < 0.3:
            up_predictions = [p for p in self.state.predictions if p.direction == "up"]
            if up_predictions:
                # Suppress low-coherence UP predictions
                for p in up_predictions:
                    p.confidence *= 0.5
                conflicts.append(
                    f"suppressed {len(up_predictions)} UP predictions: EEG phi={self.state.eeg.phi:.3f} < 0.3"
                )

        # Conflict: goal priority > 0.8 but no supporting rules
        high_priority_goals = [g for g in self.state.active_goals if g.priority > 0.8]
        if high_priority_goals and not self.state.rules:
            for g in high_priority_goals:
                g.priority *= 0.7
            conflicts.append(
                f"reduced {len(high_priority_goals)} high-priority goals: no supporting rules"
            )

        # Conflict: cross-domain transfer confidence < 0.4 but high performance_gain claimed
        suspect_transfers = [
            t for t in self.state.cross_domain_transfers
            if t.confidence < 0.4 and t.performance_gain > 0.3
        ]
        if suspect_transfers:
            for t in suspect_transfers:
                t.performance_gain *= t.confidence  # discount gain by confidence
            conflicts.append(
                f"discounted {len(suspect_transfers)} cross-domain transfers: low confidence"
            )

        return conflicts

    def _update_coherence(
        self,
        eeg: Optional[EEGOutput],
        predictions: List[PredictionOutput],
    ) -> None:
        """
        Update PHI (global coherence) and SIGMA (noise) from EEG + prediction data.
        PHI is the primary Z³ health signal.
        """
        phi_components: List[float] = []

        # EEG component
        if eeg:
            phi_components.append(eeg.phi)

        # Prediction confidence component
        if predictions:
            avg_conf = sum(p.confidence for p in predictions) / len(predictions)
            phi_components.append(avg_conf)

        # Rule confidence component
        if self.state.rules:
            avg_rule = sum(r.confidence for r in self.state.rules) / len(self.state.rules)
            phi_components.append(avg_rule)

        if phi_components:
            new_phi = sum(phi_components) / len(phi_components)
            # Exponential moving average for stability
            alpha = 0.3
            self.state.phi = alpha * new_phi + (1 - alpha) * self.state.phi
        else:
            # Decay toward 0.5 (neutral) when no data
            self.state.phi = 0.9 * self.state.phi + 0.1 * 0.5

        # SIGMA: noise = inverse of concept confidence stability
        if self.state.abstractions:
            conf_variance = self._variance([a.confidence for a in self.state.abstractions])
            self.state.sigma = min(1.0, conf_variance * 4.0)
        else:
            self.state.sigma = 0.5

        self._phi_history.append(self.state.phi)
        self._sigma_history.append(self.state.sigma)

        # Keep history bounded
        if len(self._phi_history) > 200:
            self._phi_history = self._phi_history[-200:]
        if len(self._sigma_history) > 200:
            self._sigma_history = self._sigma_history[-200:]

    def _update_z3_state(self) -> None:
        """
        Update the Z³ identity state dict.
        This is the persistent anchor — it does not reset between cycles.
        Principle 4: Identity Continuity.
        """
        s = self.state
        s.z_cubed_state.update({
            "iteration": s.iteration,
            "phi": round(s.phi, 4),
            "sigma": round(s.sigma, 4),
            "active_concepts": len(s.abstractions),
            "active_rules": len(s.rules),
            "active_predictions": len(s.predictions),
            "active_goals": len(s.active_goals),
            "cross_domain_transfers": len(s.cross_domain_transfers),
            "learning_patterns": len(s.learning_patterns),
            "weighted_signal_sum": round(sum(s.weighted_signals.values()), 4),
            "conflicts_resolved": len(s.resolved_conflicts),
            "cycle_timestamp": s.cycle_timestamp,
        })

    def _check_drift(self) -> None:
        """
        Compute drift vector from PHI history.
        If drift exceeds threshold, apply identity correction (pull PHI back toward baseline).
        Principle 4: Z³ Anchor drift control.
        """
        if len(self._phi_history) < 10:
            self.state.drift_vector = 0.0
            return

        recent = self._phi_history[-10:]
        baseline = self._phi_history[-50:-10] if len(self._phi_history) >= 50 else self._phi_history[:-10]

        if not baseline:
            self.state.drift_vector = 0.0
            return

        recent_mean = sum(recent) / len(recent)
        baseline_mean = sum(baseline) / len(baseline)
        drift = recent_mean - baseline_mean
        self.state.drift_vector = round(drift, 4)

        if abs(drift) > self._DRIFT_THRESHOLD:
            # Apply correction: pull PHI back toward baseline
            correction = -drift * 0.2
            self.state.phi = max(0.0, min(1.0, self.state.phi + correction))
            logger.warning(
                f"Z³ drift detected: {drift:+.3f} — correction applied: {correction:+.3f}"
            )

    @staticmethod
    def _variance(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    # ─────────────────────────────────────────────
    # Serialization — for state cache and HTTP layer
    # ─────────────────────────────────────────────

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Return the full CoordinatorState as a plain dict for the state cache.
        This is the ONLY method the HTTP layer should read from.
        """
        s = self.state
        return {
            "iteration": s.iteration,
            "phi": s.phi,
            "sigma": s.sigma,
            "drift_vector": s.drift_vector,
            "cycle_timestamp": s.cycle_timestamp,
            "z_cubed_state": s.z_cubed_state,
            "weighted_signals": s.weighted_signals,
            "resolved_conflicts": s.resolved_conflicts,
            "abstractions": [asdict(a) for a in s.abstractions],
            "rules": [asdict(r) for r in s.rules],
            "predictions": [asdict(p) for p in s.predictions],
            "eeg": asdict(s.eeg) if s.eeg else None,
            "cross_domain_transfers": [asdict(t) for t in s.cross_domain_transfers],
            "active_goals": [asdict(g) for g in s.active_goals],
            "learning_patterns": [asdict(p) for p in s.learning_patterns],
        }
