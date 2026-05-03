"""
Safe Agency Layer
=================
Bounded internal agency for the cognitive mesh.

This module gives the mesh a small action vocabulary, a planner that scores
candidate actions from the current world-model/recursive state, a measurable
runtime task harness, regression gates, and memory-governed retrieval signals.
It deliberately does not execute arbitrary code or external side effects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class AgencyAction:
    action_id: str
    description: str
    expected_gain: float
    risk: float
    reason: str

    @property
    def score(self) -> float:
        return self.expected_gain - self.risk

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = round(self.score, 6)
        return data


class AgencyLayer:
    """Safe internal action/planning layer for the mesh core."""

    ACTIONS = {
        "stabilize_learning": "Lower learning rate and enable convergence/introspection.",
        "increase_plasticity": "Increase learning rate when predictive loss rises.",
        "tighten_abstraction": "Raise abstraction threshold to reduce concept explosion.",
        "loosen_abstraction": "Lower abstraction threshold when concepts are too sparse.",
        "increase_exploration": "Raise goal exploration rate.",
        "activate_knowledge_transfer": "Enable cross-domain transfer machinery.",
        "request_more_data": "Record an internal request for more observations.",
    }

    def __init__(self, evaluation_window: int = 20, regression_tolerance: float = 0.03) -> None:
        self.evaluation_window = int(evaluation_window)
        self.regression_tolerance = float(regression_tolerance)
        self.action_history: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []
        self.regression_events: List[Dict[str, Any]] = []
        self.internal_requests: List[Dict[str, Any]] = []

    def evaluate_tasks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert runtime state into measurable task scores."""
        rec = state.get("recursive_state", {}) or {}
        wm = state.get("world_model", {}) or {}
        latest = wm.get("latest", {}) or {}
        control = rec.get("control", {}) or {}

        recursive_loss = float(rec.get("coherence_loss", 0.5) or 0.5)
        wm_loss = float(rec.get("world_model_loss", latest.get("total_loss", 0.5)) or 0.5)
        pred_loss = float(rec.get("world_model_prediction_loss", latest.get("prediction_loss", 0.5)) or 0.5)
        recon_loss = float(rec.get("world_model_reconstruction_loss", latest.get("reconstruction_loss", 0.5)) or 0.5)
        memory_loss = float(rec.get("world_model_memory_loss", latest.get("memory_loss", 1.0)) or 1.0)
        concept_pressure = float(control.get("concept_pressure", 0.5) or 0.5)
        concept_pressure_penalty = min(1.0, abs(concept_pressure - 0.35))

        scores = {
            "recursive_coherence": 1.0 - _clamp(recursive_loss),
            "world_model_prediction": 1.0 - _clamp(pred_loss),
            "world_model_compression": 1.0 - _clamp(recon_loss),
            "memory_recall": 1.0 - _clamp(memory_loss),
            "concept_pressure": 1.0 - _clamp(concept_pressure_penalty),
            "world_model_total": 1.0 - _clamp(wm_loss),
        }
        score = sum(scores.values()) / max(len(scores), 1)
        task_state = {
            "timestamp": time.time(),
            "score": round(score, 6),
            "scores": {k: round(v, 6) for k, v in scores.items()},
            "losses": {
                "recursive_loss": round(recursive_loss, 6),
                "world_model_loss": round(wm_loss, 6),
                "prediction_loss": round(pred_loss, 6),
                "reconstruction_loss": round(recon_loss, 6),
                "memory_loss": round(memory_loss, 6),
                "concept_pressure": round(concept_pressure, 6),
            },
        }
        self.task_history.append(task_state)
        self.task_history = self.task_history[-500:]
        return task_state

    def retrieve_memory_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a planning-relevant memory signal from world and resonant memory."""
        rec = state.get("recursive_state", {}) or {}
        wm = state.get("world_model", {}) or {}
        latest = wm.get("latest", {}) or {}
        resonant = state.get("resonant_memory", {}) or {}
        r_metrics = resonant.get("metrics", {}) if isinstance(resonant, dict) else {}

        wm_memory_loss = float(rec.get("world_model_memory_loss", latest.get("memory_loss", 1.0)) or 1.0)
        wm_novelty = float(latest.get("novelty", wm_memory_loss) or wm_memory_loss)
        resonant_conf = float(r_metrics.get("last_reconstruction_confidence", 0.0) or 0.0)
        average_resonance = float(r_metrics.get("average_resonance", 0.0) or 0.0)
        memory_alignment = _clamp((1.0 - wm_memory_loss) * 0.6 + resonant_conf * 0.25 + average_resonance * 0.15)

        return {
            "memory_alignment": round(memory_alignment, 6),
            "novelty": round(_clamp(wm_novelty), 6),
            "world_model_memory_loss": round(_clamp(wm_memory_loss), 6),
            "resonant_reconstruction_confidence": round(_clamp(resonant_conf), 6),
            "average_resonance": round(_clamp(average_resonance), 6),
        }

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Score safe candidate actions from current state and return a plan."""
        task = self.evaluate_tasks(state)
        memory = self.retrieve_memory_context(state)
        rec = state.get("recursive_state", {}) or {}
        control = rec.get("control", {}) or {}
        losses = task["losses"]

        candidates: List[AgencyAction] = []
        if losses["recursive_loss"] > 0.45 or float(rec.get("sigma", 0.5) or 0.5) > 0.45:
            candidates.append(AgencyAction("stabilize_learning", "Stabilize coherence/noise", 0.18, 0.04, "recursive loss or sigma elevated"))
        if losses["prediction_loss"] > 0.45:
            candidates.append(AgencyAction("increase_plasticity", "Improve next-state adaptation", 0.16, 0.07, "world-model prediction loss elevated"))
        if losses["concept_pressure"] > 0.55 or float(control.get("concept_pressure", 0.0) or 0.0) > 0.70:
            candidates.append(AgencyAction("tighten_abstraction", "Reduce concept explosion", 0.14, 0.03, "concept pressure too high"))
        if float(control.get("concept_pressure", 0.5) or 0.5) < 0.10 and int((state.get("metrics", {}) or {}).get("total_observations", 0) or 0) > 30:
            candidates.append(AgencyAction("loosen_abstraction", "Allow new concepts", 0.10, 0.04, "concept pressure too low"))
        if memory["novelty"] > 0.65 or memory["memory_alignment"] < 0.35:
            candidates.append(AgencyAction("increase_exploration", "Explore under-modeled regions", 0.12, 0.05, "high novelty or weak memory alignment"))
        if int((state.get("metrics", {}) or {}).get("total_concepts", 0) or 0) > 5:
            candidates.append(AgencyAction("activate_knowledge_transfer", "Permit transfer checks", 0.07, 0.02, "enough concepts exist"))
        if int((state.get("metrics", {}) or {}).get("total_observations", 0) or 0) < 25:
            candidates.append(AgencyAction("request_more_data", "Need more observations", 0.08, 0.01, "insufficient observations"))

        if not candidates:
            candidates.append(AgencyAction("request_more_data", "Continue observation gathering", 0.03, 0.01, "no intervention clearly beneficial"))

        candidates.sort(key=lambda a: a.score, reverse=True)
        selected = candidates[0]
        return {
            "timestamp": time.time(),
            "selected_action": selected.to_dict(),
            "candidates": [c.to_dict() for c in candidates],
            "task_state": task,
            "memory_context": memory,
        }

    def execute(self, action_id: str, core: Any) -> Dict[str, Any]:
        """Execute a bounded internal action against the core."""
        before = self._snapshot_controls(core)
        changes: List[str] = []
        cs = core.cognitive_system

        if action_id == "stabilize_learning":
            old = float(getattr(cs.learning_engine, "learning_rate", 0.01))
            new = _clamp(old * 0.985, 0.001, 0.05)
            cs.learning_engine.learning_rate = new
            core._toggles["concept_convergence"] = True
            core._toggles["deep_introspection"] = True
            changes.extend([f"learning_rate {old:.5f}->{new:.5f}", "concept_convergence=True", "deep_introspection=True"])
        elif action_id == "increase_plasticity":
            old = float(getattr(cs.learning_engine, "learning_rate", 0.01))
            new = _clamp((old * 1.02) + 0.0002, 0.001, 0.05)
            cs.learning_engine.learning_rate = new
            changes.append(f"learning_rate {old:.5f}->{new:.5f}")
        elif action_id == "tighten_abstraction":
            old = float(getattr(cs.abstraction, "similarity_threshold", 0.85))
            new = _clamp(old + 0.005, 0.50, 0.97)
            cs.abstraction.similarity_threshold = new
            changes.append(f"abstraction_threshold {old:.3f}->{new:.3f}")
        elif action_id == "loosen_abstraction":
            old = float(getattr(cs.abstraction, "similarity_threshold", 0.85))
            new = _clamp(old - 0.005, 0.50, 0.97)
            cs.abstraction.similarity_threshold = new
            changes.append(f"abstraction_threshold {old:.3f}->{new:.3f}")
        elif action_id == "increase_exploration":
            old = float(getattr(cs.goals, "exploration_rate", 0.3))
            new = _clamp(old + 0.015, 0.05, 0.80)
            cs.goals.exploration_rate = new
            changes.append(f"exploration_rate {old:.3f}->{new:.3f}")
        elif action_id == "activate_knowledge_transfer":
            core._toggles["knowledge_transfer"] = True
            changes.append("knowledge_transfer=True")
        elif action_id == "request_more_data":
            req = {"timestamp": time.time(), "reason": "planner requested additional observations"}
            self.internal_requests.append(req)
            self.internal_requests = self.internal_requests[-100:]
            changes.append("internal_data_request_logged")
        else:
            return {"success": False, "error": f"unknown action {action_id}", "changes": []}

        after = self._snapshot_controls(core)
        event = {
            "timestamp": time.time(),
            "action_id": action_id,
            "success": True,
            "changes": changes,
            "before": before,
            "after": after,
            "baseline_task_score": (self.task_history[-1]["score"] if self.task_history else None),
            "status": "pending_regression_check",
        }
        self.action_history.append(event)
        self.action_history = self.action_history[-500:]
        return event

    def check_regression(self, state: Dict[str, Any], core: Any) -> Dict[str, Any]:
        """Evaluate pending actions and revert if task score regressed too much."""
        current = self.evaluate_tasks(state)
        checked = []
        for event in self.action_history:
            if event.get("status") != "pending_regression_check":
                continue
            baseline = event.get("baseline_task_score")
            if baseline is None:
                event["status"] = "accepted"
                continue
            if current["score"] + self.regression_tolerance < float(baseline):
                self._restore_controls(core, event.get("before", {}))
                event["status"] = "reverted"
                event["regression_score"] = current["score"]
                self.regression_events.append(dict(event))
            else:
                event["status"] = "accepted"
                event["regression_score"] = current["score"]
            checked.append({"action_id": event.get("action_id"), "status": event.get("status"), "score": current["score"]})
        self.regression_events = self.regression_events[-200:]
        return {"timestamp": time.time(), "current_task_score": current["score"], "checked": checked}

    def get_state(self) -> Dict[str, Any]:
        return {
            "actions_available": dict(self.ACTIONS),
            "last_plan": self.action_history[-1] if self.action_history else None,
            "action_history": self.action_history[-50:],
            "task_history": self.task_history[-50:],
            "regression_events": self.regression_events[-50:],
            "internal_requests": self.internal_requests[-20:],
        }

    def _snapshot_controls(self, core: Any) -> Dict[str, Any]:
        cs = core.cognitive_system
        return {
            "learning_rate": float(getattr(cs.learning_engine, "learning_rate", 0.01)),
            "abstraction_similarity_threshold": float(getattr(cs.abstraction, "similarity_threshold", 0.85)),
            "goal_exploration_rate": float(getattr(cs.goals, "exploration_rate", 0.3)),
            "toggles": dict(getattr(core, "_toggles", {})),
        }

    def _restore_controls(self, core: Any, snapshot: Dict[str, Any]) -> None:
        cs = core.cognitive_system
        if "learning_rate" in snapshot:
            cs.learning_engine.learning_rate = float(snapshot["learning_rate"])
        if "abstraction_similarity_threshold" in snapshot:
            cs.abstraction.similarity_threshold = float(snapshot["abstraction_similarity_threshold"])
        if "goal_exploration_rate" in snapshot:
            cs.goals.exploration_rate = float(snapshot["goal_exploration_rate"])
        if "toggles" in snapshot:
            core._toggles.update(dict(snapshot["toggles"]))
