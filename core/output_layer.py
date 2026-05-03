"""
Native Mesh Output Layer
========================
Deterministically renders cached coordinator state into human-readable text.
No external model calls, no API clients, and no network dependency.
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class OutputLayer:
    """Final native output node for mesh status/chat rendering."""

    def __init__(self, model: str = "native") -> None:
        self.model = model

    def render(
        self,
        user_query: str,
        coordinator_state: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Render a deterministic response from cached mesh state."""
        return self._render_native(user_query, coordinator_state)

    def render_status(self, coordinator_state: Dict[str, Any]) -> str:
        s = coordinator_state
        metrics = s.get("metrics", {}) if isinstance(s.get("metrics", {}), dict) else {}
        phi = s.get("phi", metrics.get("global_coherence_phi", 0.5))
        sigma = s.get("sigma", metrics.get("noise_level_sigma", 0.5))
        drift = s.get("drift_vector", 0.0)
        iteration = s.get("iteration", 0)
        n_concepts = len(s.get("abstractions", s.get("concepts", [])) or [])
        if isinstance(s.get("concepts"), dict):
            n_concepts = max(n_concepts, len(s.get("concepts", {})))
        n_rules = len(s.get("rules", {}) or {}) if isinstance(s.get("rules"), dict) else len(s.get("rules", []) or [])
        n_predictions = len(s.get("predictions", []) or [])
        n_goals = len(s.get("active_goals", s.get("goals", [])) or [])
        if isinstance(s.get("goals"), dict):
            n_goals = max(n_goals, len(s.get("goals", {})))
        conflicts = s.get("resolved_conflicts", []) or []

        coherence_label = (
            "CRITICAL" if phi < 0.3 else
            "LOW" if phi < 0.5 else
            "MODERATE" if phi < 0.7 else
            "HIGH"
        )
        drift_note = ""
        if abs(drift) > 0.15:
            direction = "upward" if drift > 0 else "downward"
            drift_note = f" | DRIFT {direction} ({drift:+.3f})"

        return (
            f"[Cycle {iteration}] PHI={phi:.3f} ({coherence_label}) "
            f"SIGMA={sigma:.3f}{drift_note} | "
            f"Concepts={n_concepts} Rules={n_rules} "
            f"Predictions={n_predictions} Goals={n_goals} "
            f"Conflicts={len(conflicts)}"
        )

    def _render_native(self, user_query: str, coordinator_state: Dict[str, Any]) -> str:
        query = (user_query or "").strip().lower()
        status = self.render_status(coordinator_state)
        summary = self._summarize_state(coordinator_state)

        if any(term in query for term in ["status", "health", "alive", "state"]):
            return status
        if any(term in query for term in ["learn", "learning", "memory", "pattern", "rule"]):
            metrics = coordinator_state.get("metrics", {})
            return (
                f"{status}\n"
                f"Learning snapshot: samples={metrics.get('samples_processed', 0)}, "
                f"patterns={metrics.get('patterns_discovered', 0)}, "
                f"rules={metrics.get('total_rules', 0)}, "
                f"goals={metrics.get('total_goals', 0)}."
            )
        if any(term in query for term in ["provider", "data", "feed", "stream", "entity"]):
            providers = coordinator_state.get("providers", {})
            if providers:
                provider_bits = []
                for provider in list(providers.values())[:8]:
                    provider_bits.append(f"{provider.get('name', '?')}={provider.get('status', 'unknown')}")
                return f"{status}\nProviders: " + ", ".join(provider_bits)
            metrics = coordinator_state.get("metrics", {})
            return f"{status}\nStreams tracked: {metrics.get('streams_tracked', metrics.get('symbols_tracked', 0))}."
        return f"{status}\n\nState summary:\n{summary}"

    @staticmethod
    def _summarize_state(state: Dict[str, Any]) -> str:
        lines = []
        metrics = state.get("metrics", {}) if isinstance(state.get("metrics", {}), dict) else {}
        phi = state.get("phi", metrics.get("global_coherence_phi", 0.5))
        sigma = state.get("sigma", metrics.get("noise_level_sigma", 0.5))
        lines.append(f"Iteration: {state.get('iteration', 0)}")
        lines.append(f"PHI (coherence): {phi:.4f}")
        lines.append(f"SIGMA (noise): {sigma:.4f}")
        lines.append(f"Drift vector: {state.get('drift_vector', 0.0):+.4f}")

        ws = state.get("weighted_signals", {})
        if ws:
            lines.append("Weighted signals: " + ", ".join(f"{k}={v:.3f}" for k, v in ws.items()))

        abstractions = state.get("abstractions", []) or []
        lines.append(f"Active abstractions: {len(abstractions)}")
        for a in abstractions[:5]:
            lines.append(f"  - [{a.get('domain','')}] {a.get('label','')} conf={a.get('confidence',0):.3f}")

        rules = state.get("rules", {}) or {}
        rule_values = list(rules.values()) if isinstance(rules, dict) else list(rules)
        lines.append(f"Rules: {len(rule_values)}")
        for r in rule_values[:5]:
            ants = r.get('antecedents', r.get('if', [])) if isinstance(r, dict) else []
            cons = r.get('consequent', r.get('then', '')) if isinstance(r, dict) else ''
            conf = r.get('confidence', 0) if isinstance(r, dict) else 0
            lines.append(f"  - {' & '.join(ants)} => {cons} conf={conf:.3f}")

        predictions = state.get("predictions", []) or []
        lines.append(f"Predictions: {len(predictions)}")
        for p in predictions[:5]:
            if isinstance(p, dict):
                lines.append(f"  - {p.get('symbol', p.get('entity_id', '?'))}: {p.get('type', p.get('direction', '?'))} conf={p.get('confidence', 0):.3f}")

        goals = state.get("goals", {}) or {}
        goal_values = list(goals.values()) if isinstance(goals, dict) else list(goals)
        lines.append(f"Goals: {len(goal_values)}")
        for g in goal_values[:5]:
            if isinstance(g, dict):
                lines.append(f"  - {g.get('description', g.get('goal', '?'))} priority={g.get('priority', 0):.3f}")

        return "\n".join(lines)
