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
        z3 = self._get_z3(coordinator_state)
        if z3:
            baseline = z3.get("baseline", {}) if isinstance(z3.get("baseline"), dict) else {}
            organism = z3.get("organism_state", {}) if isinstance(z3.get("organism_state"), dict) else {}
            decision = z3.get("last_decision", {}) if isinstance(z3.get("last_decision"), dict) else {}
            phi = float(baseline.get("phi", organism.get("coherence", 0.5)))
            sigma = float(baseline.get("sigma", organism.get("noise", 0.5)))
            drift = float(baseline.get("drift_vector", organism.get("drift", 0.0)))
            iteration = int(baseline.get("iteration", organism.get("cycle", 0)))
            state_label = str(organism.get("state", "stable")).upper()
            next_watch = z3.get("next_watch_target") or "coherence_baseline"
            return (
                f"[Z3 Cycle {iteration}] PHI={phi:.3f} ({state_label}) "
                f"SIGMA={sigma:.3f} Drift={drift:+.3f} | "
                f"Baseline=v{baseline.get('version', 1)} "
                f"Decision={decision.get('action', 'observe')} "
                f"Watch={next_watch} "
                f"NoveltyEvents={len(z3.get('novelty_events', []) or [])}"
            )

        s = coordinator_state
        metrics = s.get("metrics", {}) if isinstance(s.get("metrics", {}), dict) else {}
        phi = s.get("phi", metrics.get("global_coherence_phi", 0.5))
        sigma = s.get("sigma", metrics.get("noise_level_sigma", 0.5))
        drift = s.get("drift_vector", 0.0)
        iteration = s.get("iteration", 0)
        coherence_label = "CRITICAL" if phi < 0.3 else "LOW" if phi < 0.5 else "MODERATE" if phi < 0.7 else "HIGH"
        return f"[Cycle {iteration}] PHI={phi:.3f} ({coherence_label}) SIGMA={sigma:.3f} Drift={drift:+.3f}"

    def _render_native(self, user_query: str, coordinator_state: Dict[str, Any]) -> str:
        query = (user_query or "").strip().lower()
        status = self.render_status(coordinator_state)
        summary = self._summarize_state(coordinator_state)

        if any(term in query for term in ["status", "health", "alive", "state"]):
            return status
        if any(term in query for term in ["learn", "learning", "memory", "pattern", "rule"]):
            z3 = self._get_z3(coordinator_state)
            if z3:
                metrics = z3.get("public_metrics", {}) if isinstance(z3.get("public_metrics"), dict) else {}
                return (
                    f"{status}\n"
                    f"Z3 learning/memory view: concepts={metrics.get('concepts', 0)}, "
                    f"rules={metrics.get('rules', 0)}, "
                    f"world_model_loss={metrics.get('world_model_loss', 0)}, "
                    f"memory_reconstruction_confidence={metrics.get('memory_reconstruction_confidence', 0)}."
                )
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
    def _get_z3(state: Dict[str, Any]) -> Dict[str, Any]:
        z3 = state.get("z3", {}) if isinstance(state, dict) else {}
        return z3 if isinstance(z3, dict) else {}

    @staticmethod
    def _summarize_state(state: Dict[str, Any]) -> str:
        z3 = OutputLayer._get_z3(state)
        if z3:
            baseline = z3.get("baseline", {}) if isinstance(z3.get("baseline"), dict) else {}
            organism = z3.get("organism_state", {}) if isinstance(z3.get("organism_state"), dict) else {}
            metrics = z3.get("public_metrics", {}) if isinstance(z3.get("public_metrics"), dict) else {}
            decision = z3.get("last_decision", {}) if isinstance(z3.get("last_decision"), dict) else {}
            events = z3.get("novelty_events", []) or []
            lines = [
                baseline.get("summary", "Z3 baseline is active."),
                f"Organism state: {organism.get('state', 'stable')} | novelty_pressure={organism.get('novelty_pressure', 0)}",
                f"Public metrics: observations={metrics.get('observations', 0)}, concepts={metrics.get('concepts', 0)}, rules={metrics.get('rules', 0)}, prediction_accuracy={metrics.get('prediction_accuracy', 0)}",
                f"Last decision: {decision.get('action', 'observe')} — {decision.get('reason', 'No adjudication change pending.')}",
                f"Next watch target: {z3.get('next_watch_target') or 'coherence_baseline'}",
            ]
            if events:
                latest = events[0]
                if isinstance(latest, dict):
                    lines.append(
                        f"Latest novelty: {latest.get('severity', 'unknown')} {latest.get('signal_type', 'signal')} from {latest.get('source', 'unknown')} score={latest.get('novelty_score', 0)}."
                    )
            else:
                lines.append("Novelty feed: no compressed novelty event is currently above threshold.")
            return "\n".join(lines)
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
