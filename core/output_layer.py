"""
Mesh Output Layer
=================
The ONLY node in the mesh permitted to produce natural language (Principle 7).

Responsibilities:
- Receives the CoordinatorState from the Coordinator
- Renders it into coherent human-readable text for the chat interface
- Handles user queries by reading from CoordinatorState — never from raw modules
- Uses LLM (via OpenAI-compatible API) only here, nowhere else in the mesh

Principle 2: All other nodes produce structured data.
Principle 7: Only one node produces human-readable responses.
Principle 6: This node reads from CoordinatorState only — no direct module access.
"""
import logging
import os
import json
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Lazy import — only loaded when output_layer is actually used
_openai_client = None


def _get_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI()
        except Exception as e:
            logger.error(f"OutputLayer: failed to init OpenAI client: {e}")
    return _openai_client


class OutputLayer:
    """
    The final output node of the mesh.

    Accepts a CoordinatorState dict and a user query string.
    Returns a natural language response string.

    This is the ONLY class in the codebase that calls the LLM.
    Principle 7: Final Output Layer.
    """

    _SYSTEM_PROMPT = """You are the output layer of a cognitive mesh system.
You receive structured state data from the coordinator and render it into clear, concise responses.

Rules you must follow:
1. Only use information present in the state data provided. Do not hallucinate metrics.
2. Be direct and analytical. No corporate tone, no disclaimers.
3. When PHI is below 0.4, note that coherence is low and data may be unreliable.
4. When drift_vector exceeds 0.15, note that the system is drifting from baseline.
5. Format numbers to 3 decimal places. Use plain language for directions (up/down/critical).
6. Never pretend to have capabilities you don't have. If a metric is missing, say so.
"""

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model

    def render(
        self,
        user_query: str,
        coordinator_state: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Render a natural language response to a user query.

        Parameters
        ----------
        user_query : str
            The user's question or command.
        coordinator_state : dict
            The full CoordinatorState dict from MeshCoordinator.get_state_dict().
        history : list, optional
            Prior conversation turns as [{"role": "user"|"assistant", "content": "..."}].

        Returns
        -------
        str
            Natural language response. Empty string on failure.
        """
        client = _get_client()
        if client is None:
            return "Output layer unavailable: LLM client not initialized."

        # Build the state summary injected as context
        state_summary = self._summarize_state(coordinator_state)

        messages = [{"role": "system", "content": self._SYSTEM_PROMPT}]

        # Inject state as a system context block
        messages.append({
            "role": "system",
            "content": f"Current mesh state:\n{state_summary}"
        })

        # Inject conversation history
        if history:
            messages.extend(history[-10:])  # last 10 turns only

        messages.append({"role": "user", "content": user_query})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OutputLayer.render failed: {e}")
            return f"Output layer error: {e}"

    def render_status(self, coordinator_state: Dict[str, Any]) -> str:
        """
        Render a brief status summary without a user query.
        Used for periodic status broadcasts.
        """
        s = coordinator_state
        phi = s.get("phi", 0.5)
        sigma = s.get("sigma", 0.5)
        drift = s.get("drift_vector", 0.0)
        iteration = s.get("iteration", 0)
        n_concepts = len(s.get("abstractions", []))
        n_rules = len(s.get("rules", []))
        n_predictions = len(s.get("predictions", []))
        n_goals = len(s.get("active_goals", []))
        conflicts = s.get("resolved_conflicts", [])

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

    @staticmethod
    def _summarize_state(state: Dict[str, Any]) -> str:
        """
        Produce a compact, structured text summary of the CoordinatorState
        for injection into the LLM context window.
        """
        lines = []

        lines.append(f"Iteration: {state.get('iteration', 0)}")
        lines.append(f"PHI (coherence): {state.get('phi', 0.5):.4f}")
        lines.append(f"SIGMA (noise): {state.get('sigma', 0.5):.4f}")
        lines.append(f"Drift vector: {state.get('drift_vector', 0.0):+.4f}")

        # Weighted signals
        ws = state.get("weighted_signals", {})
        if ws:
            lines.append("Weighted signals: " + ", ".join(f"{k}={v:.3f}" for k, v in ws.items()))

        # Abstractions
        abstractions = state.get("abstractions", [])
        lines.append(f"Active concepts: {len(abstractions)}")
        for a in abstractions[:5]:
            lines.append(f"  - [{a.get('domain','')}] {a.get('label','')} conf={a.get('confidence',0):.3f}")

        # Rules
        rules = state.get("rules", [])
        lines.append(f"Active rules: {len(rules)}")
        for r in rules[:5]:
            lines.append(
                f"  - {' & '.join(r.get('antecedents',[]))} => {r.get('consequent','')} "
                f"conf={r.get('confidence',0):.3f}"
            )

        # Predictions
        predictions = state.get("predictions", [])
        lines.append(f"Active predictions: {len(predictions)}")
        for p in predictions[:5]:
            lines.append(
                f"  - {p.get('symbol','')} {p.get('direction','')} "
                f"conf={p.get('confidence',0):.3f} ticks={p.get('ticks_remaining',0)}"
            )

        # EEG
        eeg = state.get("eeg")
        if eeg:
            lines.append(
                f"EEG: phi={eeg.get('phi',0):.3f} sigma={eeg.get('sigma',0):.3f} "
                f"dominant={eeg.get('dominant_frequency','?')} "
                f"attention={eeg.get('attention_score',0):.3f}"
            )
            plp = eeg.get("phase_lock_pairs", [])
            if plp:
                lines.append(f"  Phase-locked pairs: {len(plp)}")

        # Goals
        goals = state.get("active_goals", [])
        lines.append(f"Active goals: {len(goals)}")
        for g in goals[:3]:
            lines.append(
                f"  - [{g.get('type','')}] {g.get('description','')} "
                f"priority={g.get('priority',0):.2f} progress={g.get('progress',0):.0%}"
            )

        # Cross-domain
        transfers = state.get("cross_domain_transfers", [])
        if transfers:
            lines.append(f"Cross-domain transfers: {len(transfers)}")
            for t in transfers[:3]:
                lines.append(
                    f"  - {t.get('source_domain','')} -> {t.get('target_domain','')} "
                    f"conf={t.get('confidence',0):.3f} gain={t.get('performance_gain',0):.3f}"
                )

        # Conflicts resolved
        conflicts = state.get("resolved_conflicts", [])
        if conflicts:
            lines.append(f"Conflicts resolved this cycle: {len(conflicts)}")
            for c in conflicts:
                lines.append(f"  - {c}")

        return "\n".join(lines)
