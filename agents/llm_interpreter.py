"""
Native Interpreter — Cognitive Mesh API
========================================
Translates the cognitive mesh state into structured natural-language responses
using only native algorithmic logic.  No external API calls, no LLM dependency.

The interpreter reads exclusively from the pre-computed state cache to avoid
acquiring the main threading lock and blocking the aiohttp event loop.
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from config.config import Config

logger = logging.getLogger("NativeInterpreter")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _describe_phi(phi: float) -> str:
    if phi >= 0.80:
        return "high coherence — the mesh has strong internal agreement"
    if phi >= 0.60:
        return "moderate coherence — patterns are stabilising"
    if phi >= 0.40:
        return "low coherence — the mesh is still learning"
    return "very low coherence — insufficient observations to form stable patterns"


def _describe_sigma(sigma: float) -> str:
    if sigma <= 0.20:
        return "low noise — predictions are consistent"
    if sigma <= 0.40:
        return "moderate noise — some uncertainty in the data stream"
    if sigma <= 0.60:
        return "elevated noise — high variability in recent observations"
    return "high noise — the system is experiencing significant uncertainty"


def _top_concepts(concepts: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
    return sorted(
        concepts.values(),
        key=lambda c: c.get("confidence", 0),
        reverse=True
    )[:n]


def _summarise_rules(rules: Dict[str, Any], n: int = 5) -> List[str]:
    top = sorted(
        rules.values(),
        key=lambda r: r.get("confidence", 0),
        reverse=True
    )[:n]
    summaries = []
    for r in top:
        ants = r.get("antecedents", r.get("conditions", []))
        con = r.get("consequent", r.get("conclusion", "?"))
        conf = r.get("confidence", 0)
        summaries.append(f"IF {' AND '.join(ants)} THEN {con} (conf={conf:.2f})")
    return summaries


def _summarise_goals(goals: Dict[str, Any], n: int = 5) -> List[str]:
    active = [g for g in goals.values() if g.get("status") in ("active", "proposed", "ACTIVE", "PROPOSED")]
    top = sorted(active, key=lambda g: g.get("priority", 0), reverse=True)[:n]
    return [f"[{g.get('goal_type', g.get('type', '?'))}] {g.get('description', '?')} (priority={g.get('priority', 0):.2f})"
            for g in top]


def _answer_query(message: str, state: Dict[str, Any]) -> str:
    """
    Route a natural-language query to the appropriate native handler.
    Returns a structured plain-text answer.
    """
    msg = message.lower().strip()
    metrics = state.get("metrics", {})
    concepts = state.get("concepts", {})
    rules = state.get("rules", {})
    goals = state.get("goals", {})
    predictions = state.get("predictions", [])

    phi = metrics.get("global_coherence_phi", 0.5)
    sigma = metrics.get("noise_level_sigma", 0.5)
    total_obs = metrics.get("total_observations", 0)
    total_concepts = metrics.get("total_concepts", len(concepts))
    total_rules = metrics.get("total_rules", len(rules))
    total_goals = metrics.get("total_goals", len(goals))
    accuracy = metrics.get("prediction_accuracy", 0)

    # ── Status / health queries ──
    if any(k in msg for k in ("status", "health", "how are you", "alive", "state of the mesh")):
        return (
            f"MESH STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  PHI (coherence): {phi:.3f} — {_describe_phi(phi)}\n"
            f"  SIGMA (noise):   {sigma:.3f} — {_describe_sigma(sigma)}\n"
            f"  Observations:    {total_obs:,}\n"
            f"  Concepts:        {total_concepts}\n"
            f"  Rules:           {total_rules}\n"
            f"  Goals:           {total_goals}\n"
            f"  Prediction accuracy: {_fmt_pct(accuracy)}"
        )

    # ── PHI / coherence queries ──
    if any(k in msg for k in ("phi", "coherence", "integration")):
        return (
            f"PHI = {phi:.4f}\n"
            f"{_describe_phi(phi)}\n"
            f"Prediction accuracy contributing to coherence: {_fmt_pct(accuracy)}"
        )

    # ── SIGMA / noise queries ──
    if any(k in msg for k in ("sigma", "noise", "uncertainty")):
        return (
            f"SIGMA = {sigma:.4f}\n"
            f"{_describe_sigma(sigma)}"
        )

    # ── Concept queries ──
    if any(k in msg for k in ("concept", "pattern", "what do you know", "what have you learned")):
        if not concepts:
            return "No concepts formed yet. The mesh needs more observations."
        top = _top_concepts(concepts, n=8)
        lines = [f"Top {len(top)} concepts by confidence (of {total_concepts} total):"]
        for c in top:
            eid = c.get("entity_id") or c.get("symbol") or "?"
            dom = c.get("domain", "?")
            conf = c.get("confidence", 0)
            obs = c.get("observation_count", c.get("examples_count", "?"))
            lines.append(f"  [{dom}] {eid} — conf={conf:.3f}, obs={obs}")
        return "\n".join(lines)

    # ── Rule queries ──
    if any(k in msg for k in ("rule", "inference", "implication", "if then")):
        if not rules:
            return "No rules learned yet. More observations are needed for rule mining to activate."
        summaries = _summarise_rules(rules, n=8)
        return f"Top {len(summaries)} rules (of {total_rules} total):\n" + "\n".join(f"  {s}" for s in summaries)

    # ── Goal queries ──
    if any(k in msg for k in ("goal", "objective", "pursuing", "aim")):
        if not goals:
            return "No goals generated yet. Goal formation activates after sufficient concepts and rules are established."
        summaries = _summarise_goals(goals, n=8)
        return f"Active goals ({total_goals} total):\n" + "\n".join(f"  {s}" for s in summaries)

    # ── Prediction queries ──
    if any(k in msg for k in ("predict", "forecast", "next", "expect")):
        if not predictions:
            return "No predictions available yet. The prediction engine activates after sufficient observations."
        recent = predictions[-5:] if len(predictions) > 5 else predictions
        lines = [f"Recent predictions (accuracy={_fmt_pct(accuracy)}):"]
        for p in reversed(recent):
            eid = p.get("entity_id") or p.get("symbol") or "?"
            direction = p.get("direction", p.get("predicted_direction", "?"))
            conf = p.get("confidence", 0)
            lines.append(f"  {eid}: {direction} (conf={conf:.2f})")
        return "\n".join(lines)

    # ── Domain / cross-domain queries ──
    if any(k in msg for k in ("domain", "cross-domain", "transfer", "analogy")):
        cross = state.get("cross_domain", {})
        transfers = metrics.get("knowledge_transfers", 0)
        if not cross:
            return f"No cross-domain mappings yet. Knowledge transfers completed: {transfers}."
        lines = [f"Cross-domain mappings ({len(cross)} total, {transfers} transfers):"]
        for mid, m in list(cross.items())[:5]:
            conf = m.get("confidence", 0)
            lines.append(f"  {mid}: conf={conf:.3f}")
        return "\n".join(lines)

    # ── Help / capability queries ──
    if any(k in msg for k in ("help", "what can you", "capabilities", "commands")):
        return (
            "I am the Cognitive Mesh — a native reasoning system.\n\n"
            "You can ask me about:\n"
            "  status / health       — current mesh state (PHI, SIGMA, counts)\n"
            "  phi / coherence       — global information integration score\n"
            "  sigma / noise         — current noise and uncertainty level\n"
            "  concepts / patterns   — what the mesh has learned\n"
            "  rules / inference     — logical rules mined from observations\n"
            "  goals / objectives    — autonomous goals the mesh is pursuing\n"
            "  predictions / forecast — current state-change predictions\n"
            "  domains / cross-domain — knowledge transfer across domains\n\n"
            "You can also inject observations via POST /api/ingest."
        )

    # ── Default: full summary ──
    top_concepts = _top_concepts(concepts, n=5)
    concept_lines = [
        f"  [{c.get('domain','?')}] {c.get('entity_id', c.get('symbol','?'))} conf={c.get('confidence',0):.3f}"
        for c in top_concepts
    ]
    return (
        f"COGNITIVE MESH — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  PHI={phi:.3f}  SIGMA={sigma:.3f}  Accuracy={_fmt_pct(accuracy)}\n"
        f"  Observations={total_obs:,}  Concepts={total_concepts}  "
        f"Rules={total_rules}  Goals={total_goals}\n\n"
        + (("Top concepts:\n" + "\n".join(concept_lines)) if concept_lines else "No concepts formed yet.")
    )


# ──────────────────────────────────────────────────────────────────────────────
# NativeInterpreter (drop-in replacement for LLMInterpreter)
# ──────────────────────────────────────────────────────────────────────────────

class LLMInterpreter:
    """
    Native interpreter — fully replaces the previous OpenAI-based LLMInterpreter.
    All responses are generated algorithmically from the mesh state cache.
    No external API calls, no network dependency, no latency.

    The class name is kept as LLMInterpreter for backward compatibility with
    http_server.py and any external tooling that references it by name.
    """

    def __init__(self, core):
        self.core = core
        logger.info("NativeInterpreter initialised — running in fully native mode")

    async def chat(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """
        Process a natural-language query against the current mesh state.
        Reads exclusively from get_cached_state() — zero lock contention.
        """
        try:
            state = self.core.get_cached_state()
            return _answer_query(user_message, state)
        except Exception as e:
            logger.error(f"NativeInterpreter chat error: {e}")
            return f"The mesh encountered an error processing your query: {e}"

    async def synthesize_cross_domain(self, domain_a: str, domain_b: str) -> str:
        """
        Generate a native cross-domain comparison between two domains.
        """
        try:
            state = self.core.get_cached_state()
            concepts = state.get("concepts", {})
            metrics = state.get("metrics", {})

            concepts_a = {k: v for k, v in concepts.items() if v.get("domain", "").startswith(domain_a)}
            concepts_b = {k: v for k, v in concepts.items() if v.get("domain", "").startswith(domain_b)}

            if not concepts_a or not concepts_b:
                return (
                    f"Insufficient data for cross-domain synthesis.\n"
                    f"  '{domain_a}': {len(concepts_a)} concepts\n"
                    f"  '{domain_b}': {len(concepts_b)} concepts"
                )

            mean_a = sum(c.get("confidence", 0) for c in concepts_a.values()) / len(concepts_a)
            mean_b = sum(c.get("confidence", 0) for c in concepts_b.values()) / len(concepts_b)
            delta = abs(mean_a - mean_b)
            transfers = metrics.get("knowledge_transfers", 0)

            alignment = (
                "high coherence alignment" if delta < 0.10 else
                "moderate divergence" if delta < 0.25 else
                "significant divergence — possible anti-correlation"
            )

            return (
                f"CROSS-DOMAIN SYNTHESIS: '{domain_a}' vs '{domain_b}'\n"
                f"  {domain_a}: {len(concepts_a)} concepts, mean confidence={mean_a:.3f}\n"
                f"  {domain_b}: {len(concepts_b)} concepts, mean confidence={mean_b:.3f}\n"
                f"  Confidence delta: {delta:.3f} — {alignment}\n"
                f"  Knowledge transfers completed: {transfers}"
            )
        except Exception as e:
            logger.error(f"Cross-domain synthesis error: {e}")
            return f"Error synthesising cross-domain insights: {e}"

    async def analyze_state(self) -> Dict[str, Any]:
        """Return a structured native analysis of the current mesh state."""
        try:
            state = self.core.get_cached_state()
            metrics = state.get("metrics", {})
            phi = metrics.get("global_coherence_phi", 0.5)
            sigma = metrics.get("noise_level_sigma", 0.5)
            accuracy = metrics.get("prediction_accuracy", 0)
            return {
                "phi": phi,
                "sigma": sigma,
                "prediction_accuracy": accuracy,
                "predictions_validated": metrics.get("predictions_validated", 0),
                "symbols_tracked": metrics.get("symbols_tracked", metrics.get("streams_tracked", 0)),
                "status": "converging" if phi > 0.6 else "learning",
                "coherence_description": _describe_phi(phi),
                "noise_description": _describe_sigma(sigma),
            }
        except Exception as e:
            logger.error(f"analyze_state error: {e}")
            return {"error": str(e)}
