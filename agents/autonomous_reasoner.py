"""
Autonomous Reasoning Agent
===========================
Performs pattern analysis, hypothesis generation, and goal formulation
using the mesh's own native cognitive engines as the sole reasoning
layer.  All outputs are generated algorithmically — no external API calls.

Architecture (original vision):
  1. Native algorithmic reasoning  ← always runs, no external dependency
  2. Human-readable output          ← generated from native analysis only

The silicon vessel reasons for itself. No external API calls.
"""

import logging
import asyncio
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger("AutonomousReasoner")


# ──────────────────────────────────────────────────────────────────────────────
# Native Reasoning Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_concepts_by_domain(concepts: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Group concepts by their domain prefix."""
    clusters: Dict[str, List[Dict]] = defaultdict(list)
    for cid, concept in concepts.items():
        domain = concept.get("domain", "unknown")
        clusters[domain].append(concept)
    return dict(clusters)


def _top_concepts_by_confidence(concepts: Dict[str, Any], n: int = 10) -> List[Dict]:
    """Return the n highest-confidence concepts."""
    return sorted(
        concepts.values(),
        key=lambda c: c.get("confidence", 0),
        reverse=True
    )[:n]


def _detect_cross_domain_patterns(concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect concepts that share similar confidence trajectories across domains.
    Returns candidate cross-domain correlation pairs.
    """
    clusters = _cluster_concepts_by_domain(concepts)
    domain_list = list(clusters.keys())
    patterns = []

    for i, d_a in enumerate(domain_list):
        for d_b in domain_list[i + 1:]:
            confs_a = [c.get("confidence", 0) for c in clusters[d_a]]
            confs_b = [c.get("confidence", 0) for c in clusters[d_b]]
            if not confs_a or not confs_b:
                continue
            mean_a = sum(confs_a) / len(confs_a)
            mean_b = sum(confs_b) / len(confs_b)
            # Simple correlation proxy: both trending high or both trending low
            if abs(mean_a - mean_b) < 0.15:
                patterns.append({
                    "domain_a": d_a,
                    "domain_b": d_b,
                    "mean_confidence_a": round(mean_a, 4),
                    "mean_confidence_b": round(mean_b, 4),
                    "coherence_delta": round(abs(mean_a - mean_b), 4),
                    "type": "confidence_correlation",
                })
    return patterns


def _native_hypotheses(observations: List[Dict[str, Any]]) -> List[str]:
    """
    Generate testable hypotheses from raw observations using purely
    algorithmic analysis (no LLM required).
    """
    if not observations:
        return []

    hypotheses = []

    # Count domain frequency
    domain_counts = Counter(obs.get("domain", "unknown") for obs in observations)
    top_domain = domain_counts.most_common(1)[0][0] if domain_counts else "unknown"
    hypotheses.append(
        f"H1: Domain '{top_domain}' is the current primary attractor "
        f"({domain_counts[top_domain]}/{len(observations)} observations)."
    )

    # Detect value trends
    values = [obs.get("value", obs.get("price")) for obs in observations if obs.get("value") or obs.get("price")]
    numeric_values = [float(v) for v in values if v is not None]
    if len(numeric_values) >= 4:
        first_half = numeric_values[:len(numeric_values) // 2]
        second_half = numeric_values[len(numeric_values) // 2:]
        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)
        direction = "increasing" if mean_second > mean_first else "decreasing"
        pct_change = abs((mean_second - mean_first) / max(abs(mean_first), 1e-8)) * 100
        hypotheses.append(
            f"H2: Observable values are {direction} across the window "
            f"(Δ≈{pct_change:.2f}%). Investigate causal drivers."
        )

    # Detect multi-domain activity
    unique_domains = len(domain_counts)
    if unique_domains >= 3:
        hypotheses.append(
            f"H3: Multi-domain activity detected across {unique_domains} domains. "
            f"Phase-locking or interference patterns may be forming."
        )

    # Attention spike hypothesis
    entity_counts = Counter(
        obs.get("entity_id", obs.get("symbol", "?")) for obs in observations
    )
    top_entity, top_count = entity_counts.most_common(1)[0] if entity_counts else ("?", 0)
    if top_count > len(observations) * 0.3:
        hypotheses.append(
            f"H4: Entity '{top_entity}' accounts for {top_count}/{len(observations)} "
            f"observations — possible attention concentration or data imbalance."
        )

    return hypotheses


def _native_goal_formulation(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Formulate goals from current mesh metrics using rule-based logic.
    No LLM required.
    """
    goals = []
    phi = metrics.get("global_coherence_phi", 0.5)
    sigma = metrics.get("noise_level_sigma", 0.5)
    accuracy = metrics.get("prediction_accuracy", 0.5)
    total_concepts = metrics.get("total_concepts", 0)
    total_domains = metrics.get("total_domains", 0)

    if phi < 0.4:
        goals.append({
            "title": "Increase Global Coherence",
            "rationale": f"PHI={phi:.3f} is below the coherence threshold (0.4). "
                         "The mesh is in a chaotic regime.",
            "metric": "PHI > 0.6 within 100 cognitive cycles",
            "priority": 0.9,
            "generated_by": "native_reasoner",
        })

    if sigma > 0.6:
        goals.append({
            "title": "Reduce Noise Level",
            "rationale": f"SIGMA={sigma:.3f} indicates high noise. "
                         "Concept pruning and rule confidence feedback should be amplified.",
            "metric": "SIGMA < 0.4 within 200 cognitive cycles",
            "priority": 0.85,
            "generated_by": "native_reasoner",
        })

    if accuracy < 0.45 and metrics.get("predictions_validated", 0) > 10:
        goals.append({
            "title": "Improve Prediction Accuracy",
            "rationale": f"Prediction accuracy={accuracy:.3f} is below chance. "
                         "Rule confidence feedback loop may need recalibration.",
            "metric": "Prediction accuracy > 0.55 over next 50 validations",
            "priority": 0.8,
            "generated_by": "native_reasoner",
        })

    if total_domains < 2 and total_concepts > 5:
        goals.append({
            "title": "Expand Domain Coverage",
            "rationale": "Mesh is operating in a single domain. "
                         "Cross-domain transfer and analogy formation require ≥2 domains.",
            "metric": "Register observations from at least 2 distinct domains",
            "priority": 0.7,
            "generated_by": "native_reasoner",
        })

    if not goals:
        goals.append({
            "title": "Maintain Homeostasis",
            "rationale": f"Mesh is in a stable regime (PHI={phi:.3f}, SIGMA={sigma:.3f}). "
                         "Continue observation and deepen concept hierarchy.",
            "metric": "Sustain PHI > 0.5 and SIGMA < 0.5 for 500 cycles",
            "priority": 0.5,
            "generated_by": "native_reasoner",
        })

    return goals


# ──────────────────────────────────────────────────────────────────────────────
# AutonomousReasoner
# ──────────────────────────────────────────────────────────────────────────────

class AutonomousReasoner:
    """
    Autonomous reasoning agent.

    All reasoning, pattern analysis, hypothesis generation, and goal
    formulation is performed natively by the mesh's own cognitive engines.
    No external API calls.  No LLM dependency.
    """

    def __init__(self, core=None):
        self.core = core
        logger.info("AutonomousReasoner initialised — native reasoning only")

    # ── Public API ──────────────────────────────────────────────────────────

    async def analyze_patterns(
        self,
        concepts: Dict[str, Any],
        rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyse patterns across all concepts and rules.
        Returns structured insights derived natively.
        """
        if not concepts:
            return {"status": "no_data", "insights": []}

        # ── Native analysis ──
        top = _top_concepts_by_confidence(concepts, n=10)
        clusters = _cluster_concepts_by_domain(concepts)
        cross_patterns = _detect_cross_domain_patterns(concepts)

        native_insights = {
            "top_concepts": [
                {
                    "id": c.get("id", "?"),
                    "domain": c.get("domain", "?"),
                    "entity_id": c.get("entity_id", c.get("symbol", "?")),
                    "confidence": round(c.get("confidence", 0), 4),
                }
                for c in top
            ],
            "domain_clusters": {
                domain: len(concepts_list)
                for domain, concepts_list in clusters.items()
            },
            "cross_domain_patterns": cross_patterns,
            "total_concepts": len(concepts),
            "total_rules": len(rules),
        }

        return {
            "status": "success",
            "native_analysis": native_insights,
        }

    async def generate_hypotheses(
        self,
        recent_observations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate testable hypotheses from recent observations using native analysis.
        """
        return _native_hypotheses(recent_observations)

    async def formulate_goals(
        self,
        metrics: Dict[str, Any],
        concepts: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Formulate goals from current mesh state using native rule-based analysis.
        """
        return _native_goal_formulation(metrics)

    async def synthesize_insights(
        self,
        domain_a: str,
        domain_b: str
    ) -> Optional[str]:
        """
        Generate cross-domain insights by comparing two domains natively.
        """
        if not self.core:
            return None

        concepts = self.core.get_concepts_snapshot()
        concepts_a = {cid: c for cid, c in concepts.items() if c.get("domain", "").startswith(domain_a)}
        concepts_b = {cid: c for cid, c in concepts.items() if c.get("domain", "").startswith(domain_b)}

        if not concepts_a or not concepts_b:
            return f"Insufficient data for domains '{domain_a}' and '{domain_b}'."

        # Native cross-domain summary
        mean_conf_a = sum(c.get("confidence", 0) for c in concepts_a.values()) / len(concepts_a)
        mean_conf_b = sum(c.get("confidence", 0) for c in concepts_b.values()) / len(concepts_b)
        delta = abs(mean_conf_a - mean_conf_b)

        native_summary = (
            f"Domain '{domain_a}': {len(concepts_a)} concepts, "
            f"mean confidence={mean_conf_a:.3f}.\n"
            f"Domain '{domain_b}': {len(concepts_b)} concepts, "
            f"mean confidence={mean_conf_b:.3f}.\n"
            f"Confidence delta={delta:.3f} — "
            + ("high coherence alignment." if delta < 0.1 else
               "moderate divergence." if delta < 0.25 else
               "significant divergence — possible anti-correlation.")
        )

        return native_summary
