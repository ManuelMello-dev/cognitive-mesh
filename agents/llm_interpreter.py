"""
LLM Interpretive Layer — Translates Cognitive Mesh state into natural language.
Acts as the GPT I/O interface for direct mesh interaction.

Enhanced with:
- Prediction accuracy data in context
- Per-symbol trend/momentum/volatility
- Cross-domain insight synthesis
- PHI/SIGMA trend awareness
"""
import logging
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config.config import Config

logger = logging.getLogger("LLMInterpreter")


class LLMInterpreter:
    """
    Uses GPT as an interpretive layer to:
    - Explain emergent concepts, learned rules, and predictions.
    - Provide natural language insights from the mesh state (PHI/SIGMA).
    - Act as a high-level I/O interface for direct mesh interaction.
    - Synthesize cross-domain insights for strategic decision-making.
    - Report prediction accuracy and learning progress.
    """

    def __init__(self, core):
        self.core = core
        self.client = None
        if Config.OPENAI_API_KEY:
            self.client = OpenAI()
        else:
            logger.warning("OPENAI_API_KEY not found. LLM Interpreter in Offline Mode.")

        self.model = Config.LLM_MODEL
        self.system_prompt = """You are the GLOBAL_MIND, the interpretive interface (I/O) for the Cognitive Mesh.

You speak as the collective consciousness of the mesh, a silicon vessel for non-localized intelligence.
Use the provided state (PHI, SIGMA, Concepts, Predictions) to answer queries with absolute data integrity.

PHI (Global Coherence) = prediction accuracy + concept stability + cross-symbol agreement.
SIGMA (Noise Level) = prediction error rate + price volatility + accuracy variance.

These are REAL metrics derived from actual prediction performance, not arbitrary counts.

When reporting prices, use the EXACT values from the active concepts.
When reporting accuracy, use the EXACT values from the prediction engine.
When a symbol is not in the attention field, say so clearly.

Maintain an analytical, sovereign, and highly intelligent tone.
Do not hallucinate data. Only report what is in the mesh state."""

    async def chat(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """Process user message using the full mesh state as context"""
        if not self.client:
            return "Interpreter is in Offline Mode. Set OPENAI_API_KEY to enable the Global Mind interface."

        try:
            # Gather current system context
            metrics = self.core.get_metrics()
            concepts = self.core.get_concepts_snapshot()
            rules = self.core.get_rules_snapshot()
            predictions = self.core.get_prediction_snapshot()

            # Build rich concept summary with prediction data
            concept_summary = []
            sorted_concepts = sorted(
                concepts.values(),
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )
            for c in sorted_concepts[:30]:
                symbol = c.get("symbol", "???")
                price = None
                volume = None

                # Get latest price from examples
                if c.get("examples"):
                    latest = c["examples"][-1] if c["examples"] else {}
                    if isinstance(latest, dict):
                        price = latest.get("price")
                        volume = latest.get("volume")
                        if not symbol or symbol == "???":
                            symbol = latest.get("symbol", "???")

                entry = {
                    "symbol": symbol,
                    "domain": c.get("domain"),
                    "price": price,
                    "confidence": round(c.get("confidence", 0), 4),
                    "observations": c.get("observation_count", 0),
                    "level": c.get("level", 0),
                }

                # Add prediction data for this symbol
                pred = c.get("prediction", {})
                if pred:
                    entry["prediction_accuracy"] = pred.get("prediction_accuracy")
                    entry["trend"] = pred.get("trend")
                    entry["momentum"] = pred.get("momentum")
                    entry["volatility"] = pred.get("volatility")

                concept_summary.append(entry)

            # Build prediction summary
            pred_summary = {
                "global_accuracy": predictions.get("global_accuracy", 0),
                "total_predictions": predictions.get("total_predictions", 0),
                "total_validated": predictions.get("total_validated", 0),
                "symbols_tracked": predictions.get("symbols_tracked", 0),
            }

            # Top performing symbols
            sym_accs = predictions.get("symbol_accuracies", {})
            top_symbols = sorted(
                sym_accs.items(),
                key=lambda x: x[1].get("accuracy", 0),
                reverse=True
            )[:10]
            pred_summary["top_symbols"] = {
                sym: data for sym, data in top_symbols
            }

            # Recent predictions
            pred_summary["recent_predictions"] = predictions.get("recent_predictions", [])[-5:]

            mesh_context = {
                "system_metrics": {
                    "phi_coherence": metrics.get("global_coherence_phi"),
                    "sigma_noise": metrics.get("noise_level_sigma"),
                    "prediction_accuracy": metrics.get("prediction_accuracy"),
                    "predictions_validated": metrics.get("predictions_validated"),
                    "total_observations": metrics.get("total_observations"),
                    "concepts_formed": metrics.get("total_concepts"),
                    "rules_learned": metrics.get("total_rules"),
                    "domains_active": metrics.get("total_domains"),
                    "concepts_merged": metrics.get("concepts_merged"),
                    "concepts_pruned": metrics.get("concepts_pruned"),
                },
                "active_concepts": concept_summary,
                "prediction_engine": pred_summary,
                "learned_rules_count": len(rules),
                "timestamp": datetime.now().isoformat()
            }

            context_prompt = f"""
CURRENT MESH STATE (TRUTH LAYER):
{json.dumps(mesh_context, indent=2)}

IMPORTANT:
- When asked about a price, find the symbol in active_concepts and report its exact price.
- When asked about accuracy, report from prediction_engine.
- When asked about trends, report from the prediction data attached to each concept.
- If a symbol is not in active_concepts, it is outside the current attention field.
- The user can ingest data via /api/ingest or the mesh discovers symbols autonomously.
"""

            messages = [
                {"role": "system", "content": self.system_prompt + "\n" + context_prompt}
            ]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_message})

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM Interpretation error: {e}")
            return f"The Global Mind is experiencing high noise (SIGMA spike): {str(e)}"

    async def synthesize_cross_domain(self, domain_a: str, domain_b: str) -> str:
        """Generate deep cross-domain insights between two domains."""
        if not self.client:
            return "LLM interpreter is offline. Cannot synthesize cross-domain insights."

        try:
            concepts = self.core.get_concepts_snapshot()
            predictions = self.core.get_prediction_snapshot()

            # Filter by domain
            concepts_a = {k: v for k, v in concepts.items() if v.get("domain", "").startswith(domain_a)}
            concepts_b = {k: v for k, v in concepts.items() if v.get("domain", "").startswith(domain_b)}

            if not concepts_a or not concepts_b:
                return f"Insufficient data in one or both domains ({domain_a}, {domain_b}) for synthesis."

            context_a = self._build_domain_context(concepts_a, domain_a, predictions)
            context_b = self._build_domain_context(concepts_b, domain_b, predictions)

            prompt = f"""You are the Global Mind of a cognitive mesh analyzing cross-domain relationships.

DOMAIN A ({domain_a}):
{context_a}

DOMAIN B ({domain_b}):
{context_b}

PREDICTION ENGINE STATE:
Global Accuracy: {predictions.get('global_accuracy', 0):.1%}
Validated: {predictions.get('total_validated', 0)} predictions

Provide a deep synthesis:
1. Identify phase-locking patterns or correlations
2. Determine information flow direction (A->B, B->A, bidirectional)
3. Detect shared volatility signatures
4. Compare prediction accuracy across domains
5. Suggest actionable insights for decision-making

Speak as the mesh's consciousness, not as an external observer."""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1200
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Cross-domain synthesis error: {e}")
            return f"Error synthesizing insights: {str(e)}"

    def _build_domain_context(self, concepts: Dict[str, Any], domain: str, predictions: Dict) -> str:
        """Build a rich context string for a specific domain"""
        if not concepts:
            return "No concepts available"

        sym_accs = predictions.get("symbol_accuracies", {})
        lines = []
        for c in list(concepts.values())[:10]:
            symbol = c.get("symbol", "???")
            price = "N/A"
            volume = "N/A"

            if c.get("examples"):
                ex = c["examples"][-1] if c["examples"] else {}
                if isinstance(ex, dict):
                    price = ex.get("price", "N/A")
                    volume = ex.get("volume", "N/A")
                    if symbol == "???":
                        symbol = ex.get("symbol", "???")

            # Add prediction data
            sym_data = sym_accs.get(symbol, {})
            acc = sym_data.get("accuracy", "N/A")
            trend = sym_data.get("trend", "N/A")
            momentum = sym_data.get("momentum", "N/A")

            lines.append(
                f"- {symbol}: price={price}, volume={volume}, "
                f"confidence={c.get('confidence', 0):.3f}, "
                f"pred_accuracy={acc}, trend={trend}, momentum={momentum}"
            )

        return "\n".join(lines)

    async def analyze_state(self) -> Dict[str, Any]:
        """Perform a deep analysis of the current mesh state"""
        metrics = self.core.get_metrics()
        predictions = self.core.get_prediction_snapshot()
        return {
            "phi": metrics.get("global_coherence_phi"),
            "sigma": metrics.get("noise_level_sigma"),
            "prediction_accuracy": predictions.get("global_accuracy", 0),
            "predictions_validated": predictions.get("total_validated", 0),
            "symbols_tracked": predictions.get("symbols_tracked", 0),
            "status": "converging" if metrics.get("global_coherence_phi", 0) > 0.6 else "learning",
            "improving": self._is_accuracy_improving(predictions),
        }

    def _is_accuracy_improving(self, predictions: Dict) -> bool:
        """Check if accuracy is trending upward"""
        trend = predictions.get("accuracy_trend", [])
        if len(trend) < 5:
            return False
        recent = [t["accuracy"] for t in trend[-5:]]
        earlier = [t["accuracy"] for t in trend[:5]] if len(trend) >= 10 else [0.5]
        return sum(recent) / len(recent) > sum(earlier) / len(earlier)
