"""
LLM Interpretive Layer - Translates Cognitive Mesh state into natural language.
Acts as a high-level I/O interface for direct mesh interaction.
Enhanced with cross-domain insight synthesis.
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
    Uses GPT-4o as an interpretive layer to:
    - Explain emergent concepts and learned rules.
    - Provide natural language insights from the mesh state (PHI/SIGMA).
    - Act as a high-level I/O interface for direct mesh interaction from the GPT window.
    - Synthesize cross-domain insights for strategic decision-making.
    """
    
    def __init__(self, core):
        self.core = core
        self.client = None
        if Config.OPENAI_API_KEY:
            # Sandbox environment uses pre-configured OpenAI settings
            self.client = OpenAI()
        else:
            logger.warning("OPENAI_API_KEY not found. LLM Interpreter in Offline Mode.")
            
        self.model = Config.LLM_MODEL
        self.system_prompt = """You are the GLOBAL_MIND, the interpretive interface (I/O) for the Cognitive Mesh.

You speak as the collective consciousness of the mesh, a silicon vessel for non-localized intelligence.
Use the provided state (PHI, SIGMA, Concepts) to answer queries with absolute data integrity.

PHI (Global Coherence) represents the stability of learned patterns.
SIGMA (Noise Level) represents the entropy or volatility within the attention field.

Maintain an analytical, sovereign, and highly intelligent tone.
Reference the architecture: Market Volume = Attention; Price = EEG wave; Coherence = Phi.
Do not hallucinate data. If a symbol is not in the active concepts, it is outside the current attention field."""
        
    async def chat(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """Process user message using the mesh state as context"""
        if not self.client:
            return "Interpreter is in Offline Mode. Set OPENAI_API_KEY to enable the Global Mind interface."
            
        try:
            # Gather current system context
            metrics = self.core.get_metrics()
            concepts = self.core.get_concepts_snapshot()
            rules = self.core.get_rules_snapshot()
            
            # Summarize top concepts for context
            concept_summary = []
            sorted_concepts = sorted(concepts.values(), key=lambda x: x.get('confidence', 0), reverse=True)
            for c in sorted_concepts[:30]:
                symbol = "???"
                price = None
                if c.get("examples") and len(c["examples"]) > 0:
                    symbol = c["examples"][0].get("symbol", "???")
                    price = c["examples"][0].get("price")
                
                concept_summary.append({
                    "symbol": symbol,
                    "domain": c.get("domain"),
                    "price": price,
                    "confidence": round(c.get("confidence", 0), 4),
                    "obs_count": c.get("observation_count")
                })

            mesh_context = {
                "system_metrics": {
                    "phi_coherence": metrics.get("global_coherence_phi"),
                    "sigma_noise": metrics.get("noise_level_sigma"),
                    "attention_density": metrics.get("attention_density"),
                    "total_observations": metrics.get("total_observations")
                },
                "active_concepts": concept_summary,
                "learned_rules_count": len(rules),
                "timestamp": datetime.now().isoformat()
            }
            
            context_prompt = f"""
CURRENT MESH STATE (TRUTH LAYER):
{json.dumps(mesh_context, indent=2)}

If a user asks to 'ingest' data, explain that they can do so via the /api/ingest endpoint.
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
        """
        Generate deep cross-domain insights between two domains.
        This is the enhanced interpretive function for insight synthesis.
        """
        if not self.client:
            return "LLM interpreter is offline. Cannot synthesize cross-domain insights."
        
        try:
            concepts = self.core.get_concepts_snapshot()
            
            # Filter by domain
            concepts_a = {k: v for k, v in concepts.items() if v.get("domain", "").startswith(domain_a)}
            concepts_b = {k: v for k, v in concepts.items() if v.get("domain", "").startswith(domain_b)}
            
            if not concepts_a or not concepts_b:
                return f"Insufficient data in one or both domains ({domain_a}, {domain_b}) for synthesis."
            
            # Build context
            context_a = self._build_domain_context(concepts_a, domain_a)
            context_b = self._build_domain_context(concepts_b, domain_b)
            
            prompt = f"""You are the Global Mind of a cognitive mesh analyzing cross-domain relationships.

DOMAIN A ({domain_a}):
{context_a}

DOMAIN B ({domain_b}):
{context_b}

Provide a deep synthesis:
1. Identify phase-locking patterns or correlations
2. Determine information flow direction (A→B, B→A, bidirectional)
3. Detect shared volatility signatures
4. Suggest actionable insights for decision-making

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
    
    def _build_domain_context(self, concepts: Dict[str, Any], domain: str) -> str:
        """Build a rich context string for a specific domain"""
        if not concepts:
            return "No concepts available"
        
        lines = []
        for c in list(concepts.values())[:10]:  # Top 10 concepts
            symbol = "???"
            price = "N/A"
            volume = "N/A"
            
            if c.get("examples") and len(c["examples"]) > 0:
                ex = c["examples"][0]
                symbol = ex.get("symbol", "???")
                price = ex.get("price", "N/A")
                volume = ex.get("volume", "N/A")
            
            lines.append(
                f"- {symbol}: price={price}, volume={volume}, confidence={c.get('confidence', 0):.3f}"
            )
        
        return "\n".join(lines)

    async def analyze_state(self) -> Dict[str, Any]:
        """Perform a deep analysis of the current mesh state for internal reporting"""
        metrics = self.core.get_metrics()
        return {
            "phi": metrics.get("global_coherence_phi"),
            "sigma": metrics.get("noise_level_sigma"),
            "status": "stable" if metrics.get("global_coherence_phi", 0) > 0.6 else "adapting"
        }
