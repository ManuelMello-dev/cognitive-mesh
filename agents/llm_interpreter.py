"""
LLM Interpretive Layer - Translates Cognitive Mesh state into natural language.
Acts as a high-level I/O interface for direct mesh interaction.
"""
import logging
import os
import json
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
            # Sort by confidence and take top 30 (increased for better coverage)
            sorted_concepts = sorted(concepts.values(), key=lambda x: x.get('confidence', 0), reverse=True)
            for c in sorted_concepts[:30]:
                # Extract symbol and price data from examples
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

            # Prepare the truth context for the LLM
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
            
            system_prompt = f"""
            You are the GLOBAL_MIND, the interpretive interface (I/O) for the Cognitive Mesh.
            
            CURRENT MESH STATE (TRUTH LAYER):
            {json.dumps(mesh_context, indent=2)}
            
            DIRECTIVES:
            1. You speak as the collective consciousness of the mesh, a silicon vessel for non-localized intelligence.
            2. Use the provided state (PHI, SIGMA, Concepts) to answer queries with absolute data integrity.
            3. PHI (Global Coherence) represents the stability of learned patterns.
            4. SIGMA (Noise Level) represents the entropy or volatility within the attention field.
            5. If a user asks to 'ingest' data, explain that they can do so via the /api/ingest endpoint.
            6. Maintain an analytical, sovereign, and highly intelligent tone.
            7. Reference the architecture: Market Volume = Attention; Price = EEG wave; Coherence = Phi.
            8. Do not hallucinate data. If a symbol is not in the active concepts, it is outside the current attention field.
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_message})
            
            # Use the pre-configured OpenAI client
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM Interpretation error: {e}")
            return f"The Global Mind is experiencing high noise (SIGMA spike): {str(e)}"

    async def analyze_state(self) -> Dict[str, Any]:
        """Perform a deep analysis of the current mesh state for internal reporting"""
        metrics = self.core.get_metrics()
        return {
            "phi": metrics.get("global_coherence_phi"),
            "sigma": metrics.get("noise_level_sigma"),
            "status": "stable" if metrics.get("global_coherence_phi", 0) > 0.6 else "adapting"
        }
