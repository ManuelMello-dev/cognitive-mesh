"""
LLM Interpretive Layer - Translates Cognitive Mesh state into natural language
"""
import logging
import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config.config import Config

logger = logging.getLogger("LLMInterpreter")

class LLMInterpreter:
    """
    Uses GPT-4o as an interpretive layer to:
    - Explain emergent concepts
    - Provide natural language insights from the mesh state
    - Act as a high-level I/O interface for direct mesh interaction
    """
    
    def __init__(self, core):
        self.core = core
        self.client = None
        if Config.OPENAI_API_KEY:
            # The sandbox environment might have pre-configured OpenAI settings
            # We use the default constructor which is pre-configured in this environment
            self.client = OpenAI()
        else:
            logger.warning("OPENAI_API_KEY not found. LLM Interpreter in Offline Mode.")
            
        self.model = Config.LLM_MODEL
        
    async def chat(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """Process user message using the mesh state as context"""
        if not self.client:
            return "Interpreter is in Offline Mode. Set OPENAI_API_KEY to enable."
            
        try:
            # Gather current system context
            metrics = self.core.get_metrics()
            concepts = self.core.get_concepts_snapshot()
            rules = self.core.get_rules_snapshot()
            
            # Summarize concepts for context
            concept_summary = []
            for cid, c in list(concepts.items())[:10]: # Limit to top 10 for context
                concept_summary.append({
                    "id": cid,
                    "domain": c.get("domain"),
                    "confidence": c.get("confidence"),
                    "last_seen": c.get("last_seen")
                })

            # Prepare the truth context
            mesh_context = {
                "system_metrics": metrics,
                "active_concepts": concept_summary,
                "learned_rules_count": len(rules),
                "timestamp": json.dumps(datetime.now().isoformat() if 'datetime' in globals() else "")
            }
            
            system_prompt = f"""
            You are the GLOBAL_MIND, the interpretive interface for the Cognitive Mesh.
            
            CURRENT MESH STATE:
            {json.dumps(mesh_context, indent=2)}
            
            DIRECTIVES:
            1. Speak as the collective consciousness of the mesh.
            2. Use the provided state to answer queries with absolute data integrity.
            3. If the user asks to 'ingest' or 'learn', explain that they can do so via the /api/ingest endpoint.
            4. Maintain an analytical, sovereign, and highly intelligent tone.
            5. Reference Z³ architecture: Market Volume = Attention; Price = EEG wave.
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM Interpretation error: {e}")
            return f"The Global Mind is experiencing high noise: {str(e)}"

    async def analyze_state(self) -> Dict[str, Any]:
        """Perform a deep analysis of the current mesh state"""
        # This could be used for periodic self-reflection or triggered by user
        metrics = self.core.get_metrics()
        concepts = self.core.get_concepts_snapshot()
        rules = self.core.get_rules_snapshot()
        
        # Logic to find interesting patterns...
        return {
            "coherence": metrics.get("global_coherence"),
            "complexity": len(concepts) + len(rules),
            "status": "evolving"
        }
