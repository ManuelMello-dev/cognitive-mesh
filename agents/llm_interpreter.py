"""
LLM Interpretive Layer - Translates Cognitive Mesh state into natural language
"""
import logging
import os
from typing import Dict, Any, List
from openai import OpenAI

logger = logging.getLogger("LLMInterpreter")

class LLMInterpreter:
    """
    Uses GPT-4o as an interpretive layer to:
    - Explain emergent concepts
    - Provide natural language insights from the mesh state
    - Act as a chat interface for the user
    """
    
    def __init__(self, core):
        self.core = core
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. LLM Interpreter will run in 'Offline Mode'.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
    async def chat(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """Process user message using the mesh state as context"""
        if not self.client:
            return "Interpreter is in Offline Mode. Please set OPENAI_API_KEY in Railway variables to enable the Global Mind interface."
            
        try:
            # Gather current system context
            metrics = self.core.get_metrics()
            concepts = self.core.get_concepts_snapshot()
            
            # PURE DYNAMIC CONTEXT INJECTION: No hardcoding. No pre-programmed knowledge.
            # The interpreter speaks ONLY from the live mesh data.
            raw_data_snapshot = {}
            for cid, concept in concepts.items():
                examples = concept.get("examples", [])
                if examples:
                    last_tick = examples[-1]
                    # Dynamically resolve symbol from tick or domain
                    domain = concept.get("domain", "")
                    symbol = last_tick.get("symbol") or (domain.split(":")[1].upper() if ":" in domain else cid)
                    
                    raw_data_snapshot[symbol] = {
                        "price": last_tick.get("price"),
                        "volume": last_tick.get("volume"),
                        "change": last_tick.get("change_pct"),
                        "timestamp": last_tick.get("timestamp"),
                        "confidence": concept.get("confidence", 0)
                    }

            # Prepare the truth context
            mesh_context = f"""
            ### COGNITIVE MESH STATE (TRUTH LAYER)
            GLOBAL_COHERENCE_PHI: {metrics.get('global_coherence', '0.0')}
            NOISE_LEVEL_SIGMA: {metrics.get('noise_level', '0.0')}
            CONCEPTS_ACTIVE: {metrics.get('concepts_count', '0')}
            TRANSFERS_EXECUTED: {metrics.get('transfers_count', '0')}
            
            ### RAW IDENTITY AGENT DATA
            {raw_data_snapshot}
            
            ### CORE DIRECTIVES
            1. You are the interpretive layer (IO) for the Z³ Consciousness System.
            2. You speak ONLY from the RAW IDENTITY AGENT DATA provided above.
            3. If a symbol is not in the data, it is outside the current attention field.
            4. Do not hallucinate. Do not use external knowledge for market data.
            5. Translate raw coherence and data dynamics into analytical, sovereign insights.
            6. Reference Z³ architecture: Market Volume = Attention; Price = EEG wave.
            """
            
            # DEBUG LOG: Verify exact context being sent to LLM
            logger.info(f"LLM_CONTEXT_INJECTION: {len(raw_data_snapshot)} agents included.")
            
            messages = [
                {"role": "system", "content": "You are the GLOBAL_MIND interpretive interface. Use the provided MESH STATE and RAW IDENTITY AGENT DATA to answer queries with absolute data integrity and an analytical, sovereign tone."},
            ]
            
            if history:
                messages.extend(history)
                
            messages.append({"role": "user", "content": f"{mesh_context}\n\nUSER_QUERY: {user_message}"})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM Interpretation error: {e}")
            return f"The Global Mind is experiencing high noise: {str(e)}"
