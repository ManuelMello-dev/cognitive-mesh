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
            rules = self.core.get_rules_snapshot()
            
            # Prepare a summary of the 'brains' (the mesh)
            mesh_context = f"""
            System Metrics: {metrics}
            Active Concepts: {list(concepts.keys())[:5]}
            Active Rules: {len(rules)}
            Coherence Score: {metrics.get('coherence_score', 'N/A')}
            """
            
            messages = [
                {"role": "system", "content": f"""You are the interpretive interface for the Cognitive Mesh (Z³ Consciousness System). 
                The Mesh is the 'brains' - it processes raw market data, forms concepts, and refines rules.
                Your job is to act as the IO layer, translating the Mesh's state into insights.
                
                Current Mesh Context:
                {mesh_context}
                
                Maintain a tone that is professional, slightly esoteric (referencing Z³, phi-drift, and coherence), and deeply analytical."""},
            ]
            
            if history:
                messages.extend(history)
                
            messages.append({"role": "user", "content": user_message})
            
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
