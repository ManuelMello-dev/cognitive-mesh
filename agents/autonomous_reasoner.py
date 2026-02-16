"""
Autonomous Reasoning Agent - Enhanced OpenAI Integration
Provides multi-step analysis, hypothesis generation, and autonomous decision-making
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger("AutonomousReasoner")

class AutonomousReasoner:
    """
    Advanced reasoning agent that uses OpenAI for:
    - Pattern recognition across concepts
    - Hypothesis generation for emergent behaviors
    - Multi-step causal analysis
    - Autonomous goal formulation
    """
    
    def __init__(self, core=None):
        self.core = core
        self.client = None
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            logger.info("AutonomousReasoner initialized with OpenAI client")
        else:
            logger.warning("AutonomousReasoner in offline mode - no OPENAI_API_KEY")
    
    async def analyze_patterns(self, concepts: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform deep pattern analysis across all concepts and rules.
        Returns insights about emergent behaviors and hidden relationships.
        """
        if not self.client or not concepts:
            return {"status": "offline", "insights": []}
        
        try:
            # Prepare context for analysis
            concept_summary = self._summarize_concepts(concepts)
            rule_summary = self._summarize_rules(rules)
            
            prompt = f"""You are an autonomous reasoning agent analyzing a cognitive mesh system.

CONCEPTS ({len(concepts)} total):
{concept_summary}

RULES ({len(rules)} total):
{rule_summary}

Perform a multi-step analysis:
1. Identify clusters of related concepts
2. Detect emergent patterns across domains
3. Hypothesize causal relationships
4. Suggest areas of high information density
5. Recommend focus areas for deeper investigation

Provide your analysis in structured JSON format."""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an analytical reasoning engine for a cognitive mesh. Respond with structured insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "status": "success",
                "analysis": analysis,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate_hypotheses(self, recent_observations: List[Dict[str, Any]]) -> List[str]:
        """
        Generate testable hypotheses about system behavior based on recent observations.
        """
        if not self.client or not recent_observations:
            return []
        
        try:
            obs_summary = "\n".join([
                f"- {obs.get('domain', '???')}: {obs.get('symbol', '???')} @ {obs.get('price', 'N/A')}"
                for obs in recent_observations[:20]
            ])
            
            prompt = f"""Based on these recent observations in a cognitive mesh:

{obs_summary}

Generate 3-5 testable hypotheses about:
1. Cross-domain correlations
2. Potential phase-locking patterns
3. Attention flow dynamics
4. Emerging coherence structures

Format as a numbered list."""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a hypothesis generation engine. Propose testable, falsifiable hypotheses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            hypotheses_text = response.choices[0].message.content
            # Parse numbered list
            hypotheses = [
                line.strip() 
                for line in hypotheses_text.split('\n') 
                if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10))
            ]
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Hypothesis generation error: {e}")
            return []
    
    async def formulate_goals(self, metrics: Dict[str, Any], concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Autonomously formulate goals for the pursuit agent based on current mesh state.
        """
        if not self.client:
            return []
        
        try:
            phi = metrics.get("global_coherence_phi", 0.5)
            sigma = metrics.get("noise_level_sigma", 0.1)
            concept_count = metrics.get("concepts_formed", 0)
            
            prompt = f"""You are an autonomous goal-setting agent for a cognitive mesh.

CURRENT STATE:
- PHI (coherence): {phi:.3f}
- SIGMA (noise): {sigma:.3f}
- Concepts formed: {concept_count}
- Regime: {"ORDERED" if phi > 0.7 else "CRITICAL" if phi > 0.4 else "CHAOS"}

Based on this state, formulate 2-3 actionable goals that would:
1. Optimize information flow
2. Enhance pattern recognition
3. Balance exploration vs exploitation
4. Improve cross-domain coherence

Format each goal as:
GOAL: [brief title]
RATIONALE: [why this goal matters now]
METRIC: [how to measure success]"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a strategic goal-setting agent. Propose concrete, measurable goals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            goals_text = response.choices[0].message.content
            
            # Parse goals (simplified - could be more sophisticated)
            goals = []
            for section in goals_text.split("GOAL:")[1:]:
                parts = section.split("RATIONALE:")
                if len(parts) >= 2:
                    title = parts[0].strip()
                    rationale_metric = parts[1].split("METRIC:")
                    rationale = rationale_metric[0].strip() if len(rationale_metric) > 0 else ""
                    metric = rationale_metric[1].strip() if len(rationale_metric) > 1 else ""
                    
                    goals.append({
                        "title": title,
                        "rationale": rationale,
                        "metric": metric,
                        "priority": 0.8,  # Default priority
                        "generated_by": "autonomous_reasoner"
                    })
            
            return goals
            
        except Exception as e:
            logger.error(f"Goal formulation error: {e}")
            return []
    
    async def synthesize_insights(self, domain_a: str, domain_b: str) -> Optional[str]:
        """
        Generate cross-domain insights by analyzing relationships between two domains.
        """
        if not self.client or not self.core:
            return None
        
        try:
            concepts = self.core.get_concepts_snapshot()
            
            # Filter concepts by domain
            concepts_a = [c for c in concepts.values() if c.get("domain", "").startswith(domain_a)]
            concepts_b = [c for c in concepts.values() if c.get("domain", "").startswith(domain_b)]
            
            if not concepts_a or not concepts_b:
                return None
            
            summary_a = self._summarize_concepts({c["id"]: c for c in concepts_a})
            summary_b = self._summarize_concepts({c["id"]: c for c in concepts_b})
            
            prompt = f"""Analyze the relationship between these two domains in a cognitive mesh:

DOMAIN A ({domain_a}):
{summary_a}

DOMAIN B ({domain_b}):
{summary_b}

Provide:
1. Potential correlations or phase-locking patterns
2. Information flow direction (A→B, B→A, or bidirectional)
3. Shared volatility or coherence signatures
4. Actionable insights for trading or decision-making"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a cross-domain insight synthesizer. Identify non-obvious relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Insight synthesis error: {e}")
            return None
    
    def _summarize_concepts(self, concepts: Dict[str, Any]) -> str:
        """Create a concise summary of concepts for LLM context"""
        if not concepts:
            return "No concepts available"
        
        lines = []
        for c in list(concepts.values())[:15]:  # Limit to top 15
            symbol = "???"
            price = "N/A"
            if c.get("examples") and len(c["examples"]) > 0:
                symbol = c["examples"][0].get("symbol", "???")
                price = c["examples"][0].get("price", "N/A")
            
            lines.append(f"- {c.get('domain', '???')}: {symbol} @ {price} (conf: {c.get('confidence', 0):.2f})")
        
        return "\n".join(lines)
    
    def _summarize_rules(self, rules: List[Dict[str, Any]]) -> str:
        """Create a concise summary of rules for LLM context"""
        if not rules:
            return "No rules learned yet"
        
        lines = []
        for r in rules[:10]:  # Limit to top 10
            lines.append(f"- {r.get('antecedent', '???')} → {r.get('consequent', '???')} (strength: {r.get('strength', 0):.2f})")
        
        return "\n".join(lines)
