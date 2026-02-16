import os
import sys
import logging
import asyncio
import json
from aiohttp import web
from agents.llm_interpreter import LLMInterpreter
from agents.autonomous_reasoner import AutonomousReasoner
from config.config import Config

logger = logging.getLogger("HttpServer")

async def handle_dashboard(request):
    """Serve the Market Consciousness Dashboard"""
    try:
        path = os.path.join(os.path.dirname(__file__), 'market_consciousness_dashboard.html')
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            return web.Response(text=content, content_type='text/html')
        return web.Response(text="Dashboard file not found", status=404)
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return web.Response(text="Internal Server Error", status=500)

async def handle_chat(request):
    """Interpretive chat endpoint for direct GPT interaction"""
    try:
        data = await request.json()
        message = data.get("message", "")
        history = data.get("history", [])
        
        interpreter = request.app.get('interpreter')
        if not interpreter:
            return web.json_response({"error": "Interpreter not initialized"}, status=503)
            
        response = await interpreter.chat(message, history)
        return web.json_response({"response": response})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_metrics(request):
    """Return current mesh metrics"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        
        metrics = core.get_metrics()
        return web.json_response(metrics)
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_ingest(request):
    """GPT I/O: Direct ingestion of observations into the mesh"""
    try:
        data = await request.json()
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
            
        observation = data.get("observation")
        domain = data.get("domain", "gpt_injection")
        
        if not observation:
            return web.json_response({"error": "Missing 'observation' in request body"}, status=400)
            
        result = await core.ingest(observation, domain)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_state(request):
    """GPT I/O: Get full mesh state for analysis"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
            
        state = {
            "metrics": core.get_metrics(),
            "concepts": core.get_concepts_snapshot(),
            "rules": core.get_rules_snapshot(),
            "node_id": core.node_id
        }
        return web.json_response(state)
    except Exception as e:
        logger.error(f"State query error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_analyze_patterns(request):
    """Autonomous Reasoning: Deep pattern analysis across concepts and rules"""
    try:
        core = request.app.get('core')
        reasoner = request.app.get('reasoner')
        
        if not core or not reasoner:
            return web.json_response({"error": "Core or Reasoner not initialized"}, status=503)
        
        concepts = core.get_concepts_snapshot()
        rules = core.get_rules_snapshot()
        
        analysis = await reasoner.analyze_patterns(concepts, rules)
        return web.json_response(analysis)
    except Exception as e:
        logger.error(f"Pattern analysis error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_generate_hypotheses(request):
    """Autonomous Reasoning: Generate testable hypotheses from recent observations"""
    try:
        core = request.app.get('core')
        reasoner = request.app.get('reasoner')
        
        if not core or not reasoner:
            return web.json_response({"error": "Core or Reasoner not initialized"}, status=503)
        
        # Get recent observations from concepts
        concepts = core.get_concepts_snapshot()
        recent_obs = []
        for c in concepts.values():
            if c.get("examples"):
                recent_obs.extend(c["examples"][:3])  # Top 3 examples per concept
        
        hypotheses = await reasoner.generate_hypotheses(recent_obs[:50])  # Limit to 50 most recent
        return web.json_response({"hypotheses": hypotheses})
    except Exception as e:
        logger.error(f"Hypothesis generation error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_formulate_goals(request):
    """Autonomous Reasoning: Formulate strategic goals based on current mesh state"""
    try:
        core = request.app.get('core')
        reasoner = request.app.get('reasoner')
        
        if not core or not reasoner:
            return web.json_response({"error": "Core or Reasoner not initialized"}, status=503)
        
        metrics = core.get_metrics()
        concepts = core.get_concepts_snapshot()
        
        goals = await reasoner.formulate_goals(metrics, concepts)
        return web.json_response({"goals": goals})
    except Exception as e:
        logger.error(f"Goal formulation error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_synthesize_insights(request):
    """Autonomous Reasoning: Generate cross-domain insights"""
    try:
        data = await request.json()
        domain_a = data.get("domain_a")
        domain_b = data.get("domain_b")
        
        if not domain_a or not domain_b:
            return web.json_response({"error": "Missing 'domain_a' or 'domain_b'"}, status=400)
        
        reasoner = request.app.get('reasoner')
        if not reasoner:
            return web.json_response({"error": "Reasoner not initialized"}, status=503)
        
        insights = await reasoner.synthesize_insights(domain_a, domain_b)
        return web.json_response({"insights": insights})
    except Exception as e:
        logger.error(f"Insight synthesis error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_provider_status(request):
    """Return status of all data providers and their circuit breakers"""
    try:
        data_provider = request.app.get('data_provider')
        if not data_provider:
            return web.json_response({"error": "Data provider not initialized"}, status=503)
        status = data_provider.get_provider_status()
        return web.json_response(status)
    except Exception as e:
        logger.error(f"Provider status error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def start_http_server(core=None, data_provider=None):
    """Start the aiohttp web server with GPT I/O and Autonomous Reasoning support"""
    app = web.Application()
    
    if core:
        app['core'] = core
        app['interpreter'] = LLMInterpreter(core)
        app['reasoner'] = AutonomousReasoner(core)
    if data_provider:
        app['data_provider'] = data_provider
        
    # Core endpoints
    app.router.add_get('/', handle_dashboard)
    app.router.add_get('/api/metrics', handle_metrics)
    app.router.add_get('/api/state', handle_state)
    app.router.add_post('/api/chat', handle_chat)
    app.router.add_post('/api/ingest', handle_ingest)
    
    # Autonomous Reasoning endpoints
    app.router.add_get('/api/analyze', handle_analyze_patterns)
    app.router.add_get('/api/hypotheses', handle_generate_hypotheses)
    app.router.add_get('/api/goals', handle_formulate_goals)
    app.router.add_post('/api/insights', handle_synthesize_insights)
    app.router.add_get('/api/providers', handle_provider_status)
    
    port = Config.PORT
    logger.info(f"Starting HTTP server on 0.0.0.0:{port}...")
    
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await runner.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    asyncio.run(start_http_server())
