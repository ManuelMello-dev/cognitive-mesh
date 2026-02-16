import os
import logging
import asyncio
import json
from aiohttp import web
from agents.llm_interpreter import LLMInterpreter
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

async def start_http_server(core=None):
    """Start the aiohttp web server with GPT I/O support"""
    app = web.Application()
    
    if core:
        app['core'] = core
        app['interpreter'] = LLMInterpreter(core)
        
    app.router.add_get('/', handle_dashboard)
    app.router.add_get('/api/metrics', handle_metrics)
    app.router.add_get('/api/state', handle_state)
    app.router.add_post('/api/chat', handle_chat)
    app.router.add_post('/api/ingest', handle_ingest)
    
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
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start_http_server())
