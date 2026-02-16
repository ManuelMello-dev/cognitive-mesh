import os
import logging
import asyncio
from aiohttp import web
from agents.llm_interpreter import LLMInterpreter

logger = logging.getLogger("HttpServer")

async def handle_dashboard(request):
    """Serve the Market Consciousness Dashboard"""
    try:
        path = os.path.join(os.path.dirname(__file__), 'market_consciousness_dashboard.html')
        with open(path, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return web.Response(text="Dashboard file not found", status=404)

async def handle_eeg(request):
    """Serve the Market EEG Monitor"""
    try:
        path = os.path.join(os.path.dirname(__file__), 'market_eeg_monitor.html')
        with open(path, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')
    except Exception as e:
        logger.error(f"Error serving EEG monitor: {e}")
        return web.Response(text="EEG Monitor file not found", status=404)

async def handle_chat(request):
    """Interpretive chat endpoint"""
    try:
        data = await request.json()
        message = data.get("message", "")
        history = data.get("history", [])
        
        # Access the interpreter via the app state
        interpreter = request.app.get('interpreter')
        if not interpreter:
            return web.json_response({"response": "The Global Mind is currently offline (interpreter not initialized)."}, status=503)
            
        response = await interpreter.chat(message, history)
        return web.json_response({"response": response})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_metrics(request):
    """Return current mesh metrics for the dashboard"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        
        metrics = core.get_metrics()
        # Add extra info for the dashboard grid
        active_concepts = core.get_concepts_snapshot()
        metrics['active_symbols'] = list(active_concepts.keys())
        
        return web.json_response(metrics)
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def start_http_server(core=None):
    """Start the aiohttp web server"""
    app = web.Application()
    
    # Store core and interpreter in app state if core is provided
    if core:
        app['core'] = core
        app['interpreter'] = LLMInterpreter(core)
        
    app.router.add_get('/', handle_dashboard)
    app.router.add_get('/dashboard', handle_dashboard)
    app.router.add_get('/eeg', handle_eeg)
    app.router.add_get('/api/metrics', handle_metrics)
    app.router.add_post('/api/chat', handle_chat)
    
    # Railway typically expects port 8080 for web services
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Attempting to bind HTTP server to 0.0.0.0:{port}")
    
    runner = web.AppRunner(app, access_log=logger)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    
    logger.info(f"Starting HTTP server on port {port}...")
    await site.start()
    
    # Return the runner to keep it alive or just wait forever if used in gather
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await runner.cleanup()

if __name__ == "__main__":
    # For standalone testing
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(start_http_server())
    except KeyboardInterrupt:
        pass
