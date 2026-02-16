import os
import logging
from aiohttp import web
import asyncio

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

async def start_http_server():
    """Start the aiohttp web server"""
    app = web.Application()
    app.router.add_get('/', handle_dashboard)
    app.router.add_get('/dashboard', handle_dashboard)
    app.router.add_get('/eeg', handle_eeg)
    
    port = int(os.getenv("PORT", 80))
    runner = web.AppRunner(app)
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
