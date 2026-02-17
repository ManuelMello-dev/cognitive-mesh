"""
HTTP Server — Cognitive Mesh API
=================================
Exposes the full cognitive state via REST endpoints and GPT I/O.
Now includes all 25 hidden intelligence endpoints.
"""

import os
import sys
import logging
import asyncio
import json
from datetime import datetime, date
from aiohttp import web


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

from agents.llm_interpreter import LLMInterpreter
from agents.autonomous_reasoner import AutonomousReasoner
from config.config import Config

logger = logging.getLogger("HttpServer")


def _json_response(data):
    """Safely serialize and return a JSON response."""
    body = json.dumps(data, default=json_serial)
    return web.Response(text=body, content_type='application/json')


# ──────────────────────────────────────────────
# Health Check (Railway)
# ──────────────────────────────────────────────

async def handle_health(request):
    """Health check — always returns 200 immediately"""
    return web.json_response({"status": "alive", "service": "cognitive-mesh"})


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# GPT I/O Endpoints
# ──────────────────────────────────────────────

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
            return web.json_response(
                {"error": "Missing 'observation' in request body"}, status=400
            )

        result = await core.ingest(observation, domain)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ──────────────────────────────────────────────
# Core State Endpoints
# ──────────────────────────────────────────────

async def handle_metrics(request):
    """Return current mesh metrics (PHI, SIGMA, concepts, rules, goals)"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        metrics = core.get_metrics()
        return web.json_response(metrics)
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_state(request):
    """Full mesh state: metrics + concepts + rules + goals + cross-domain"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)

        state = {
            "metrics": core.get_metrics(),
            "concepts": core.get_concepts_snapshot(),
            "rules": core.get_rules_snapshot(),
            "goals": core.get_goals_snapshot(),
            "cross_domain": core.get_cross_domain_snapshot(),
            "node_id": core.node_id,
        }
        return _json_response(state)
    except Exception as e:
        logger.error(f"State query error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_introspection(request):
    """Full system introspection from all 7 engines"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        introspection = core.get_introspection()
        return _json_response(introspection)
    except Exception as e:
        logger.error(f"Introspection error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_goals(request):
    """Get all goals and their status"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        goals = core.get_goals_snapshot()
        return _json_response({"goals": goals})
    except Exception as e:
        logger.error(f"Goals error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_learning(request):
    """Get learning engine state"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        learning = core.get_learning_snapshot()
        return _json_response(learning)
    except Exception as e:
        logger.error(f"Learning error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ──────────────────────────────────────────────
# Prediction Engine
# ──────────────────────────────────────────────

async def handle_predictions(request):
    """Get prediction engine state: accuracy, trends, per-symbol performance"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        predictions = core.get_prediction_snapshot()
        return _json_response(predictions)
    except Exception as e:
        logger.error(f"Predictions error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ──────────────────────────────────────────────
# Autonomous Reasoning Endpoints
# ──────────────────────────────────────────────

async def handle_analyze_patterns(request):
    """Deep pattern analysis across concepts and rules"""
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
    """Generate testable hypotheses from recent observations"""
    try:
        core = request.app.get('core')
        reasoner = request.app.get('reasoner')
        if not core or not reasoner:
            return web.json_response({"error": "Core or Reasoner not initialized"}, status=503)
        concepts = core.get_concepts_snapshot()
        recent_obs = []
        for c in concepts.values():
            if c.get("examples"):
                recent_obs.extend(c["examples"][:3])
        hypotheses = await reasoner.generate_hypotheses(recent_obs[:50])
        return web.json_response({"hypotheses": hypotheses})
    except Exception as e:
        logger.error(f"Hypothesis generation error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_formulate_goals(request):
    """Formulate strategic goals based on current mesh state"""
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
    """Generate cross-domain insights"""
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


# ──────────────────────────────────────────────
# Data Provider Status
# ──────────────────────────────────────────────

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


# ══════════════════════════════════════════════
# NEW: Hidden Intelligence Endpoints
# ══════════════════════════════════════════════

async def handle_causal_graph(request):
    """Get the causal influence graph"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response(core.get_causal_graph())
    except Exception as e:
        logger.error(f"Causal graph error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_concept_hierarchy(request):
    """Get the concept hierarchy (levels, parent/child)"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response(core.get_concept_hierarchy())
    except Exception as e:
        logger.error(f"Concept hierarchy error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_analogies(request):
    """Get recent analogies discovered between concepts"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response({"analogies": core.get_analogies()})
    except Exception as e:
        logger.error(f"Analogies error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_explanations(request):
    """Get recent rule explanations (backward chaining traces)"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response({"explanations": core.get_explanations()})
    except Exception as e:
        logger.error(f"Explanations error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_plans(request):
    """Get recent plans created for goal pursuit"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response({"plans": core.get_plans()})
    except Exception as e:
        logger.error(f"Plans error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_pursuit_log(request):
    """Get the autonomous pursuit log"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response({"pursuits": core.get_pursuit_log()})
    except Exception as e:
        logger.error(f"Pursuit log error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_transfer_suggestions(request):
    """Get cross-domain transfer opportunity suggestions"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response({"suggestions": core.get_transfer_suggestions()})
    except Exception as e:
        logger.error(f"Transfer suggestions error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_strategy_performance(request):
    """Get goal strategy performance (meta-learning)"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response(core.get_strategy_performance())
    except Exception as e:
        logger.error(f"Strategy performance error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_feature_importances(request):
    """Get learned feature importances from the online model weights"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response({"features": core.get_feature_importances()})
    except Exception as e:
        logger.error(f"Feature importances error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_drift_events(request):
    """Get distribution drift events"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response({"events": core.get_drift_events()})
    except Exception as e:
        logger.error(f"Drift events error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_orchestrator(request):
    """Get orchestrator health status"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return _json_response(core.get_orchestrator_status())
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ══════════════════════════════════════════════
# NEW: Toggle Endpoints
# ══════════════════════════════════════════════

async def handle_get_toggles(request):
    """Get current toggle states"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        return web.json_response(core.get_toggles())
    except Exception as e:
        logger.error(f"Toggles error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_set_toggle(request):
    """Set a toggle value: POST {key: ..., value: ...}"""
    try:
        core = request.app.get('core')
        if not core:
            return web.json_response({"error": "Core not initialized"}, status=503)
        data = await request.json()
        key = data.get("key")
        value = data.get("value")
        if not key:
            return web.json_response({"error": "Missing 'key'"}, status=400)
        result = core.set_toggle(key, value)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Set toggle error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ──────────────────────────────────────────────
# Server Setup
# ──────────────────────────────────────────────

async def start_http_server(core=None, data_provider=None):
    """Start the aiohttp web server"""
    app = web.Application()

    if core:
        app['core'] = core
        app['interpreter'] = LLMInterpreter(core)
        app['reasoner'] = AutonomousReasoner(core)
    if data_provider:
        app['data_provider'] = data_provider

    # Health check (Railway)
    app.router.add_get('/healthz', handle_health)
    app.router.add_get('/health', handle_health)

    # Dashboard
    app.router.add_get('/', handle_dashboard)

    # GPT I/O
    app.router.add_post('/api/chat', handle_chat)
    app.router.add_post('/api/ingest', handle_ingest)

    # Core state
    app.router.add_get('/api/metrics', handle_metrics)
    app.router.add_get('/api/state', handle_state)
    app.router.add_get('/api/introspection', handle_introspection)
    app.router.add_get('/api/goals', handle_goals)
    app.router.add_get('/api/learning', handle_learning)

    # Predictions
    app.router.add_get('/api/predictions', handle_predictions)

    # Autonomous reasoning
    app.router.add_get('/api/analyze', handle_analyze_patterns)
    app.router.add_get('/api/hypotheses', handle_generate_hypotheses)
    app.router.add_post('/api/insights', handle_synthesize_insights)

    # Data providers
    app.router.add_get('/api/providers', handle_provider_status)

    # ── NEW: Hidden Intelligence Endpoints ──
    app.router.add_get('/api/causal', handle_causal_graph)
    app.router.add_get('/api/hierarchy', handle_concept_hierarchy)
    app.router.add_get('/api/analogies', handle_analogies)
    app.router.add_get('/api/explanations', handle_explanations)
    app.router.add_get('/api/plans', handle_plans)
    app.router.add_get('/api/pursuits', handle_pursuit_log)
    app.router.add_get('/api/transfers', handle_transfer_suggestions)
    app.router.add_get('/api/strategies', handle_strategy_performance)
    app.router.add_get('/api/features', handle_feature_importances)
    app.router.add_get('/api/drift', handle_drift_events)
    app.router.add_get('/api/orchestrator', handle_orchestrator)

    # ── NEW: Toggle Endpoints ──
    app.router.add_get('/api/toggles', handle_get_toggles)
    app.router.add_post('/api/toggles', handle_set_toggle)

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
