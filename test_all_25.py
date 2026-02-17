"""
Test all 25 audit items: engine wiring, API snapshots, toggles.
Runs offline (no HTTP server needed) — tests the core objects directly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cognitive_intelligent_system import CognitiveIntelligentSystem
from core.distributed_core import DistributedCognitiveCore
from continuous_learning_engine import ContinuousLearningEngine
import json

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")

print("=" * 60)
print("TESTING ALL 25 AUDIT ITEMS")
print("=" * 60)

# ── 1. Instantiate core ──
print("\n[1] Instantiate DistributedCognitiveCore")
try:
    core = DistributedCognitiveCore(node_id='test-node')
    check("Core created", True)
except Exception as e:
    check("Core created", False, str(e))
    sys.exit(1)

cs = core.cognitive_system

# ── 2. Toggles exist ──
print("\n[2] Toggles")
toggles = core.get_toggles()
check("Toggles dict exists", isinstance(toggles, dict))
check("causal_discovery toggle", 'causal_discovery' in toggles)
check("self_evolution toggle", 'self_evolution' in toggles)
check("autonomous_reasoning toggle", 'autonomous_reasoning' in toggles)
check("prediction_horizon toggle", 'prediction_horizon' in toggles)
check("dead_zone_sensitivity toggle", 'dead_zone_sensitivity' in toggles)

# ── 3. Set toggle ──
print("\n[3] Set toggle")
result = core.set_toggle('prediction_horizon', 10)
check("Set prediction_horizon=10", result.get('prediction_horizon') == 10)
result = core.set_toggle('dead_zone_sensitivity', 'aggressive')
check("Set dead_zone=aggressive", result.get('dead_zone_sensitivity') == 'aggressive')

# ── 4. Ingest observations to populate state ──
print("\n[4] Ingest test observations")
import asyncio

async def ingest_test_data():
    symbols = ['BTC', 'ETH', 'SOL', 'AAPL', 'MSFT']
    for i in range(30):
        for sym in symbols:
            obs = {
                'symbol': sym,
                'price': 50000 + i * 100 if sym == 'BTC' else 3000 + i * 10,
                'volume': 1000000 + i * 10000,
                'change_24h': 0.5 + i * 0.1,
            }
            domain = f"crypto:{sym}" if sym in ['BTC','ETH','SOL'] else f"stock:{sym}"
            await core.ingest(obs, domain)

asyncio.run(ingest_test_data())
check("Ingested 150 observations", core._observation_count >= 150)

# ── 5. Snapshot methods exist and return data ──
print("\n[5] Snapshot methods")

# Causal graph
cg = core.get_causal_graph()
check("get_causal_graph returns dict", isinstance(cg, dict))
check("causal graph has 'total_links'", 'total_links' in cg)

# Concept hierarchy
ch = core.get_concept_hierarchy()
check("get_concept_hierarchy returns dict", isinstance(ch, dict))

# Analogies
an = core.get_analogies()
check("get_analogies returns list", isinstance(an, list))

# Explanations
ex = core.get_explanations()
check("get_explanations returns list", isinstance(ex, list))

# Plans
pl = core.get_plans()
check("get_plans returns list", isinstance(pl, list))

# Pursuit log
pu = core.get_pursuit_log()
check("get_pursuit_log returns list", isinstance(pu, list))

# Transfer suggestions
ts = core.get_transfer_suggestions()
check("get_transfer_suggestions returns list", isinstance(ts, list))

# Strategy performance
sp = core.get_strategy_performance()
check("get_strategy_performance returns dict", isinstance(sp, dict))

# Feature importances
fi = core.get_feature_importances()
check("get_feature_importances returns list", isinstance(fi, list))

# Drift events
de = core.get_drift_events()
check("get_drift_events returns list", isinstance(de, list))

# Orchestrator
orch = core.get_orchestrator_status()
check("get_orchestrator_status returns dict", isinstance(orch, dict))

# ── 6. Metrics include new fields ──
print("\n[6] Metrics include new fields")
metrics = core.get_metrics()
check("causal_links_discovered in metrics", 'causal_links_discovered' in metrics)
check("prediction_accuracy in metrics", 'prediction_accuracy' in metrics)

# ── 7. Introspection includes new sections ──
print("\n[7] Introspection includes hidden intelligence")
intro = core.get_introspection()
check("causal_graph in introspection", 'causal_graph' in intro)
check("concept_hierarchy in introspection", 'concept_hierarchy' in intro)
check("recent_analogies in introspection", 'recent_analogies' in intro)
check("recent_explanations in introspection", 'recent_explanations' in intro)
check("recent_plans in introspection", 'recent_plans' in intro)
check("pursuit_log in introspection", 'pursuit_log' in intro)
check("transfer_suggestions in introspection", 'transfer_suggestions' in intro)
check("strategy_performance in introspection", 'strategy_performance' in intro)
check("toggles in introspection", 'toggles' in intro)

# ── 8. Learning engine drift events ──
print("\n[8] Learning engine drift tracking")
le = ContinuousLearningEngine(feature_dim=10)
for i in range(100):
    le.process_observation({'price': 100 + i * 0.1, 'volume': 1000})
# Inject a big drift
for i in range(30):
    le.process_observation({'price': 500 + i, 'volume': 50000})
insights = le.get_insights()
check("feature_importances in insights", 'feature_importances' in insights)
check("drift_events in insights", 'drift_events' in insights)
check("lr_history in insights", 'lr_history' in insights)
fi2 = le.get_feature_importances()
check("Feature importances populated", len(fi2) > 0, f"got {len(fi2)}")

# ── 9. Cognitive system caches ──
print("\n[9] Cognitive system intelligence caches")
check("_recent_analogies deque exists", hasattr(cs, '_recent_analogies'))
check("_recent_explanations deque exists", hasattr(cs, '_recent_explanations'))
check("_recent_plans deque exists", hasattr(cs, '_recent_plans'))
check("_causal_discovery_log deque exists", hasattr(cs, '_causal_discovery_log'))
check("_pursuit_log deque exists", hasattr(cs, '_pursuit_log'))
check("_transfer_suggestions_cache exists", hasattr(cs, '_transfer_suggestions_cache'))

# ── 10. JSON serialization ──
print("\n[10] JSON serialization of all snapshots")
from datetime import datetime, date
def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

for name, data in [
    ('metrics', metrics),
    ('causal_graph', cg),
    ('concept_hierarchy', ch),
    ('introspection', intro),
    ('toggles', toggles),
]:
    try:
        json.dumps(data, default=json_serial)
        check(f"{name} serializable", True)
    except Exception as e:
        check(f"{name} serializable", False, str(e))

# ── Summary ──
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"FAILURES: {FAIL}")
print("=" * 60)
