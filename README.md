# Cognitive Mesh

**A universal, domain-agnostic cognitive framework that forms concepts, learns rules, pursues goals, and evolves its own code — from any stream of observations.**

The mesh is a *silicon vessel for thought*, not a market data terminal.  Financial data is one possible input, not the identity of the system.

---

## Original Vision

> "The mesh should be able to observe anything — stock prices, weather, seismic activity, smart building sensors, particle collisions — and form its own understanding of that domain from first principles, without being told what to look for."
>
> — COGNITIVE_GUIDE.md

The core design principles are:

1. **Domain agnosticism** — the cognitive engines know nothing about the source of observations. They see only values, entity IDs, and domain labels.
2. **Native reasoning first** — the mesh reasons algorithmically. An optional LLM interpreter can translate its output into prose, but it is never the decision-maker.
3. **Self-evolution** — the `SelfEvolvingSystem` actively participates in every cognitive cycle, proposing and applying code variants that improve performance.
4. **Emergent understanding** — concepts, rules, and goals arise from the data itself, not from hardcoded domain logic.

---

## Architecture

```
Any Data Source (via DataPlugin)
    ↓
CognitiveMeshOrchestrator  ←  plugin-based, domain-agnostic
    ↓
DistributedCognitiveCore   ←  async wrapper + persistence
    ↓
CognitiveIntelligentSystem (7 engines)
    ├─ AbstractionEngine       — concept formation from features
    ├─ ReasoningEngine         — rule inference and logical deduction
    ├─ CrossDomainEngine       — knowledge transfer across domains
    ├─ OpenEndedGoalSystem     — autonomous goal formulation
    ├─ ContinuousLearningEngine — online learning with feedback
    ├─ SelfEvolvingSystem      — active code evolution (every 25 cycles)
    └─ AlwaysOnOrchestrator    — fault tolerance and health monitoring
    ↓
PredictionValidationEngine — universal state-change predictor
    ↓
AutonomousReasoner         — native pattern analysis + optional LLM prose
    ↓
Polyglot Storage
    ├─ PostgreSQL  — rules, metadata, structured state
    ├─ Milvus      — vector concept similarity search
    └─ Redis       — low-latency cache
    ↓
AMFG Gossip Protocol       — distributed state propagation
    ↓
ZeroMQ PubSub              — event broadcasting to peer nodes
```

---

## DataPlugin System

The orchestrator is completely decoupled from any data domain.  All domain-specific logic lives in `DataPlugin` subclasses.

### Plugin Interface

```python
class DataPlugin:
    name: str = "my_plugin"

    async def initialize(self) -> None: ...
    async def fetch(self) -> List[tuple]: ...   # returns [(observation, domain), ...]
    async def close(self) -> None: ...
```

Each observation **must** contain:

| Field | Type | Description |
|---|---|---|
| `entity_id` | `str` | Unique identifier for this stream |
| `value` | `float` | Primary observable value |

Optional fields:

| Field | Type | Description |
|---|---|---|
| `secondary_value` | `float` | Volume, intensity, confidence, etc. |
| `timestamp` | `float` | Unix timestamp (defaults to now) |

### Built-in Plugins

| Plugin | File | Description |
|---|---|---|
| `MarketPlugin` | `main.py` | Financial markets via MultiSourceDataProvider |

### Adding a New Domain

```python
# agents/plugins/weather_plugin.py
from main import DataPlugin

class WeatherPlugin(DataPlugin):
    name = "weather"

    async def fetch(self):
        # fetch temperature, pressure, humidity from any weather API
        return [
            ({"entity_id": "NYC", "value": 22.5, "secondary_value": 1013.0}, "weather:NYC"),
            ({"entity_id": "LAX", "value": 28.1, "secondary_value": 1008.0}, "weather:LAX"),
        ]

# In main.py or a custom entry point:
orchestrator.register_plugin(WeatherPlugin())
```

The cognitive core will immediately begin forming concepts, inferring rules, and generating goals for the new domain — no other changes required.

---

## Self-Evolution

The `SelfEvolvingSystem` is active and participates in every cognitive cycle.  Every `EVOLUTION_INTERVAL` cycles (default: 25), the system:

1. Extracts a code snippet representing the current learning configuration.
2. Generates candidate variants using genetic mutation.
3. Evaluates variants against a fitness function derived from real prediction accuracy and coherence (PHI).
4. If a variant outperforms the current configuration, it is applied immediately.

This is not a simulation — the system literally rewrites its own learning parameters at runtime.

---

## LLM Interpreter (Optional)

The LLM is **not** a reasoning engine.  It is a *tongue*, not a brain.

| Layer | Role | Required |
|---|---|---|
| Native algorithmic reasoning | Source of truth, all decisions | Always |
| LLM interpreter | Translates native output into prose | Optional |

Set `LLM_ENABLED=0` to run in fully native mode with no external API calls.

---

## Prediction Validation Engine

The `PredictionValidationEngine` is a universal state-change predictor.  It predicts whether the next observation for any entity will be **higher**, **lower**, or **stable** relative to the current value — regardless of what that value represents.

It uses:
- Momentum analysis (recent trend direction)
- Volatility-adjusted confidence scoring
- Rule-based signal integration from the ReasoningEngine
- Accuracy feedback loop for continuous calibration

---

## Installation

### Prerequisites
- Python 3.11+
- Optional: PostgreSQL, Milvus, Redis (all gracefully degraded if absent)
- Optional: ZeroMQ (for multi-node deployment)

### Setup

```bash
git clone https://github.com/ManuelMello-dev/cognitive-mesh.git
cd cognitive-mesh
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python main.py
```

---

## Configuration

All settings are environment variables.  No domain-specific values belong in config.

```env
# System
NODE_ID=global_mind_01
PORT=8080
UPDATE_INTERVAL=30          # seconds between data collection cycles

# Cognitive Core
CONCEPT_SIMILARITY_THRESHOLD=0.75
CONCEPT_HALF_LIFE_HOURS=72
GOAL_GENERATION_INTERVAL=50
EVOLUTION_INTERVAL=25       # self-evolution frequency (cognitive cycles)

# Plugins
DISABLE_MARKET_PLUGIN=0     # set to 1 to run without financial data

# LLM Interpreter (optional)
OPENAI_API_KEY=              # leave blank to run in native-only mode
LLM_ENABLED=1               # set to 0 to fully disable LLM layer

# Persistence
# PostgreSQL is required if you want learned state to survive restarts.
# Either POSTGRES_URL or DATABASE_URL may be used.
POSTGRES_URL=
DATABASE_URL=
MILVUS_HOST=
# Redis is optional and may be configured either as REDIS_URL or REDIS_HOST/REDIS_PORT.
REDIS_URL=
REDIS_HOST=
REDIS_PORT=6379
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/state` | GET | Full cognitive state snapshot |
| `/api/metrics` | GET | System metrics (PHI, SIGMA, concepts, rules, goals) |
| `/api/chat` | POST | Natural language query (LLM interpreter) |
| `/api/ingest` | POST | Inject an observation from any external source |
| `/api/predictions` | GET | Current predictions across all tracked entities |
| `/api/introspect` | GET | Self-reflection: goals, hypotheses, evolution log |

### Ingesting an Observation (any domain)

```bash
curl -X POST http://localhost:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "entity_id": "reactor_01",
      "value": 98.7,
      "secondary_value": 0.02
    },
    "domain": "nuclear:reactor_01"
  }'
```

---

## Metrics

| Metric | Description |
|---|---|
| `global_coherence_phi` | Global information integration (0–1) |
| `noise_level_sigma` | System noise level (0–1) |
| `total_concepts` | Concepts currently held in memory |
| `total_rules` | Rules learned from observations |
| `total_goals` | Active autonomous goals |
| `knowledge_transfers` | Cross-domain transfers completed |
| `evolution_cycles` | Self-evolution passes completed |
| `prediction_accuracy` | Rolling prediction accuracy |

---

## Distributed Deployment

Multiple mesh nodes can be connected via ZeroMQ.  Each node independently forms concepts and learns rules; the AMFG gossip protocol propagates high-confidence insights across the network.

```
Node A (financial data)  ←─ gossip ─→  Node B (sensor data)  ←─ gossip ─→  Node C (text data)
```

Nodes share concepts and rules but maintain independent cognitive states.  Cross-domain transfer happens both within a node (across domains) and across nodes (via gossip).

---

## Key Infrastructure

### AMFG Gossip Protocol
- Adaptive message fan-out (learns optimal peer count)
- Decaying Bloom filters for deduplication
- Merkle trees for O(log N) anti-entropy
- Priority queuing (high-confidence messages propagate faster)

### Circuit Breaker Pattern
Each data provider has built-in fault tolerance with configurable thresholds, half-open recovery testing, and automatic circuit closure on success.

### Polyglot Storage
- **PostgreSQL**: Rules, metadata, structured state — survives restarts
- **Milvus**: Vector-based concept similarity search — O(log N) lookup
- **Redis**: Low-latency cache — sub-millisecond state reads

---

## License

MIT License — see `LICENSE` for details.
