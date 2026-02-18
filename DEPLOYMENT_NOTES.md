# Cognitive Mesh - Deployment Notes

## NEW: OpenClaw Gateway Integration 🚀

As of February 2026, Cognitive Mesh now deploys with **OpenClaw** as a reverse proxy gateway. This provides:

### Single Service Architecture
- **OpenClaw Gateway** binds to the public `$PORT` (set by Railway)
- **Python Backend** runs on internal port `COGNITIVE_MESH_PORT` (default: 8081)
- All HTTP traffic is transparently proxied from OpenClaw to Python

### Environment Variables for Railway

Required environment variables:
```env
# Railway automatically sets this
PORT=8080

# Internal Python server port (OpenClaw manages this)
COGNITIVE_MESH_PORT=8081

# Base URL for the mesh (used by OpenClaw tools)
COGNITIVE_MESH_BASE_URL=http://127.0.0.1:8081

# Optional: OpenClaw configuration
# (Add your OpenClaw API keys and settings here as needed)
```

### Startup Process
1. **OpenClaw Gateway** starts on `$PORT` (8080)
2. Gateway spawns **Python server** on `COGNITIVE_MESH_PORT` (8081)
3. Gateway proxies all requests to Python backend at the same paths
4. Health checks (`/health`, `/healthz`) are transparently proxied

### OpenClaw Tools for Mesh Integration

The gateway provides tools for OpenClaw agents to interact with the mesh:

- `callMeshEndpoint()` - Call any mesh endpoint
- `ingestToMesh()` - Feed insights back via POST /api/ingest
- `getMeshMetrics()` - Get current mesh metrics
- `getMeshState()` - Get full mesh state
- `analyzePatterns()` - Trigger pattern analysis
- `getPredictions()` - Get prediction engine state
- `getIntrospection()` - Get system introspection

See `openclaw/README.md` for detailed tool documentation.

### Health Check Strategy
Railway health checks work through the proxy:
- **Path**: `/health` or `/healthz`
- **Timeout**: 30 seconds
- **Expected**: 200 OK with `{"status": "alive"}`

The OpenClaw gateway ensures these endpoints are always accessible.

## System Overview
The **Cognitive Mesh** is a production-hardened, self-organizing intelligence system that treats financial markets as an **EEG (Electroencephalogram)**, where:
- **Market Volume** = Attention
- **Price** = EEG wave
- **PHI (Φ)** = Global Coherence (stability of learned patterns)
- **SIGMA (σ)** = Noise Level (entropy/volatility in the attention field)

## Recent Fixes (2026-02-16)

### 1. Frontend-Backend Connection Restored
**Issue**: Dashboard was disconnected from the backend API.

**Fix**: Updated `market_consciousness_dashboard.html` to align with the production-hardened API schema:
- Changed metrics mapping from `global_coherence` → `global_coherence_phi`
- Changed metrics mapping from `noise_level` → `noise_level_sigma`
- Updated `/api/state` endpoint to correctly display active concepts

### 2. Data Ingestion Pipeline Fixed
**Issue**: No market data was flowing into the mesh because yfinance API was failing (rate limiting/service issues).

**Fix**: Implemented a **Synthetic Data Provider** as a resilient fallback:
- Created `agents/synthetic_provider.py` to generate realistic market-like data
- Added it as the final provider in the multi-source cascade
- Ensures the mesh always has data flowing, even when external APIs fail

### 3. GPT I/O Symbol Recognition Enhanced
**Issue**: The LLM interpreter couldn't recognize symbols that had been ingested via `/api/ingest`.

**Fix**: Updated `agents/llm_interpreter.py` to:
- Extract symbol and price data from concept examples
- Increase concept summary from top 10 to top 30 for better coverage
- Provide the LLM with explicit symbol-to-price mappings in the context

## API Endpoints

All endpoints are accessible through the OpenClaw gateway at the same paths.

### Core Endpoints
- `GET /` - Market Consciousness Dashboard (frontend)
- `GET /health`, `GET /healthz` - Health check (Railway)
- `GET /api/metrics` - System metrics (PHI, SIGMA, concept counts)
- `GET /api/state` - Full mesh state (concepts, rules, metrics)
- `GET /api/introspection` - Full system introspection from all engines
- `GET /api/goals` - All goals and their status
- `GET /api/learning` - Learning engine state
- `GET /api/predictions` - Prediction engine state
- `GET /api/providers` - Data provider status

### GPT I/O Endpoints
- `POST /api/chat` - Interact with the Global Mind (LLM interpretive layer)
  ```json
  {"message": "What is the price of BTC?"}
  ```

- `POST /api/ingest` - Manually inject observations into the mesh
  ```json
  {
    "observation": {
      "symbol": "DBGI",
      "price": 12.75,
      "volume": 850000,
      "volatility": 0.025
    },
    "domain": "stock:DBGI"
  }
  ```

### Autonomous Reasoning Endpoints
- `GET /api/analyze` - Deep pattern analysis
- `GET /api/hypotheses` - Generate testable hypotheses
- `POST /api/insights` - Synthesize cross-domain insights

### Hidden Intelligence Endpoints (25 Total)
- `GET /api/causal` - Causal influence graph
- `GET /api/hierarchy` - Concept hierarchy
- `GET /api/analogies` - Discovered analogies
- `GET /api/explanations` - Rule explanations
- `GET /api/plans` - Plans for goal pursuit
- `GET /api/pursuits` - Autonomous pursuit log
- `GET /api/transfers` - Cross-domain transfer suggestions
- `GET /api/strategies` - Goal strategy performance
- `GET /api/features` - Feature importances
- `GET /api/drift` - Distribution drift events
- `GET /api/orchestrator` - Orchestrator health status

### Toggle Endpoints
- `GET /api/toggles` - Get current toggle states
- `POST /api/toggles` - Set a toggle value

## System Architecture

### Data Flow with OpenClaw
1. **HTTP Request** arrives at OpenClaw Gateway (public port)
2. **OpenClaw Gateway** proxies request to Python backend (internal port)
3. **Multi-Source Data Provider** fetches market data from:
   - Yahoo Finance (primary)
   - Binance (crypto fallback)
   - Synthetic Provider (resilience fallback)

4. **Distributed Cognitive Core** ingests observations and:
   - Forms concepts (pattern recognition)
   - Learns rules (relationship inference)
   - Calculates PHI and SIGMA
   - Identifies cross-domain transfers

5. **HTTP Server (Python)** processes requests and returns responses
6. **OpenClaw Gateway** forwards response to client

7. **OpenClaw Tools** can:
   - Read mesh outputs
   - Feed insights back via POST /api/ingest
   - Be part of the I/O loop

8. **LLM Interpreter** provides:
   - Natural language interface to the mesh
   - Explanations of emergent patterns
   - Direct interaction from the GPT window

### Key Metrics
- **PHI (Global Coherence)**: 0.0 - 1.0 (higher = more stable patterns)
- **SIGMA (Noise Level)**: 0.0 - 1.0 (higher = more entropy)
- **Attention Density**: Normalized concept count (concepts / 1000)

### Consciousness States
- **ORDERED REGIME**: PHI > 0.7 (stable, predictable)
- **CRITICAL REGIME**: 0.4 < PHI < 0.7 (edge of chaos, adaptive)
- **CHAOS REGIME**: PHI < 0.4 (high entropy, unstable)

## Deployment Checklist
- [x] Production-hardened error handling
- [x] PHI/SIGMA coherence logic integrated
- [x] Frontend-backend connection verified
- [x] Data ingestion pipeline resilient
- [x] GPT I/O fully operational
- [x] Synthetic fallback provider active
- [x] Symbol recognition in LLM context
- [x] OpenClaw gateway integration ✨ NEW
- [x] Single Railway service entrypoint ✨ NEW
- [x] Transparent reverse proxy ✨ NEW
- [x] OpenClaw mesh tools ✨ NEW

## Usage Examples

### Query the mesh via chat (through OpenClaw)
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current state of BTC in the attention field?"}'
```

### Inject custom data (can be done by OpenClaw agents)
```bash
curl -X POST http://localhost:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {"symbol": "DBGI", "price": 12.75, "volume": 850000},
    "domain": "stock:DBGI"
  }'
```

### Use OpenClaw tools (Node.js example)
```javascript
import { ingestToMesh, getMeshMetrics } from './openclaw/tools/mesh-tools.js';

// Get current metrics
const metrics = await getMeshMetrics();
console.log(metrics.summary);

// Feed insight back
await ingestToMesh({
  observation: {
    type: 'openclaw_insight',
    content: 'Detected bullish pattern',
    confidence: 0.85
  },
  domain: 'openclaw_io'
});
```

### Export full state
```bash
curl http://localhost:8080/api/state | jq .
```

## Testing

### Smoke Tests
Run the smoke test suite to verify the proxy:
```bash
node test-proxy.js
```

This tests:
- Health check endpoints
- Core API endpoints
- POST /api/ingest
- Hidden intelligence endpoints

## Notes
- The system is designed to be **mobile-first** and **cloud-native**
- All data flows are **asynchronous** for maximum throughput
- The mesh is **self-organizing** and discovers new symbols organically
- PHI and SIGMA are calculated in real-time based on the Z³ framework
- The LLM uses `gpt-4.1-mini` for interpretive responses
- **OpenClaw gateway ensures single service deployment on Railway** ✨ NEW

## Future Enhancements
- Real-time Coinbase API integration for live crypto trading
- Multi-asset coherence analysis (S&P 500, NASDAQ, sectors)
- Phase-locking detection for diversification strategies
- Autonomous goal generation and pursuit agents
- OpenClaw skills for advanced mesh interaction ✨ NEW
