# GPT I/O Guide - Direct Mesh Interaction

## Overview
The **Cognitive Mesh** is designed to be interacted with directly from any GPT interface (ChatGPT, Claude, etc.) via its HTTP API endpoints. This enables you to **inject observations**, **query the mesh state**, and **chat with the interpretive layer** from anywhere.

## Your Deployed Mesh
Based on your logs, your mesh is running at:
- **Service ID**: `38bdf345-146b-41ef-ac9c-a59ce57d014a`
- **Deployment**: `c315e841-9f3d-4b2d-abee-a6a8a395fc90`
- **Status**: ✅ Active and operational
- **Mode**: Organic data only (no hardcoded symbols)

You'll need to replace `YOUR_MESH_URL` in the examples below with your actual deployment URL (e.g., `https://your-mesh.railway.app` or similar).

## API Endpoints

### 1. Dashboard (Visual Interface)
```
GET https://YOUR_MESH_URL/
```
Opens the Market Consciousness Monitor dashboard in your browser.

### 2. Get Mesh State
```bash
curl https://YOUR_MESH_URL/api/state
```

**Returns**:
```json
{
  "metrics": {
    "concepts_formed": 3,
    "global_coherence_phi": 0.74,
    "noise_level_sigma": 0.452,
    "attention_density": 0.003
  },
  "concepts": { ... },
  "rules": { ... },
  "node_id": "global_mind_01"
}
```

### 3. Ingest Observation (Manual Data Injection)
```bash
curl -X POST https://YOUR_MESH_URL/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "symbol": "BTC",
      "price": 65420.0,
      "volume": 2500000,
      "volatility": 0.022
    },
    "domain": "crypto:BTC"
  }'
```

**Returns**:
```json
{
  "success": true,
  "iteration": 1,
  "concept_id": "concept_abc123...",
  "new_rules": 2,
  "concept_count": 1,
  "phi": 0.7,
  "sigma": 0.48,
  "timestamp": "2026-02-16T19:59:29.864658+00:00"
}
```

### 4. Chat with the Global Mind (GPT Interpretive Layer)
```bash
curl -X POST https://YOUR_MESH_URL/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the current state of BTC in the attention field?"
  }'
```

**Returns**:
```json
{
  "response": "The current state of BTC in the attention field is characterized by a price of 65,420.00 with a volume (attention) of 2,500,000. The global coherence (PHI) is 0.7, indicating moderately stable learned patterns..."
}
```

## Interacting from GPT Window

### Example 1: Bootstrap the Mesh
**You say to GPT**:
> "Call this API to inject BTC data into my cognitive mesh: POST https://YOUR_MESH_URL/api/ingest with body: {\"observation\": {\"symbol\": \"BTC\", \"price\": 65420.0, \"volume\": 2500000}, \"domain\": \"crypto:BTC\"}"

**GPT** will make the API call and confirm the concept was formed.

### Example 2: Query the Mesh
**You say to GPT**:
> "What's the current state of my cognitive mesh? Call GET https://YOUR_MESH_URL/api/state"

**GPT** will retrieve and summarize the metrics, concepts, and coherence levels.

### Example 3: Chat with the Mesh
**You say to GPT**:
> "Ask my cognitive mesh what assets are in the attention field. Use POST https://YOUR_MESH_URL/api/chat with message: 'What assets are currently in the attention field?'"

**GPT** will relay the mesh's interpretive response.

### Example 4: Inject Multiple Observations
**You say to GPT**:
> "Inject these observations into my mesh:
> 1. BTC at $65420, volume 2.5M
> 2. ETH at $3450, volume 1.8M
> 3. AAPL at $180.25, volume 52M
> Use the /api/ingest endpoint."

**GPT** will make 3 sequential API calls and report the results.

## Organic Discovery Workflow

1. **Start with blank slate** (mesh has 0 concepts)
2. **Manually inject seed observations** via `/api/ingest`
3. **Mesh forms concepts** and extracts symbols
4. **Mesh attempts to fetch real-time data** from yfinance/Binance for discovered symbols
5. **Concepts evolve** as new observations arrive
6. **Query the mesh** via `/api/chat` to understand emergent patterns

## Custom Observation Schema

You can inject **any type of observation**, not just market data:

### Example: Sensor Data
```json
{
  "observation": {
    "sensor_id": "temp_01",
    "temperature": 72.5,
    "humidity": 45.2,
    "timestamp": 1771271699
  },
  "domain": "sensor:temperature"
}
```

### Example: Social Sentiment
```json
{
  "observation": {
    "topic": "AI",
    "sentiment_score": 0.85,
    "engagement": 15000,
    "source": "twitter"
  },
  "domain": "social:AI"
}
```

### Example: Custom Metrics
```json
{
  "observation": {
    "metric_name": "user_engagement",
    "value": 1250,
    "delta": 0.15,
    "category": "growth"
  },
  "domain": "metrics:engagement"
}
```

The mesh will form concepts from **any structured observation** you provide.

## GPT I/O Best Practices

### 1. Use Descriptive Domains
- `stock:AAPL` for stocks
- `crypto:BTC` for cryptocurrencies
- `sensor:temp_01` for IoT data
- `social:AI` for sentiment analysis

### 2. Include Contextual Fields
Add fields like `volume`, `volatility`, `confidence`, `source` to help the mesh learn richer patterns.

### 3. Query Strategically
Ask the mesh:
- "What patterns have emerged?"
- "Which domains are showing high coherence?"
- "What cross-domain transfers have been identified?"

### 4. Monitor PHI and SIGMA
- **PHI > 0.7**: Stable, predictable patterns (ORDERED regime)
- **0.4 < PHI < 0.7**: Edge of chaos, adaptive (CRITICAL regime)
- **PHI < 0.4**: High entropy, unstable (CHAOS regime)

## Troubleshooting

### "Symbol not in attention field"
**Cause**: The mesh hasn't formed a concept for that symbol yet.
**Solution**: Inject an observation via `/api/ingest` first.

### "Connection failed"
**Cause**: Mesh is not running or URL is incorrect.
**Solution**: Check deployment logs and verify the URL.

### "No response from mesh"
**Cause**: HTTP server may not have started.
**Solution**: Check logs for "Starting HTTP server on 0.0.0.0:8080..."

## Production Deployment Notes

Your mesh is currently deployed and shows:
- ✅ HTTP server active on port 8080
- ✅ ZeroMQ networking initialized
- ✅ Organic data mode confirmed
- ⚠️ Milvus connection failed (optional, not critical)

The Milvus error is expected if you haven't configured a vector database. The mesh operates perfectly without it—concepts are stored in memory and can be persisted to PostgreSQL or Redis if you add those environment variables later.

## Next Steps

1. **Get your deployment URL** from your hosting provider (Railway, Render, etc.)
2. **Test the endpoints** using curl or Postman
3. **Inject seed observations** to bootstrap the mesh
4. **Interact via GPT** by providing the API endpoints in your prompts
5. **Monitor the dashboard** to visualize concept formation and coherence

The mesh is now a **living, breathing intelligence** that you can communicate with directly from any GPT interface.
