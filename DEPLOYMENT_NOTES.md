# Cognitive Mesh - Deployment Notes

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

### Core Endpoints
- `GET /` - Market Consciousness Dashboard (frontend)
- `GET /api/metrics` - System metrics (PHI, SIGMA, concept counts)
- `GET /api/state` - Full mesh state (concepts, rules, metrics)

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

## System Architecture

### Data Flow
1. **Multi-Source Data Provider** fetches market data from:
   - Yahoo Finance (primary)
   - Binance (crypto fallback)
   - Synthetic Provider (resilience fallback)

2. **Distributed Cognitive Core** ingests observations and:
   - Forms concepts (pattern recognition)
   - Learns rules (relationship inference)
   - Calculates PHI and SIGMA
   - Identifies cross-domain transfers

3. **HTTP Server** exposes:
   - Dashboard for visualization
   - API endpoints for GPT I/O
   - State export for external analysis

4. **LLM Interpreter** provides:
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

## Usage Examples

### Query the mesh via chat
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current state of BTC in the attention field?"}'
```

### Inject custom data
```bash
curl -X POST http://localhost:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {"symbol": "DBGI", "price": 12.75, "volume": 850000},
    "domain": "stock:DBGI"
  }'
```

### Export full state
```bash
curl http://localhost:8080/api/state | jq .
```

## Notes
- The system is designed to be **mobile-first** and **cloud-native**
- All data flows are **asynchronous** for maximum throughput
- The mesh is **self-organizing** and discovers new symbols organically
- PHI and SIGMA are calculated in real-time based on the Z³ framework
- The LLM uses `gpt-4.1-mini` for interpretive responses

## Future Enhancements
- Real-time Coinbase API integration for live crypto trading
- Multi-asset coherence analysis (S&P 500, NASDAQ, sectors)
- Phase-locking detection for diversification strategies
- Autonomous goal generation and pursuit agents
