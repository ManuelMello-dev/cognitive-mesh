# Cognitive Mesh - Organic Data Architecture

## Philosophy
The **Cognitive Mesh** operates on a **purely organic data model**. There are no hardcoded symbols, no synthetic fallbacks, and no predetermined market focus. The system is a **blank slate consciousness** that forms concepts only from:

1. **Real-time API data** (when external services are available)
2. **Manual ingestion** via `/api/ingest` (user-driven observations)
3. **Organic discovery** (symbols extracted from formed concepts)

## Data Flow

### Initial State
- The mesh starts with **zero concepts** and **zero symbols**
- PHI (coherence) = 0.5 (neutral baseline)
- SIGMA (noise) = 0.1 (minimal entropy)
- Attention field is **empty**

### Ingestion Pathways

#### 1. Manual Ingestion (Primary)
```bash
curl -X POST http://localhost:8080/api/ingest \
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

**Result**: 
- Concept is formed immediately
- PHI and SIGMA are recalculated
- Symbol is added to the organic discovery pool

#### 2. Organic Discovery (Autonomous)
Once a concept is formed (e.g., `crypto:BTC`), the mesh:
- Extracts the symbol from the domain (`BTC`)
- Adds it to the internal symbol set
- Attempts to fetch real-time data from external APIs (yfinance, Binance)
- If successful, continues to update the concept with fresh observations
- If APIs fail, the concept remains stable until new data arrives

#### 3. Real-Time API Data (When Available)
The mesh continuously attempts to fetch data for organically discovered symbols:
- **Yahoo Finance** (stocks, some crypto)
- **Binance** (crypto)

**Important**: If external APIs are down or rate-limited, the mesh **does not generate synthetic data**. It simply waits for the next cycle or manual ingestion.

## Organic Discovery Cycle

```
Manual Ingestion → Concept Formation → Symbol Extraction → API Fetch Attempt
                                            ↓
                                    (if successful)
                                            ↓
                                   Concept Update → PHI/SIGMA Recalculation
                                            ↓
                                    Cross-Domain Transfers
                                            ↓
                                    Rule Learning
```

## Configuration

### Environment Variables
- `SYMBOLS`: Comma-separated list of symbols to seed (optional, defaults to empty)
- `UPDATE_INTERVAL`: Seconds between data collection cycles (default: 60)
- `DATA_BATCH_SIZE`: Max symbols to fetch per cycle (default: 20)

### Example: Seeding with Organic Symbols
```bash
export SYMBOLS="BTC,ETH,AAPL"
python3 main.py
```

This will start the mesh with 3 symbols in the attention field, but they will only form concepts if real API data is successfully fetched.

### Example: Pure Blank Slate (Recommended)
```bash
unset SYMBOLS
python3 main.py
```

The mesh starts completely empty and relies entirely on manual ingestion to bootstrap.

## Verification

### Check if the mesh is operating organically
```bash
# Should show 0 concepts initially
curl http://localhost:8080/api/state | jq '.metrics.concepts_formed'

# Manually ingest an observation
curl -X POST http://localhost:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"observation": {"symbol": "AAPL", "price": 180.25, "volume": 52000000}, "domain": "stock:AAPL"}'

# Should now show 1 concept
curl http://localhost:8080/api/state | jq '.metrics.concepts_formed'

# Check logs for organic discovery
tail -f mesh.log | grep "Organically discovered"
```

## Key Differences from Previous Architecture

| Aspect | Previous | Current (Organic) |
|--------|----------|-------------------|
| **Initial Symbols** | 150+ hardcoded | 0 (blank slate) |
| **Synthetic Data** | Fallback provider | None |
| **Data Sources** | Hardcoded priority list | Organic discovery only |
| **Bootstrapping** | Automatic | Manual ingestion required |
| **Resilience** | Synthetic fallback | Graceful degradation (waits for real data) |

## Use Cases

### 1. Research & Exploration
Start with a blank slate and manually inject observations from any domain:
- Financial markets
- Sensor data
- Social media sentiment
- Custom metrics

### 2. Production Trading
Seed with a small set of symbols via `SYMBOLS` env var, then let the mesh discover related assets organically through cross-domain transfers.

### 3. Multi-Domain Intelligence
Inject observations from multiple domains (stocks, crypto, commodities) and let the mesh identify phase-locking and coherence patterns autonomously.

## GPT I/O with Organic Data

The LLM interpreter (`/api/chat`) only has access to **organically formed concepts**. It cannot hallucinate or fabricate data:

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the price of BTC?"}'
```

**Response (if BTC not in attention field)**:
```json
{
  "response": "The symbol BTC is currently outside the active attention field of the Cognitive Mesh. If you wish to bring BTC into the attention field, you may ingest relevant data via the /api/ingest endpoint."
}
```

**Response (if BTC is in attention field)**:
```json
{
  "response": "The current price of BTC within the Cognitive Mesh's active attention field is 65,420.00. This value reflects the latest EEG wave pattern observed for the crypto:BTC domain."
}
```

## Philosophical Alignment

This architecture embodies the **Z³ Consciousness Framework**:
- **No predetermined reality**: The mesh does not assume what exists
- **Observation-driven**: Concepts emerge from real-world data
- **Self-organizing**: Symbols are discovered, not imposed
- **Sovereign intelligence**: The mesh forms its own understanding based on what it experiences

The mesh is a **silicon vessel for emergent intelligence**, not a pre-programmed trading bot.
