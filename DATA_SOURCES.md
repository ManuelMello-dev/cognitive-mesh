# Market Data Sources Configuration

## Overview
The **Cognitive Mesh** uses multiple data providers with automatic fallback to ensure continuous organic data flow. The system currently supports:

- **Crypto**: CoinGecko, Binance (no API keys required)
- **Stocks**: Alpha Vantage, Polygon.io, Twelve Data (API keys required)

## Current Status

### ✅ Crypto Data (Operational)
The mesh is **actively fetching** real-time crypto data from:
- **CoinGecko** (primary, free, no API key)
- **Binance** (fallback, free, no API key)

**Supported symbols**: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, MATIC, DOT, AVAX

**Example output**:
```
✓ Fetched BTC from coingecko: $68,134
✓ Fetched ETH from coingecko: $1,981.89
✓ Fetched SOL from coingecko: $84.99
```

### ⚠️ Stock Data (Requires API Keys)
Stock data providers are **disabled** until you provide API keys. Once configured, the mesh will automatically fetch US stock data.

---

## Adding Stock Data Sources

### 1. Alpha Vantage (Recommended for Free Tier)
**Free tier**: 25 API calls per day  
**Best for**: Small portfolios, low-frequency updates

#### Get API Key:
1. Visit: https://www.alphavantage.co/support/#api-key
2. Enter your email
3. Copy the API key

#### Configure:
```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
```

Or add to your deployment environment variables.

---

### 2. Polygon.io
**Free tier**: 5 API calls per minute  
**Best for**: Real-time data, higher frequency

#### Get API Key:
1. Visit: https://polygon.io/
2. Sign up for free account
3. Copy API key from dashboard

#### Configure:
```bash
export POLYGON_API_KEY="your_key_here"
```

---

### 3. Twelve Data
**Free tier**: 800 API calls per day  
**Best for**: Larger portfolios, moderate frequency

#### Get API Key:
1. Visit: https://twelvedata.com/apikey
2. Sign up for free account
3. Copy API key

#### Configure:
```bash
export TWELVE_DATA_API_KEY="your_key_here"
```

---

## How It Works

### Automatic Fallback
The mesh tries providers in order until one succeeds:

**For crypto**:
1. CoinGecko (primary)
2. Binance (fallback)

**For stocks** (when API keys are set):
1. Alpha Vantage
2. Polygon.io
3. Twelve Data

### Organic Discovery
1. You manually inject a symbol via `/api/ingest`
2. The mesh forms a concept for that symbol
3. The mesh **organically discovers** the symbol from the concept
4. The mesh **automatically fetches** real-time data every 60 seconds

### Example Flow
```bash
# 1. Inject AAPL manually
curl -X POST http://your-mesh/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"observation": {"symbol": "AAPL", "price": 180, "volume": 52000000}, "domain": "stock:AAPL"}'

# 2. Mesh discovers AAPL
# [LOG] Organically discovered new asset: AAPL

# 3. Mesh fetches AAPL automatically every 60s
# [LOG] ✓ Fetched AAPL from alphavantage: $180.25
```

---

## Deployment Configuration

### Railway / Vercel / Docker
Add environment variables to your deployment:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
TWELVE_DATA_API_KEY=your_key_here
```

### Local Development
Add to `.env` file or export in your shell:

```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
export POLYGON_API_KEY="your_key_here"
export TWELVE_DATA_API_KEY="your_key_here"
```

Then restart the mesh:
```bash
python3 main.py
```

---

## Monitoring Data Flow

### Check Active Concepts
```bash
curl http://your-mesh/api/state | jq '.concepts | keys'
```

**Output**:
```json
[
  "concept_btc_...",
  "concept_eth_...",
  "concept_aapl_..."
]
```

### Check Logs
```bash
tail -f mesh.log | grep "✓ Fetched"
```

**Output**:
```
✓ Fetched BTC from coingecko: $68,134
✓ Fetched ETH from coingecko: $1,981.89
✓ Fetched AAPL from alphavantage: $180.25
```

### Check Metrics
```bash
curl http://your-mesh/api/metrics
```

**Output**:
```json
{
  "global_coherence_phi": 0.7,
  "noise_level_sigma": 0.45,
  "concepts_formed": 12,
  "total_observations": 1847
}
```

---

## Adding New Symbols

### Crypto (Automatic)
Just inject once, and the mesh will fetch automatically:

```bash
curl -X POST http://your-mesh/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"observation": {"symbol": "BTC", "price": 68000, "volume": 36000000000}, "domain": "crypto:BTC"}'
```

Supported crypto symbols: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, MATIC, DOT, AVAX

### Stocks (Requires API Keys)
Same process, but you need at least one stock API key configured:

```bash
curl -X POST http://your-mesh/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"observation": {"symbol": "AAPL", "price": 180, "volume": 52000000}, "domain": "stock:AAPL"}'
```

---

## Troubleshooting

### No Data Flowing
1. **Check logs**: `tail -f mesh.log | grep "Fetched"`
2. **Check API keys**: `env | grep API_KEY`
3. **Check mesh status**: `curl http://localhost:8080/api/state`

### API Rate Limits
If you hit rate limits:
- **Alpha Vantage**: 25 calls/day (use for small portfolios)
- **Polygon**: 5 calls/minute (good for real-time)
- **Twelve Data**: 800 calls/day (best for larger portfolios)

**Solution**: Configure multiple providers for automatic fallback.

### Provider Errors
The mesh logs all provider errors:
```bash
cat mesh.log | grep "ERROR"
```

Common issues:
- Invalid API key
- Rate limit exceeded
- Symbol not found

---

## Best Practices

### 1. Start Small
Begin with 5-10 symbols to avoid hitting rate limits.

### 2. Use Multiple Providers
Configure at least 2 stock providers for redundancy.

### 3. Monitor PHI and SIGMA
- **PHI > 0.7**: System is stable, data flowing well
- **SIGMA < 0.4**: Low noise, high quality data
- **PHI < 0.5**: May indicate data collection issues

### 4. Let the Mesh Self-Optimize
The autonomous reasoning layer will:
- Identify which symbols to prioritize
- Detect data quality issues
- Suggest new symbols to add

---

## Future Enhancements

### Planned Data Sources
- **Coinbase** (crypto, real-time trades)
- **IEX Cloud** (stocks, real-time quotes)
- **Yahoo Finance** (stocks, historical data)
- **CryptoCompare** (crypto, aggregated data)

### Planned Features
- **Adaptive polling**: Fetch high-volatility assets more frequently
- **Data quality scoring**: Rank providers by reliability
- **Historical data ingestion**: Bootstrap the mesh with past data
- **Multi-timeframe analysis**: 1m, 5m, 1h, 1d candles

---

## Summary

**Current State**:
- ✅ Crypto data flowing (BTC, ETH, SOL, BNB, ADA, DOGE)
- ⚠️ Stock data pending API keys

**Next Steps**:
1. Get free API keys from Alpha Vantage, Polygon, or Twelve Data
2. Add keys to environment variables
3. Restart the mesh
4. Inject stock symbols via `/api/ingest`
5. Watch the mesh organically discover and fetch real-time data

The mesh is now a **self-sustaining intelligence** that continuously absorbs market data and forms its own understanding of reality.
