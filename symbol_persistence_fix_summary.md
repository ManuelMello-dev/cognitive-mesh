# Symbol Persistence Fix Summary

I implemented the runtime persistence fix so the market polling universe no longer disappears after restart.

## What was wrong

The system could restore **learned cognitive state** but still lose the **active market symbol universe** because the `MarketPlugin` kept `_crypto_symbols` and `_stock_symbols` only in RAM. After a restart, if fresh discovery failed or returned nothing, the plugin had no symbols to poll, so providers were never called even though the service stayed online.

## What changed

| Area | Change |
|---|---|
| `main.py` / `MarketPlugin` | Added `export_runtime_state()`, `persist_runtime_state()`, and `restore_runtime_state()` |
| `main.py` / restore path | After `core.load_state()`, the orchestrator now restores market symbol state before normal loops continue |
| `main.py` / checkpoint path | Periodic checkpoints now persist plugin runtime state before saving core state |
| `main.py` / shutdown path | Shutdown now persists plugin runtime state before the final core save |
| `main.py` / fallback restore | If explicit plugin cache is missing, symbol state is rebuilt from Postgres-restored core memory such as prediction symbols, price history, and meta-domains |
| `main.py` / telemetry | Added timestamps and counters for discovery/fetch continuity |

## Storage model after the fix

| Store | Role |
|---|---|
| **RAM** | Live working set for active symbols, offsets, and current polling state |
| **PostgreSQL** | Durable structured registry for the market plugin runtime state, plus existing prediction/history state |
| **Milvus** | Durable vector memory for learned concept relationships and symbol-related conceptual embeddings already maintained by the core |

This means the system now resumes with a usable symbol universe even if the scanner does not immediately rediscover assets after restart.

## Validation performed

| Check | Result |
|---|---|
| `python3.11 -m py_compile main.py` | Passed |
| Isolated restore/persist validation script | Passed |
| Cached-state restore | Restored symbols and offsets correctly |
| Fallback rebuild from core memory | Reconstructed crypto and stock symbols correctly |
| Postgres persistence writeback | Saved the market plugin state correctly |

## Validation command output

> `market_plugin_persistence_ok`

## Files changed

| File | Purpose |
|---|---|
| `main.py` | Implements symbol persistence, restore, and checkpoint integration |
| `test_market_plugin_persistence.py` | Focused validation script for the new restore/save behavior |

## Expected production effect

After deploy, the app should no longer come back in an **alive-but-not-polling** state just because scanner discovery is empty on startup. It should restore the last known active symbols from persistence, register crypto symbols with the provider cascade again, and resume provider calls immediately.
