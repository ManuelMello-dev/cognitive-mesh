"""
Cognitive Mesh — Configuration
================================
All settings are read from environment variables with sensible defaults.
No domain-specific values (symbols, asset classes, etc.) belong here.
Domain-specific configuration lives in the relevant DataPlugin.
"""

import os
from typing import List, Set


class Config:
    # ── System Identity ───────────────────────────────────────────────────────
    NODE_ID = os.getenv("NODE_ID", "global_mind_01")

    # ── Networking ────────────────────────────────────────────────────────────
    PORT = int(os.getenv("PORT", 8081))
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 30))  # seconds between cycles

    # ── Cognitive Core ────────────────────────────────────────────────────────
    CONCEPT_SIMILARITY_THRESHOLD = float(os.getenv("CONCEPT_SIMILARITY_THRESHOLD", 0.75))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.01))
    CONCEPT_HALF_LIFE_HOURS = float(os.getenv("CONCEPT_HALF_LIFE_HOURS", 72.0))
    DECAY_CHECK_INTERVAL = int(os.getenv("DECAY_CHECK_INTERVAL", 3600))
    CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 100))
    GOAL_GENERATION_INTERVAL = int(os.getenv("GOAL_GENERATION_INTERVAL", 50))
    MAX_RULES_PER_OBSERVATION = int(os.getenv("MAX_RULES_PER_OBSERVATION", 5))

    # ── Self-Evolution ────────────────────────────────────────────────────────
    # How often (in cognitive cycles) to attempt a self-evolution pass
    EVOLUTION_INTERVAL = int(os.getenv("EVOLUTION_INTERVAL", 25))

    # ── Data Collection ───────────────────────────────────────────────────────
    # Maximum observations to ingest per cycle across all plugins
    DATA_BATCH_SIZE = int(os.getenv("DATA_BATCH_SIZE", 150))

    # ── Plugin Flags ─────────────────────────────────────────────────────────
    # CERN collision data is the default proving data source.
    DISABLE_CERN_PLUGIN = os.getenv("DISABLE_CERN_PLUGIN", "").lower() in ("1", "true", "yes")

    # Market systems are legacy/optional. They load only when explicitly enabled.
    ENABLE_MARKET_PLUGIN = os.getenv("ENABLE_MARKET_PLUGIN", "").lower() in ("1", "true", "yes")
    ENABLE_MARKET_CONTEXT_PLUGINS = os.getenv("ENABLE_MARKET_CONTEXT_PLUGINS", "").lower() in ("1", "true", "yes")
    DISABLE_MARKET_PLUGIN = not ENABLE_MARKET_PLUGIN

    # Generic value filters. Legacy MIN/MAX_SCAN_PRICE aliases are still accepted
    # for backward compatibility when the old MarketPlugin is explicitly enabled.
    MIN_SCAN_VALUE = float(os.getenv("MIN_SCAN_VALUE", os.getenv("MIN_SCAN_PRICE", 0.0)))
    MAX_SCAN_VALUE = float(os.getenv("MAX_SCAN_VALUE", os.getenv("MAX_SCAN_PRICE", float("inf"))))
    MIN_SCAN_PRICE = MIN_SCAN_VALUE
    MAX_SCAN_PRICE = MAX_SCAN_VALUE

    # ── Persistence ───────────────────────────────────────────────────────────
    # Accept both the repository's historical variable names and the defaults
    # commonly injected by managed providers such as Railway.
    POSTGRES_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
    REDIS_URL = os.getenv("REDIS_URL")
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

