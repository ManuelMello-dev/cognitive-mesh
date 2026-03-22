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
    PORT = int(os.getenv("PORT", 8080))
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
    # Set DISABLE_MARKET_PLUGIN=1 to run without financial data
    DISABLE_MARKET_PLUGIN = os.getenv("DISABLE_MARKET_PLUGIN", "").lower() in ("1", "true", "yes")

    # Market plugin price scan filters (only relevant when market plugin is active)
    MIN_SCAN_PRICE = float(os.getenv("MIN_SCAN_PRICE", 0.0))
    MAX_SCAN_PRICE = float(os.getenv("MAX_SCAN_PRICE", float("inf")))

    # ── Persistence ───────────────────────────────────────────────────────────
    POSTGRES_URL = os.getenv("POSTGRES_URL")
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    REDIS_URL = os.getenv("REDIS_URL")

