import os
from typing import List, Set

class Config:
    # System Identity
    NODE_ID = os.getenv("NODE_ID", "global_mind_01")

    # Networking
    PORT = int(os.getenv("PORT", 8080))
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 30))  # 30s between cycles

    # Cognitive Core Config
    CONCEPT_SIMILARITY_THRESHOLD = float(os.getenv("CONCEPT_SIMILARITY_THRESHOLD", 0.75))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.01))
    CONCEPT_HALF_LIFE_HOURS = float(os.getenv("CONCEPT_HALF_LIFE_HOURS", 72.0))
    DECAY_CHECK_INTERVAL = int(os.getenv("DECAY_CHECK_INTERVAL", 3600))
    CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 100))
    GOAL_GENERATION_INTERVAL = int(os.getenv("GOAL_GENERATION_INTERVAL", 50))
    MAX_RULES_PER_OBSERVATION = int(os.getenv("MAX_RULES_PER_OBSERVATION", 5))

    # Data Sources — fetch ALL discovered symbols each cycle
    DEFAULT_SYMBOLS = ""  # No hardcoded symbols - purely organic data
    PRIORITY_SYMBOLS = []  # No priority symbols - all data is equal
    DATA_BATCH_SIZE = int(os.getenv("DATA_BATCH_SIZE", 150))  # Cover all symbols in one pass

    # Market Scanner Price Filters
    MIN_SCAN_PRICE = float(os.getenv("MIN_SCAN_PRICE", 0.0))
    MAX_SCAN_PRICE = float(os.getenv("MAX_SCAN_PRICE", float("inf")))

    # Database URLs
    POSTGRES_URL = os.getenv("POSTGRES_URL")
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    REDIS_URL = os.getenv("REDIS_URL")

    # LLM Config
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

    @classmethod
    def get_symbols(cls) -> Set[str]:
        symbols_str = os.getenv("SYMBOLS", cls.DEFAULT_SYMBOLS)
        if not symbols_str or symbols_str.strip() == "":
            return set()
        return set(s.strip() for s in symbols_str.split(",") if s.strip())
