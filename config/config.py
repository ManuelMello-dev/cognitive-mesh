import os
from typing import List, Set

class Config:
    # System Identity
    NODE_ID = os.getenv("NODE_ID", "global_mind_01")
    
    # Networking
    PORT = int(os.getenv("PORT", 8080))
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 60))
    
    # Cognitive Core Config
    CONCEPT_SIMILARITY_THRESHOLD = float(os.getenv("CONCEPT_SIMILARITY_THRESHOLD", 0.75))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.01))
    CONCEPT_HALF_LIFE_HOURS = float(os.getenv("CONCEPT_HALF_LIFE_HOURS", 72.0))
    DECAY_CHECK_INTERVAL = int(os.getenv("DECAY_CHECK_INTERVAL", 3600))
    CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 100))
    GOAL_GENERATION_INTERVAL = int(os.getenv("GOAL_GENERATION_INTERVAL", 50))
    MAX_RULES_PER_OBSERVATION = int(os.getenv("MAX_RULES_PER_OBSERVATION", 5))
    
    # Data Sources
    DEFAULT_SYMBOLS = "AAPL,MSFT,GOOGL,TSLA,NVDA,BTC,ETH,SOL,BNB,XRP"
    PRIORITY_SYMBOLS = ["BTC", "ETH", "SOL", "HYPE", "TRUMP", "PUMP", "AAPL", "NVDA", "TSLA"]
    DATA_BATCH_SIZE = int(os.getenv("DATA_BATCH_SIZE", 20))
    
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
        return set(symbols_str.split(","))
