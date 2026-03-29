"""
agents/provider_base.py
=======================
Shared DataPlugin base class used by all plugin implementations.

The DataPlugin interface is intentionally minimal:
  - initialize()  : called once at startup
  - fetch()       : called each collection cycle; returns list of (obs_dict, domain_str)
  - close()       : called at shutdown

The DataPlugin defined here is the canonical source of truth.
The DataPlugin class in main.py is a duplicate kept for backward compatibility
with the orchestrator's isinstance checks — both are structurally identical.
"""
from typing import List, Tuple, Dict, Any


class DataPlugin:
    """
    Base class for all cognitive mesh data source plugins.

    A plugin is responsible for:
      1. Discovering entities to observe (optional, can be static)
      2. Fetching observations each cycle
      3. Returning a list of (observation_dict, domain_string) tuples

    The observation_dict MUST contain at least:
      - 'value'     : float  — the primary observable
      - 'entity_id' : str    — unique identifier for this stream

    Optional fields:
      - 'secondary_value' : float  — e.g. volume, confidence, intensity
      - 'timestamp'       : float  — unix timestamp (defaults to now)
      - any domain-specific fields (stored as metadata)
    """

    name: str = "base"

    async def initialize(self) -> None:
        """Called once at startup before any fetch() calls."""
        pass

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        """
        Fetch a batch of observations.

        Returns:
            list of (observation_dict, domain_string) tuples
        """
        return []

    async def close(self) -> None:
        """Called at shutdown — release any connections or resources."""
        pass
