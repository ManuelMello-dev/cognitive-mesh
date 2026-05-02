"""
agents/plugins
==============
DataPlugin implementations for the cognitive mesh.

The mesh is data-source agnostic.  CERN collision data is the default proving
domain because it exercises the same observation contract without financial
market assumptions.  Market-context plugins remain available as legacy optional
plugins and are loaded only when explicitly enabled by environment flags.
"""

from agents.plugins.cern_collision_plugin import CERNCollisionPlugin

__all__ = [
    "CERNCollisionPlugin",
    "SentimentPlugin",
    "MacroPlugin",
    "OnChainPlugin",
    "NewsPlugin",
    "DerivativesPlugin",
    "SocialPlugin",
    "MicrostructurePlugin",
]
