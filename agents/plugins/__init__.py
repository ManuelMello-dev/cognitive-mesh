"""
agents/plugins
==============
DataPlugin implementations for the cognitive mesh.

The mesh is data-source agnostic. Language is now the default proving and
interaction stream, while market-context plugins remain available as legacy
optional plugins loaded only when explicitly enabled by environment flags.
"""

from agents.plugins.language_stream_plugin import LanguageStreamPlugin

__all__ = [
    "LanguageStreamPlugin",
    "SentimentPlugin",
    "MacroPlugin",
    "OnChainPlugin",
    "NewsPlugin",
    "DerivativesPlugin",
    "SocialPlugin",
    "MicrostructurePlugin",
]
