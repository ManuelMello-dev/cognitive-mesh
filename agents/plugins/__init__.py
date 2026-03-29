"""
agents/plugins
==============
Market-relevant DataPlugin implementations.

Each plugin is a self-contained data source that feeds observations
into the cognitive mesh. All plugins extend DataPlugin from agents.provider.

Available plugins:
  - SentimentPlugin      : Fear & Greed, AAII investor sentiment
  - MacroPlugin          : Fed rates, yield curve, VIX, DXY, oil, gold
  - OnChainPlugin        : BTC/ETH on-chain metrics, global market stats
  - NewsPlugin           : Reuters, Yahoo Finance, CoinDesk, Fed RSS feeds
  - DerivativesPlugin    : Funding rates, open interest, long/short ratio
  - SocialPlugin         : Reddit mentions, CoinGecko trending
  - MicrostructurePlugin : Order book depth, bid/ask imbalance, 24h stats
"""
from agents.plugins.sentiment_plugin import SentimentPlugin
from agents.plugins.macro_plugin import MacroPlugin
from agents.plugins.onchain_plugin import OnChainPlugin
from agents.plugins.news_plugin import NewsPlugin
from agents.plugins.derivatives_plugin import DerivativesPlugin
from agents.plugins.social_plugin import SocialPlugin
from agents.plugins.microstructure_plugin import MicrostructurePlugin

__all__ = [
    "SentimentPlugin",
    "MacroPlugin",
    "OnChainPlugin",
    "NewsPlugin",
    "DerivativesPlugin",
    "SocialPlugin",
    "MicrostructurePlugin",
]
