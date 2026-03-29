"""
MicrostructurePlugin
====================
Feeds market microstructure signals into the cognitive mesh.

Sources (all free, no API key required):
  - Binance order book depth (bid/ask imbalance, spread)
  - Binance recent trades (buy/sell pressure ratio)
  - Binance 24hr ticker stats (volume, price change, trade count)
  - CoinGecko exchange volume distribution

These signals capture the mechanics of price formation — who is
buying vs selling, how deep the order book is, and where liquidity sits.
In the EEG analogy, this is the raw neural firing — the moment-to-moment
electrical activity before it becomes a recognizable waveform.

Observation schema (domain = "micro:<asset>:<metric>"):
  entity_id     : metric name
  value         : numeric value
  asset         : underlying asset
  timestamp     : UTC epoch float
  domain_prefix : "micro"

Fetch cadence: every 60 seconds (microstructure changes fast)
"""
import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("MicrostructurePlugin")

_TRACKED_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
_ORDER_BOOK_DEPTH = 20  # levels to fetch


class MicrostructurePlugin(DataPlugin):
    name = "microstructure"
    FETCH_INTERVAL = 60  # 1 minute

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch: float = 0.0
        self._cache: List[Tuple[Dict, str]] = []

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "CognitiveMesh/1.0"},
        )
        logger.info("MicrostructurePlugin initialized")

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        if now - self._last_fetch < self.FETCH_INTERVAL and self._cache:
            return self._cache

        tasks = []
        for symbol in _TRACKED_PAIRS:
            tasks.append(self._fetch_order_book(symbol))
            tasks.append(self._fetch_ticker_stats(symbol))

        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[Tuple[Dict, str]] = []
        for item in fetched:
            if isinstance(item, list):
                results.extend(item)

        if results:
            self._cache = results
            self._last_fetch = now
        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────
    # Sources
    # ──────────────────────────────────────────

    async def _fetch_order_book(self, symbol: str) -> List[Tuple[Dict, str]]:
        """Binance order book depth — bid/ask imbalance is a leading price signal."""
        try:
            url = "https://api.binance.com/api/v3/depth"
            params = {"symbol": symbol, "limit": _ORDER_BOOK_DEPTH}
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids or not asks:
                return []

            # Total bid and ask volume in the top N levels
            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)
            total_volume = bid_volume + ask_volume

            # Bid/ask imbalance: >0.6 = buy pressure, <0.4 = sell pressure
            imbalance = round(bid_volume / total_volume, 4) if total_volume > 0 else 0.5

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_pct = round(spread / best_ask * 100, 6) if best_ask > 0 else 0

            asset = symbol.replace("USDT", "")
            obs = {
                "entity_id": f"{asset.lower()}_order_book",
                "value": imbalance,
                "asset": asset,
                "bid_volume": round(bid_volume, 4),
                "ask_volume": round(ask_volume, 4),
                "spread": round(spread, 6),
                "spread_pct": spread_pct,
                "best_bid": best_bid,
                "best_ask": best_ask,
                # Derived signals
                "buy_pressure": imbalance > 0.6,
                "sell_pressure": imbalance < 0.4,
                "tight_spread": spread_pct < 0.01,
                "wide_spread": spread_pct > 0.1,
                "direction": "buy_heavy" if imbalance > 0.6 else ("sell_heavy" if imbalance < 0.4 else "balanced"),
                "timestamp": time.time(),
            }
            return [(obs, f"micro:{asset.lower()}:order_book")]
        except Exception as e:
            logger.debug(f"Order book fetch error for {symbol}: {e}")
            return []

    async def _fetch_ticker_stats(self, symbol: str) -> List[Tuple[Dict, str]]:
        """Binance 24hr ticker — volume, price change, trade count."""
        try:
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            asset = symbol.replace("USDT", "")
            price_change_pct = float(data.get("priceChangePercent", 0))
            volume = float(data.get("volume", 0))
            quote_volume = float(data.get("quoteVolume", 0))
            trade_count = int(data.get("count", 0))
            high = float(data.get("highPrice", 0))
            low = float(data.get("lowPrice", 0))
            last = float(data.get("lastPrice", 0))

            # Position within 24h range (0 = at low, 1 = at high)
            range_position = round((last - low) / (high - low), 4) if high > low else 0.5

            obs = {
                "entity_id": f"{asset.lower()}_24h_stats",
                "value": price_change_pct,
                "asset": asset,
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": trade_count,
                "high_24h": high,
                "low_24h": low,
                "range_position": range_position,
                # Derived signals
                "strong_move": abs(price_change_pct) > 5.0,
                "high_volume": quote_volume > 1_000_000_000,  # >$1B 24h volume
                "near_high": range_position > 0.85,
                "near_low": range_position < 0.15,
                "direction": "up" if price_change_pct > 0.5 else ("down" if price_change_pct < -0.5 else "stable"),
                "timestamp": time.time(),
            }
            return [(obs, f"micro:{asset.lower()}:24h_stats")]
        except Exception as e:
            logger.debug(f"Ticker stats fetch error for {symbol}: {e}")
            return []
