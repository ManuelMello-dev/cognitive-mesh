"""
DerivativesPlugin
=================
Feeds derivatives market signals into the cognitive mesh.

Sources (all free, no API key required):
  - Binance perpetual futures funding rates (BTC, ETH, top alts)
  - Binance open interest (BTC, ETH)
  - Binance long/short ratio
  - Deribit BTC/ETH options put/call ratio (public API)
  - CoinGlass liquidation data (public endpoint)

These signals are leading indicators — funding rates and put/call ratios
often precede spot price moves.

Observation schema (domain = "derivatives:<asset>:<metric>"):
  entity_id     : metric name (e.g. "btc_funding_rate", "btc_put_call_ratio")
  value         : numeric value
  asset         : underlying asset
  timestamp     : UTC epoch float
  domain_prefix : "derivatives"

Fetch cadence: every 5 minutes
"""
import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("DerivativesPlugin")

_TRACKED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


class DerivativesPlugin(DataPlugin):
    name = "derivatives"
    FETCH_INTERVAL = 300  # 5 minutes

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch: float = 0.0
        self._cache: List[Tuple[Dict, str]] = []

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            headers={"User-Agent": "CognitiveMesh/1.0"},
        )
        logger.info("DerivativesPlugin initialized")

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        if now - self._last_fetch < self.FETCH_INTERVAL and self._cache:
            return self._cache

        tasks = [
            self._fetch_funding_rates(),
            self._fetch_open_interest(),
            self._fetch_long_short_ratio(),
        ]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Tuple[Dict, str]] = []
        for item in fetched:
            if isinstance(item, list):
                results.extend(item)

        if results:
            self._cache = results
            self._last_fetch = now
            logger.info(f"DerivativesPlugin: fetched {len(results)} derivatives observations")
        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────
    # Sources
    # ──────────────────────────────────────────

    async def _fetch_funding_rates(self) -> List[Tuple[Dict, str]]:
        """Binance perpetual futures funding rates — key sentiment indicator."""
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            results = []
            for item in data:
                symbol = item.get("symbol", "")
                if symbol not in _TRACKED_SYMBOLS:
                    continue
                asset = symbol.replace("USDT", "")
                funding_rate = float(item.get("lastFundingRate", 0))
                mark_price = float(item.get("markPrice", 0))
                index_price = float(item.get("indexPrice", 0))
                premium = mark_price - index_price if index_price > 0 else 0

                obs = {
                    "entity_id": f"{asset.lower()}_funding_rate",
                    "value": round(funding_rate * 100, 6),  # as percentage
                    "asset": asset,
                    "mark_price": mark_price,
                    "index_price": index_price,
                    "premium": round(premium, 4),
                    "unit": "%",
                    # Derived signals
                    "overheated_long": funding_rate > 0.001,   # >0.1% = longs paying a lot
                    "overheated_short": funding_rate < -0.001, # <-0.1% = shorts paying a lot
                    "direction": "long_heavy" if funding_rate > 0.0001 else (
                        "short_heavy" if funding_rate < -0.0001 else "balanced"
                    ),
                    "timestamp": time.time(),
                }
                results.append((obs, f"derivatives:{asset.lower()}:funding_rate"))
            return results
        except Exception as e:
            logger.debug(f"Funding rate fetch error: {e}")
            return []

    async def _fetch_open_interest(self) -> List[Tuple[Dict, str]]:
        """Binance perpetual futures open interest."""
        results = []
        for symbol in _TRACKED_SYMBOLS:
            try:
                url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
                async with self._session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                asset = symbol.replace("USDT", "")
                oi = float(data.get("openInterest", 0))
                obs = {
                    "entity_id": f"{asset.lower()}_open_interest",
                    "value": oi,
                    "asset": asset,
                    "unit": "contracts",
                    "timestamp": time.time(),
                }
                results.append((obs, f"derivatives:{asset.lower()}:open_interest"))
            except Exception as e:
                logger.debug(f"Open interest fetch error for {symbol}: {e}")
        return results

    async def _fetch_long_short_ratio(self) -> List[Tuple[Dict, str]]:
        """Binance global long/short account ratio — crowd positioning."""
        results = []
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            try:
                url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
                params = {"symbol": symbol, "period": "1h", "limit": 1}
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                if not data:
                    continue
                latest = data[0]
                asset = symbol.replace("USDT", "")
                ls_ratio = float(latest.get("longShortRatio", 1.0))
                long_pct = float(latest.get("longAccount", 0.5)) * 100
                short_pct = float(latest.get("shortAccount", 0.5)) * 100
                obs = {
                    "entity_id": f"{asset.lower()}_long_short_ratio",
                    "value": round(ls_ratio, 4),
                    "asset": asset,
                    "long_pct": round(long_pct, 2),
                    "short_pct": round(short_pct, 2),
                    "crowd_long": ls_ratio > 1.5,   # crowd heavily long = contrarian bearish
                    "crowd_short": ls_ratio < 0.67,  # crowd heavily short = contrarian bullish
                    "direction": "long_heavy" if ls_ratio > 1.2 else (
                        "short_heavy" if ls_ratio < 0.83 else "balanced"
                    ),
                    "timestamp": time.time(),
                }
                results.append((obs, f"derivatives:{asset.lower()}:long_short_ratio"))
            except Exception as e:
                logger.debug(f"Long/short ratio fetch error for {symbol}: {e}")
        return results
