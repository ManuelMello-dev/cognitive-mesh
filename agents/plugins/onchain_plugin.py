"""
OnChainPlugin
=============
Feeds on-chain blockchain metrics into the cognitive mesh.

Sources (all free, no API key required):
  - Blockchain.com public stats API (BTC: hash rate, mempool, tx count, fees)
  - CoinGecko global market stats (total market cap, BTC dominance, DeFi TVL)
  - Etherscan public stats (ETH: gas price, pending tx, staking ratio)
  - Alternative.me BTC on-chain metrics

Observation schema (domain = "onchain:<chain>:<metric>"):
  entity_id     : metric name (e.g. "btc_hash_rate", "eth_gas_price")
  value         : numeric value
  unit          : unit of measurement
  timestamp     : UTC epoch float
  domain_prefix : "onchain"

Fetch cadence: every 5 minutes
"""
import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("OnChainPlugin")


class OnChainPlugin(DataPlugin):
    name = "onchain"
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
        logger.info("OnChainPlugin initialized")

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        if now - self._last_fetch < self.FETCH_INTERVAL and self._cache:
            return self._cache

        results: List[Tuple[Dict, str]] = []
        tasks = [
            self._fetch_btc_stats(),
            self._fetch_global_market(),
            self._fetch_eth_gas(),
        ]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        for item in fetched:
            if isinstance(item, list):
                results.extend(item)

        if results:
            self._cache = results
            self._last_fetch = now
            logger.info(f"OnChainPlugin: fetched {len(results)} on-chain observations")
        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────
    # Sources
    # ──────────────────────────────────────────

    async def _fetch_btc_stats(self) -> List[Tuple[Dict, str]]:
        """Blockchain.com public stats — BTC network health."""
        results = []
        endpoints = {
            "hash-rate": ("btc_hash_rate", "TH/s", "btc"),
            "n-transactions": ("btc_tx_count", "count", "btc"),
            "mempool-size": ("btc_mempool_size", "bytes", "btc"),
            "miners-revenue": ("btc_miner_revenue", "USD", "btc"),
            "transaction-fees": ("btc_tx_fees", "BTC", "btc"),
            "n-unique-addresses": ("btc_active_addresses", "count", "btc"),
            "difficulty": ("btc_difficulty", "index", "btc"),
        }
        for endpoint, (entity_id, unit, chain) in endpoints.items():
            try:
                url = f"https://api.blockchain.info/stats"
                async with self._session.get(url) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    # Map endpoint names to JSON keys
                    key_map = {
                        "hash-rate": "hash_rate",
                        "n-transactions": "n_tx",
                        "mempool-size": "mempool_size",
                        "miners-revenue": "miners_revenue_usd",
                        "transaction-fees": "total_fees_btc",
                        "n-unique-addresses": "n_unique_addresses",
                        "difficulty": "difficulty",
                    }
                    key = key_map.get(endpoint)
                    if key and key in data:
                        value = float(data[key])
                        obs = {
                            "entity_id": entity_id,
                            "value": value,
                            "unit": unit,
                            "chain": chain,
                            "timestamp": time.time(),
                        }
                        # Derived signals
                        if entity_id == "btc_mempool_size":
                            obs["congested"] = value > 50_000_000  # 50MB
                        if entity_id == "btc_hash_rate":
                            obs["hash_rate_th"] = round(value / 1e12, 2)
                        results.append((obs, f"onchain:btc:{entity_id}"))
                    break  # One request gets all stats
            except Exception as e:
                logger.debug(f"BTC stats fetch error: {e}")
                break
        return results

    async def _fetch_global_market(self) -> List[Tuple[Dict, str]]:
        """CoinGecko global market stats — BTC dominance, total market cap, DeFi."""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                gdata = data.get("data", {})
                total_mcap = gdata.get("total_market_cap", {}).get("usd", 0)
                btc_dom = gdata.get("market_cap_percentage", {}).get("btc", 0)
                eth_dom = gdata.get("market_cap_percentage", {}).get("eth", 0)
                defi_mcap = gdata.get("total_value_locked", {}).get("usd", 0) if "total_value_locked" in gdata else 0
                mcap_change_24h = gdata.get("market_cap_change_percentage_24h_usd", 0)
                results = []
                results.append(({
                    "entity_id": "crypto_total_market_cap",
                    "value": total_mcap,
                    "unit": "USD",
                    "pct_change_24h": mcap_change_24h,
                    "direction": "up" if mcap_change_24h > 0.5 else ("down" if mcap_change_24h < -0.5 else "stable"),
                    "timestamp": time.time(),
                }, "onchain:global:total_market_cap"))

                results.append(({
                    "entity_id": "btc_dominance",
                    "value": round(btc_dom, 4),
                    "unit": "%",
                    "eth_dominance": round(eth_dom, 4),
                    "altcoin_season": btc_dom < 40,
                    "btc_season": btc_dom > 60,
                    "timestamp": time.time(),
                }, "onchain:global:btc_dominance"))

                return results
        except Exception as e:
            logger.debug(f"CoinGecko global fetch error: {e}")
            return []

    async def _fetch_eth_gas(self) -> List[Tuple[Dict, str]]:
        """Etherscan public gas oracle — no API key for basic stats."""
        try:
            url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                if data.get("status") != "1":
                    return []
                result = data.get("result", {})
                safe_gas = float(result.get("SafeGasPrice", 0))
                fast_gas = float(result.get("FastGasPrice", 0))
                base_fee = float(result.get("suggestBaseFee", 0))
                obs = {
                    "entity_id": "eth_gas_price",
                    "value": safe_gas,
                    "fast_gas": fast_gas,
                    "base_fee": base_fee,
                    "unit": "Gwei",
                    "congested": fast_gas > 100,
                    "cheap": safe_gas < 10,
                    "timestamp": time.time(),
                }
                return [(obs, "onchain:eth:gas_price")]
        except Exception as e:
            logger.debug(f"ETH gas fetch error: {e}")
            return []
