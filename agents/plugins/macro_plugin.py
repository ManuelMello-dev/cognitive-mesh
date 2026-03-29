"""
MacroPlugin
===========
Feeds macroeconomic signals into the cognitive mesh.

Sources (all free, no API key required):
  - FRED (Federal Reserve Economic Data) public API — no key needed for basic series
  - US Treasury yield curve (2Y, 10Y, 30Y) via FRED
  - DXY (US Dollar Index) via Yahoo Finance
  - WTI Crude Oil price via Yahoo Finance
  - Gold spot price via Yahoo Finance
  - VIX (volatility index) via Yahoo Finance

Observation schema (domain = "macro:<indicator>"):
  entity_id     : indicator name (e.g. "fed_funds_rate", "yield_curve_spread")
  value         : numeric value
  unit          : unit of measurement (%, bps, index, USD)
  timestamp     : UTC epoch float
  domain_prefix : "macro"

Fetch cadence: every 30 minutes (macro data changes slowly)
"""
import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("MacroPlugin")

# Yahoo Finance symbols for macro proxies
_YAHOO_MACRO_SYMBOLS = {
    "DXY": ("dxy", "USD Index", "index"),
    "CL=F": ("wti_crude", "WTI Crude Oil", "USD/bbl"),
    "GC=F": ("gold_spot", "Gold Spot", "USD/oz"),
    "^VIX": ("vix", "CBOE Volatility Index", "index"),
    "^TNX": ("treasury_10y_yield", "10Y Treasury Yield", "%"),
    "^TYX": ("treasury_30y_yield", "30Y Treasury Yield", "%"),
    "^FVX": ("treasury_5y_yield", "5Y Treasury Yield", "%"),
    "^IRX": ("treasury_3m_yield", "3M Treasury Yield", "%"),
}

# FRED series IDs (public, no API key needed for recent data via their JSON API)
_FRED_SERIES = {
    "DFF": ("fed_funds_rate", "Fed Funds Rate", "%"),
    "UNRATE": ("unemployment_rate", "Unemployment Rate", "%"),
    "CPIAUCSL": ("cpi_all_urban", "CPI All Urban", "index"),
    "T10Y2Y": ("yield_curve_10y2y", "10Y-2Y Yield Spread", "%"),
    "T10Y3M": ("yield_curve_10y3m", "10Y-3M Yield Spread", "%"),
    "BAMLH0A0HYM2": ("high_yield_spread", "High Yield Credit Spread", "%"),
    "VIXCLS": ("vix_close", "VIX Close", "index"),
}


class MacroPlugin(DataPlugin):
    name = "macro"
    FETCH_INTERVAL = 1800  # 30 minutes

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch: float = 0.0
        self._cache: List[Tuple[Dict, str]] = []

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            headers={"User-Agent": "CognitiveMesh/1.0"},
        )
        logger.info("MacroPlugin initialized")

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        if now - self._last_fetch < self.FETCH_INTERVAL and self._cache:
            return self._cache

        results: List[Tuple[Dict, str]] = []
        tasks = [
            self._fetch_yahoo_macro(),
            self._fetch_fred_series(),
        ]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        for item in fetched:
            if isinstance(item, list):
                results.extend(item)

        if results:
            self._cache = results
            self._last_fetch = now
            logger.info(f"MacroPlugin: fetched {len(results)} macro observations")
        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────
    # Sources
    # ──────────────────────────────────────────

    async def _fetch_yahoo_macro(self) -> List[Tuple[Dict, str]]:
        """Fetch macro proxies from Yahoo Finance chart API."""
        results = []
        for yahoo_symbol, (entity_id, label, unit) in _YAHOO_MACRO_SYMBOLS.items():
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
                params = {"interval": "1d", "range": "5d"}
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    result = data.get("chart", {}).get("result", [])
                    if not result:
                        continue
                    closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                    closes = [c for c in closes if c is not None]
                    if not closes:
                        continue
                    value = round(closes[-1], 4)
                    prev = closes[-2] if len(closes) >= 2 else value
                    pct_change = round((value - prev) / prev * 100, 4) if prev != 0 else 0.0

                    obs = {
                        "entity_id": entity_id,
                        "value": value,
                        "label": label,
                        "unit": unit,
                        "pct_change": pct_change,
                        "direction": "up" if pct_change > 0.05 else ("down" if pct_change < -0.05 else "stable"),
                        "timestamp": time.time(),
                    }

                    # Derived signals for specific indicators
                    if entity_id == "vix":
                        obs["high_fear"] = value > 30
                        obs["low_fear"] = value < 15
                        obs["vix_regime"] = "fear" if value > 30 else ("complacency" if value < 15 else "normal")

                    if entity_id == "yield_curve_10y2y":
                        obs["inverted"] = value < 0
                        obs["inversion_depth_bps"] = round(value * 100, 1)

                    results.append((obs, f"macro:{entity_id}"))
            except Exception as e:
                logger.debug(f"Yahoo macro fetch error for {yahoo_symbol}: {e}")
        return results

    async def _fetch_fred_series(self) -> List[Tuple[Dict, str]]:
        """
        Fetch key FRED series via their public JSON API.
        No API key needed — uses the public observation endpoint.
        """
        results = []
        for series_id, (entity_id, label, unit) in _FRED_SERIES.items():
            try:
                # FRED public API — returns last 10 observations, no key needed
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                async with self._session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
                    # Skip header, get last valid row
                    data_lines = [l for l in lines[1:] if "." in l.split(",")[-1]]
                    if not data_lines:
                        continue
                    last_line = data_lines[-1]
                    parts = last_line.split(",")
                    if len(parts) < 2:
                        continue
                    value = float(parts[-1])
                    prev_value = float(data_lines[-2].split(",")[-1]) if len(data_lines) >= 2 else value
                    pct_change = round((value - prev_value) / abs(prev_value) * 100, 4) if prev_value != 0 else 0.0

                    obs = {
                        "entity_id": entity_id,
                        "value": value,
                        "label": label,
                        "unit": unit,
                        "pct_change": pct_change,
                        "direction": "up" if pct_change > 0.01 else ("down" if pct_change < -0.01 else "stable"),
                        "timestamp": time.time(),
                    }

                    # Derived signals
                    if entity_id == "yield_curve_10y2y":
                        obs["inverted"] = value < 0
                    if entity_id == "fed_funds_rate":
                        obs["restrictive"] = value > 4.0
                        obs["accommodative"] = value < 1.0
                    if entity_id == "unemployment_rate":
                        obs["elevated"] = value > 5.0
                        obs["low"] = value < 4.0

                    results.append((obs, f"macro:{entity_id}"))
            except Exception as e:
                logger.debug(f"FRED fetch error for {series_id}: {e}")
        return results
