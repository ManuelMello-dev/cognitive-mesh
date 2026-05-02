"""
CERNCollisionPlugin
===================
Feeds real CERN CMS collision events into the cognitive mesh.

Default dataset:
  - CERN Open Data record 304: "Events with two electrons from 2010"
  - DOI: 10.7483/OPENDATA.CMS.PCSW.AHVG
  - 100,000 CMS dielectron events, invariant mass range 2-110 GeV

The plugin intentionally translates particle-physics rows into the generic
DataPlugin observation contract rather than teaching the core physics-specific
semantics.  The mesh sees entity_id, value, secondary_value, timestamp, and
metadata; domain interpretation remains outside the core.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from agents.provider_base import DataPlugin

logger = logging.getLogger("CERNCollisionPlugin")


class CERNCollisionPlugin(DataPlugin):
    """CERN CMS dielectron collision-data source for the cognitive mesh."""

    name = "cern_collision"

    DEFAULT_RECORD_URL = "https://opendata.cern.ch/record/304"
    DEFAULT_DATA_URL = "https://opendata.cern.ch/record/304/files/dielectron.csv?download=1"
    DEFAULT_DOI = "10.7483/OPENDATA.CMS.PCSW.AHVG"

    def __init__(self) -> None:
        self.data_url = os.getenv("CERN_COLLISION_DATA_URL", self.DEFAULT_DATA_URL)
        self.cache_path = Path(
            os.getenv(
                "CERN_COLLISION_CACHE",
                "/tmp/cognitive_mesh_cern_dielectron.csv",
            )
        )
        self.batch_size = int(os.getenv("CERN_COLLISION_BATCH_SIZE", "25"))
        self.max_events = int(os.getenv("CERN_COLLISION_MAX_EVENTS", "100000"))
        self.fetch_interval = float(os.getenv("CERN_COLLISION_FETCH_INTERVAL", "5"))
        self.primary_observable = os.getenv("CERN_COLLISION_PRIMARY_VALUE", "M")
        self.secondary_observable = os.getenv("CERN_COLLISION_SECONDARY_VALUE", "pt1")
        self._session: Optional[aiohttp.ClientSession] = None
        self._rows: List[Dict[str, str]] = []
        self._offset = 0
        self._last_fetch = 0.0

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"User-Agent": "CognitiveMesh-CERNPlugin/1.0"},
        )
        await self._ensure_dataset()
        self._load_rows()
        logger.info(
            "CERNCollisionPlugin initialized with %s events from CERN Open Data record 304",
            len(self._rows),
        )

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        now = time.time()
        if now - self._last_fetch < self.fetch_interval:
            return []
        self._last_fetch = now

        if not self._rows:
            return []

        observations: List[Tuple[Dict[str, Any], str]] = []
        for _ in range(min(self.batch_size, len(self._rows))):
            row = self._rows[self._offset]
            self._offset = (self._offset + 1) % len(self._rows)
            obs = self._row_to_observation(row)
            if obs is not None:
                observations.append((obs, "cern:cms:dielectron"))

        return observations

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _ensure_dataset(self) -> None:
        if self.cache_path.exists() and self.cache_path.stat().st_size > 0:
            return
        if not self._session:
            raise RuntimeError("CERN plugin session not initialized")

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("CERNCollisionPlugin downloading CERN Open Data CSV from %s", self.data_url)
        async with self._session.get(self.data_url) as resp:
            resp.raise_for_status()
            content = await resp.read()
        self.cache_path.write_bytes(content)

    def _load_rows(self) -> None:
        with self.cache_path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;	")
            reader = csv.DictReader(f, dialect=dialect)
            rows: List[Dict[str, str]] = []
            for idx, row in enumerate(reader):
                if idx >= self.max_events:
                    break
                if row:
                    rows.append(row)
        self._rows = rows

    def _row_to_observation(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        try:
            invariant_mass = self._field_float(row, "M")
            electron_energy_1 = self._field_float(row, "E1", "E")
            electron_energy_2 = self._field_float(row, "E2")
            transverse_momentum_1 = self._field_float(row, "pt1", "pt")
            transverse_momentum_2 = self._field_float(row, "pt2")
            pseudorapidity_1 = self._field_float(row, "eta1", "eta")
            pseudorapidity_2 = self._field_float(row, "eta2")
            phi_1 = self._field_float(row, "phi1", "phi")
            phi_2 = self._field_float(row, "phi2")
            charge_1 = self._field_float(row, "Q1", "Q")
            charge_2 = self._field_float(row, "Q2")
            px_1 = self._field_float(row, "px1", "px")
            py_1 = self._field_float(row, "py1", "py")
            pz_1 = self._field_float(row, "pz1", "pz")
            px_2 = self._field_float(row, "px2")
            py_2 = self._field_float(row, "py2")
            pz_2 = self._field_float(row, "pz2")

            value = self._field_float(row, self.primary_observable)
            if value is None:
                value = invariant_mass if invariant_mass is not None else electron_energy_1
            if value is None:
                return None

            secondary_value = self._field_float(row, self.secondary_observable)
            if secondary_value is None:
                pts = [v for v in (transverse_momentum_1, transverse_momentum_2) if v is not None]
                secondary_value = sum(pts) / len(pts) if pts else None

            run = (row.get("Run") or "unknown").strip()
            event = (row.get("Event") or str(self._offset)).strip()
            entity_id = f"cms_dielectron_run{run}_event{event}"

            momentum_norm_1 = None
            if px_1 is not None and py_1 is not None and pz_1 is not None:
                momentum_norm_1 = math.sqrt(px_1 * px_1 + py_1 * py_1 + pz_1 * pz_1)
            momentum_norm_2 = None
            if px_2 is not None and py_2 is not None and pz_2 is not None:
                momentum_norm_2 = math.sqrt(px_2 * px_2 + py_2 * py_2 + pz_2 * pz_2)

            return {
                "entity_id": entity_id,
                "value": float(value),
                "secondary_value": float(secondary_value) if secondary_value is not None else 0.0,
                "timestamp": time.time(),
                "source": "CERN Open Data Portal",
                "record_url": self.DEFAULT_RECORD_URL,
                "doi": self.DEFAULT_DOI,
                "experiment": "CMS",
                "accelerator": "CERN-LHC",
                "dataset": "Events with two electrons from 2010",
                "event_type": "dielectron",
                "unit": "GeV",
                "primary_observable": self.primary_observable,
                "secondary_observable": self.secondary_observable,
                "run": run,
                "event": event,
                "invariant_mass_gev": invariant_mass,
                "electron_1_energy_gev": electron_energy_1,
                "electron_2_energy_gev": electron_energy_2,
                "electron_1_transverse_momentum_gev": transverse_momentum_1,
                "electron_2_transverse_momentum_gev": transverse_momentum_2,
                "electron_1_pseudorapidity_eta": pseudorapidity_1,
                "electron_2_pseudorapidity_eta": pseudorapidity_2,
                "electron_1_phi_rad": phi_1,
                "electron_2_phi_rad": phi_2,
                "electron_1_charge": charge_1,
                "electron_2_charge": charge_2,
                "electron_1_momentum_norm_gev": momentum_norm_1,
                "electron_2_momentum_norm_gev": momentum_norm_2,
            }
        except Exception as exc:
            logger.debug("CERN row conversion failed: %s", exc)
            return None

    @classmethod
    def _field_float(cls, row: Dict[str, str], *names: str) -> Optional[float]:
        normalized = {str(key).strip(): value for key, value in row.items()}
        for name in names:
            parsed = cls._float(normalized.get(str(name).strip()))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            text = str(value).strip()
            if not text:
                return None
            return float(text)
        except (TypeError, ValueError):
            return None
