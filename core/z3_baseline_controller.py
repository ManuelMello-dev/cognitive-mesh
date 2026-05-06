"""
Z3 Baseline Controller
======================
Two-level Z3/Z-prime evidence compressor.

Z-prime modules emit local evidence. The persistent Z3 baseline controller does
not blindly absorb that evidence. It projects each event into a bounded
adjudication frame containing distance, coherence, novelty, drift pressure,
trust gates, and decayed salience memory. The adjudicator can then decide
whether the organism-level baseline observes, holds, rejects, or mutates.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set

from z3_contracts import NoveltyEvent


class Z3BaselineController:
    """Compress local novelty into Z3-readable adjudication evidence.

    This class implements the formal Z3/Z-prime membrane:

        Z3 baseline -> local evidence -> novelty compression -> adjudication

    It intentionally keeps baseline mutation outside the evidence generator.
    High reconstruction distance is therefore not enough to update Z3. Evidence
    must also carry sufficient coherence and salience through the trust gate.
    """

    def __init__(
        self,
        *,
        novelty_threshold: float = 0.35,
        coherence_threshold: float = 0.35,
        lambda_coherence: float = 1.35,
        memory_decay: float = 0.92,
        max_memory: int = 100,
        epsilon: float = 1e-9,
    ) -> None:
        self.novelty_threshold = float(novelty_threshold)
        self.coherence_threshold = float(coherence_threshold)
        self.lambda_coherence = float(lambda_coherence)
        self.memory_decay = float(memory_decay)
        self.epsilon = float(epsilon)
        self._memory: deque[Dict[str, Any]] = deque(maxlen=max_memory)
        self._seen_event_ids: Set[str] = set()

    def evaluate(
        self,
        *,
        events: Iterable[NoveltyEvent],
        phi: float,
        sigma: float,
        drift: float,
        baseline_version: int,
        now: float,
    ) -> Dict[str, Any]:
        """Return the compressed Z3 evidence frame for the current cycle."""
        event_list = list(events or [])
        global_coherence = self._clamp(phi)
        noise = self._clamp(sigma)
        drift_pressure = self._clamp(abs(drift))

        compressed: List[Dict[str, Any]] = []
        raw_weights: List[float] = []
        for rank, event in enumerate(event_list[:25]):
            novelty = self._clamp(event.novelty_score)
            distance = self._distance(event)
            local_coherence = math.exp(-self.lambda_coherence * distance)
            coherence = self._clamp((0.55 * global_coherence) + (0.45 * local_coherence))
            stability = self._clamp((0.50 * coherence) + (0.30 * (1.0 - noise)) + (0.20 * (1.0 - drift_pressure)))
            gate_open = bool(novelty >= self.novelty_threshold and coherence >= self.coherence_threshold)
            trusted_gate_open = bool(gate_open and noise < 0.85 and stability >= 0.35)
            salience = self._clamp(novelty * coherence * stability * (1.0 - (0.55 * noise)))
            raw_weight = max(self.epsilon, global_coherence * coherence * (1.0 + salience))
            raw_weights.append(raw_weight)
            compressed.append(
                {
                    "event_id": event.event_id,
                    "rank": rank,
                    "source": event.source,
                    "signal_type": event.signal_type,
                    "baseline_version": baseline_version,
                    "novelty": round(novelty, 6),
                    "distance": round(distance, 6),
                    "local_coherence": round(local_coherence, 6),
                    "coherence": round(coherence, 6),
                    "stability": round(stability, 6),
                    "noise": round(noise, 6),
                    "drift_pressure": round(drift_pressure, 6),
                    "salience": round(salience, 6),
                    "gate_open": gate_open,
                    "trusted_gate_open": trusted_gate_open,
                }
            )

        total_weight = sum(raw_weights) + self.epsilon
        for item, raw_weight in zip(compressed, raw_weights):
            item["weight"] = round(raw_weight / total_weight, 6)

        self._ingest_memory(compressed, now)
        memory = self.memory_summary(now)
        trusted = [item for item in compressed if item["trusted_gate_open"]]
        visible = [item for item in compressed if item["novelty"] >= self.novelty_threshold]
        latest = compressed[0] if compressed else None

        return {
            "baseline_version": baseline_version,
            "global_coherence": round(global_coherence, 6),
            "noise": round(noise, 6),
            "drift_pressure": round(drift_pressure, 6),
            "lambda_coherence": round(self.lambda_coherence, 6),
            "novelty_threshold": round(self.novelty_threshold, 6),
            "coherence_threshold": round(self.coherence_threshold, 6),
            "compressed_events": compressed,
            "latest": latest,
            "trusted_gate_count": len(trusted),
            "visible_novelty_count": len(visible),
            "trusted_novelty_pressure": round(max([item["salience"] for item in trusted] or [0.0]), 6),
            "novelty_pressure": round(max([item["novelty"] for item in compressed] or [0.0]), 6),
            "memory": memory,
        }

    def memory_summary(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Return bounded decayed salience memory for public Z3 metrics."""
        if now is None:
            now = 0.0
        entries = list(self._memory)
        salience_values = [self._clamp(item.get("salience", 0.0)) for item in entries]
        trusted_entries = [item for item in entries if item.get("trusted_gate_open")]
        return {
            "entries": len(entries),
            "trusted_entries": len(trusted_entries),
            "salience_total": round(min(1.0, sum(salience_values)), 6),
            "salience_peak": round(max(salience_values or [0.0]), 6),
            "last_event_id": entries[-1].get("event_id") if entries else None,
            "updated_at": now,
        }

    def restore(self, memory: Dict[str, Any]) -> None:
        """Restore compact memory metadata from a persisted public snapshot."""
        if not isinstance(memory, dict):
            return
        last_event_id = memory.get("last_event_id")
        if last_event_id:
            self._seen_event_ids.add(str(last_event_id))
        entries = int(self._safe_float(memory.get("entries", 0), 0.0))
        salience_peak = self._clamp(memory.get("salience_peak", 0.0))
        if entries > 0 and salience_peak > 0.0 and not self._memory:
            self._memory.append(
                {
                    "event_id": str(last_event_id or "restored-z3-memory"),
                    "salience": salience_peak,
                    "trusted_gate_open": bool(memory.get("trusted_entries", 0)),
                    "observed_at": self._safe_float(memory.get("updated_at", 0.0), 0.0),
                    "restored": True,
                }
            )

    def _ingest_memory(self, compressed: List[Dict[str, Any]], now: float) -> None:
        if self._memory:
            decayed = []
            for item in self._memory:
                copied = dict(item)
                copied["salience"] = round(self._clamp(copied.get("salience", 0.0)) * self.memory_decay, 6)
                if copied["salience"] >= 0.01:
                    decayed.append(copied)
            self._memory.clear()
            self._memory.extend(decayed)

        for item in compressed:
            event_id = str(item.get("event_id"))
            if event_id in self._seen_event_ids:
                continue
            if item.get("novelty", 0.0) < self.novelty_threshold:
                continue
            self._memory.append(
                {
                    "event_id": event_id,
                    "source": item.get("source"),
                    "signal_type": item.get("signal_type"),
                    "novelty": item.get("novelty", 0.0),
                    "coherence": item.get("coherence", 0.0),
                    "salience": item.get("salience", 0.0),
                    "trusted_gate_open": item.get("trusted_gate_open", False),
                    "observed_at": now,
                }
            )
            self._seen_event_ids.add(event_id)

    def _distance(self, event: NoveltyEvent) -> float:
        evidence = event.evidence if isinstance(event.evidence, dict) else {}
        for key in (
            "distance",
            "nearest_memory_distance",
            "memory_loss",
            "reconstruction_loss",
            "prediction_loss",
            "loss_delta",
        ):
            if key in evidence:
                return self._clamp(self._safe_float(evidence.get(key), event.novelty_score))
        return self._clamp(event.novelty_score)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, min(1.0, value))
