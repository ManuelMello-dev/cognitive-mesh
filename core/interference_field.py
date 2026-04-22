"""
Interference Field
==================
Implements local interference summaries inspired by:

    |psi_1 + psi_2|^2

The purpose is to turn pairwise resonance and discord between nearby identity
states into explicit constitutional signals rather than hidden side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np



def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))



@dataclass
class InterferenceSummary:
    constructive: float
    destructive: float
    net: float
    partner_count: int
    strongest_partner: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constructive": round(self.constructive, 6),
            "destructive": round(self.destructive, 6),
            "net": round(self.net, 6),
            "partner_count": int(self.partner_count),
            "strongest_partner": self.strongest_partner,
        }



class InterferenceField:
    """Computes neighborhood interference for a local identity state."""

    def __init__(self, neighborhood_limit: int = 6) -> None:
        self.neighborhood_limit = int(neighborhood_limit)

    @staticmethod
    def _phase_alignment(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        cosine = float(np.dot(a, b) / denom)
        return _clamp((cosine + 1.0) / 2.0)

    def summarize(
        self,
        agent_id: str,
        state: np.ndarray,
        becoming_state: np.ndarray,
        neighbors: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        constructive = 0.0
        destructive = 0.0
        partner_count = 0
        strongest_score = -1.0
        strongest_partner = ""
        partner_details: List[Dict[str, Any]] = []

        for neighbor in list(neighbors)[: self.neighborhood_limit]:
            neighbor_id = str(neighbor.get("agent_id", ""))
            if not neighbor_id or neighbor_id == agent_id:
                continue

            neighbor_state = np.asarray(neighbor.get("state"), dtype=np.float64)
            if neighbor_state.shape != state.shape:
                continue

            partner_count += 1
            alignment = self._phase_alignment(state, neighbor_state)
            separation = float(np.linalg.norm(state - neighbor_state))
            becoming_bias = self._phase_alignment(becoming_state, neighbor_state)
            resonance = alignment * (0.60 + 0.40 * becoming_bias)
            interference_energy = resonance * math.exp(-0.75 * separation)
            constructive_component = interference_energy * alignment
            destructive_component = interference_energy * (1.0 - alignment)

            constructive += constructive_component
            destructive += destructive_component
            score = constructive_component - destructive_component

            if score > strongest_score:
                strongest_score = score
                strongest_partner = neighbor_id

            partner_details.append(
                {
                    "agent_id": neighbor_id,
                    "alignment": round(alignment, 6),
                    "separation": round(separation, 6),
                    "becoming_bias": round(becoming_bias, 6),
                    "constructive": round(constructive_component, 6),
                    "destructive": round(destructive_component, 6),
                }
            )

        if partner_count > 0:
            constructive /= partner_count
            destructive /= partner_count

        summary = InterferenceSummary(
            constructive=_clamp(constructive),
            destructive=_clamp(destructive),
            net=max(-1.0, min(1.0, constructive - destructive)),
            partner_count=partner_count,
            strongest_partner=strongest_partner,
        )
        output = summary.to_dict()
        output["partners"] = partner_details[:3]
        return output
