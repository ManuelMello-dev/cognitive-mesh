"""
Z3 public membrane and adjudication tests.
Runs offline without starting the HTTP server.
"""
import os
import sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

from core.output_layer import OutputLayer
from core.z3_interface import Z3Interface


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


print("=" * 60)
print("TESTING Z3 PUBLIC MEMBRANE + ADJUDICATION")
print("=" * 60)

interface = Z3Interface(novelty_threshold=0.35, severe_threshold=0.72)
coordinator_state = {
    "iteration": 7,
    "phi": 0.82,
    "sigma": 0.12,
    "drift_vector": 0.02,
    "metrics": {
        "global_coherence_phi": 0.82,
        "noise_level_sigma": 0.12,
        "total_observations": 42,
        "total_concepts": 5,
        "total_rules": 3,
        "prediction_accuracy": 0.67,
        "memory_reconstruction_confidence": 0.8,
        "active_goals": 2,
    },
    "z_cubed_state": {
        "regime": "coherent",
        "coherence": 0.82,
        "stability": 0.88,
    },
}
world_model = {
    "iteration": 3,
    "average_recent_loss": 0.28,
    "latest": {
        "iteration": 3,
        "novelty": 0.81,
        "memory_loss": 0.81,
        "nearest_memory_distance": 0.84,
        "prediction_loss": 0.22,
        "reconstruction_loss": 0.18,
        "latent_norm": 0.44,
        "timestamp": 1710000000.0,
    },
}
recursive_state = {
    "iteration": 7,
    "coherence_loss": 0.18,
    "loss_delta": 0.02,
    "world_model_loss": 0.28,
    "world_model_memory_loss": 0.81,
}

z3_state = interface.project(
    coordinator_state=coordinator_state,
    world_model=world_model,
    resonant_memory={"metrics": {"rings": 2}},
    learning={"drift_events": []},
    predictions=[{"symbol": "BTC", "confidence": 0.7}],
    recursive_state=recursive_state,
).to_dict()

check("Z3 state identity", z3_state.get("identity") == "Z3")
check("Z3 baseline exists", isinstance(z3_state.get("baseline"), dict))
check("Novelty event created", len(z3_state.get("novelty_events", [])) >= 1)
check("Novelty event is compressed", "latent_state" not in z3_state["novelty_events"][0].get("evidence", {}))
check("Baseline updated by trusted novelty", z3_state["baseline"].get("version", 1) >= 2)
check("Decision has evidence score", isinstance((z3_state.get("last_decision") or {}).get("evidence_score"), dict))
evidence_score = (z3_state.get("last_decision") or {}).get("evidence_score") or {}
baseline_metrics = z3_state.get("baseline", {}).get("metrics", {})
organism_state = z3_state.get("organism_state", {})
check("Trusted novelty gate opens", evidence_score.get("gate_open") is True, evidence_score)
check("Local coherence is formalized", 0.0 < evidence_score.get("local_coherence", 0.0) <= 1.0, evidence_score)
check("Adjudication memory is exposed", baseline_metrics.get("adjudication_memory", {}).get("entries", 0) >= 1, baseline_metrics)
check("Organism reports trusted gate count", organism_state.get("trusted_gate_count", 0) >= 1, organism_state)
check("Transition recorded", len(z3_state.get("transitions", [])) >= 1)
check("No raw internals at public root", "rules" not in z3_state and "concepts" not in z3_state and "z_cubed_state" not in z3_state)

rendered = OutputLayer().render("what is your state", {"z3": z3_state})
check("Output speaks from Z3", rendered.startswith("[Z3 Cycle"), rendered)
check("Output avoids raw dump header", "Weighted signals" not in rendered)

# Noisy novelty should be quarantined instead of promoted.
noisy = Z3Interface(novelty_threshold=0.35, severe_threshold=0.72)
noisy_state = dict(coordinator_state)
noisy_state["phi"] = 0.22
noisy_state["sigma"] = 0.91
noisy_z3 = noisy.project(
    coordinator_state=noisy_state,
    world_model=world_model,
    resonant_memory={},
    learning={},
    predictions=[],
    recursive_state={"loss_delta": 0.2, "world_model_memory_loss": 0.81},
).to_dict()
check("Noisy novelty rejected", noisy_z3["last_decision"].get("action") == "reject_noise", noisy_z3["last_decision"])
check("Noisy evidence keeps gate closed", noisy_z3["last_decision"].get("evidence_score", {}).get("gate_open") is False, noisy_z3["last_decision"])
check("Rejected novelty does not advance baseline", noisy_z3["baseline"].get("version") == 1)

# Persisted public state should restore baseline identity into a fresh interface.
restored = Z3Interface()
restored.restore_from_public_state(z3_state)
restored_state = restored.project(
    coordinator_state=coordinator_state,
    world_model={"latest": None},
    resonant_memory={},
    learning={},
    predictions=[],
    recursive_state=recursive_state,
).to_dict()
check("Restored baseline version survives", restored_state["baseline"].get("version") >= z3_state["baseline"].get("version"))
check("Restored transitions survive", len(restored_state.get("transitions", [])) >= 1)
check("Restored adjudication memory survives", restored_state["baseline"].get("metrics", {}).get("adjudication_memory", {}).get("entries", 0) >= 1)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL}")
if FAIL:
    sys.exit(1)
