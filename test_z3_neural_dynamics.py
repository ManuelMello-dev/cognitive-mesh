"""
Z³ neural dynamics smoke tests.
Runs offline and skips gracefully when PyTorch is not installed.
"""
import os
import sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

PASS = 0
FAIL = 0
SKIP = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


def skip(name, detail=""):
    global SKIP
    SKIP += 1
    print(f"  - {name} skipped — {detail}")


print("=" * 60)
print("TESTING Z³ NEURAL DYNAMICS V2")
print("=" * 60)

try:
    import torch
    from core.z3_neural_dynamics import Z3NeuralConfig, Z3NeuralDynamics, generate_regime_sequence, prepare_embedding_pairs
except ModuleNotFoundError as exc:
    skip("PyTorch neural dynamics", str(exc))
    print("=" * 60)
    print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
    raise SystemExit(0)


torch.manual_seed(11)
config = Z3NeuralConfig(
    input_dim=8,
    context_dim=16,
    state_dim=24,
    local_dim=12,
    evidence_dim=10,
    hidden_dim=32,
    agent_count=4,
    agent_embed_dim=6,
    noise_scale=0.0,
)
model = Z3NeuralDynamics(config)
coherence_config = Z3NeuralConfig.internal_coherence(input_dim=8)
balanced_config = Z3NeuralConfig.balanced(input_dim=8)

check("Internal coherence preset down-weights prediction", coherence_config.beta_predictive < config.beta_predictive, coherence_config.beta_predictive)
check("Balanced preset keeps prediction and coherence active", balanced_config.beta_predictive > 0 and balanced_config.beta_coherence_band > 0, balanced_config)
check("Initial phi starts near unit gain", bool(torch.allclose(model.phi, torch.ones_like(model.phi), atol=2e-4)), model.phi)
check("Metric buffer size follows metric schema", tuple(model.last_metrics.shape) == (model.metric_count(),), model.last_metrics.shape)

x, y = generate_regime_sequence(8, config.input_dim, batch_size=2)
real_stream = torch.randn(2, 5, config.input_dim)
real_x, real_y = prepare_embedding_pairs(real_stream)
check("Real embedding adapter flattens next-step pairs", tuple(real_x.shape) == (8, config.input_dim) and tuple(real_y.shape) == (8, config.input_dim), (real_x.shape, real_y.shape))
output = model.forward(x[:4], target=y[:4], update_state=False, add_noise=False)

check("Z³ before shape", tuple(output["z_cubed_before"].shape) == (4, config.state_dim), output["z_cubed_before"].shape)
check("Z³ after shape", tuple(output["z_cubed_after"].shape) == (4, config.state_dim), output["z_cubed_after"].shape)
check("Z-prime agent shape", tuple(output["z_prime_after"].shape) == (4, config.agent_count, config.local_dim), output["z_prime_after"].shape)
check("Evidence shape", tuple(output["evidence"].shape) == (4, config.agent_count, config.evidence_dim), output["evidence"].shape)
check("Soft gates bounded", bool(torch.all(output["gate"] >= 0.0) and torch.all(output["gate"] <= 1.0)), output["gate"])
check("Weights normalized", bool(torch.allclose(output["weights"].sum(dim=1), torch.ones(4), atol=1e-4)), output["weights"].sum(dim=1))
check("Weights remain positive under trust floor", bool(torch.all(output["weights"] > 0.0)), output["weights"])
check("Loss is finite", bool(torch.isfinite(output["losses"]["total"])), output["losses"]["total"])
check("Metric vector matches metric schema", tuple(output["metrics"].shape) == (model.metric_count(),), output["metrics"].shape)

zero_trust = torch.zeros(3, config.agent_count)
zero_weights = model._normalize_trust(zero_trust)
expected_uniform = torch.full_like(zero_weights, 1.0 / config.agent_count)
check("Zero-trust fallback is uniform", bool(torch.allclose(zero_weights, expected_uniform, atol=1e-6)), zero_weights)

states = torch.randn(2, config.agent_count, config.local_dim)
distances = torch.cdist(states, states, p=2)
mask = torch.triu(torch.ones(config.agent_count, config.agent_count, dtype=torch.bool), diagonal=1)
manual_batch_local = distances[:, mask].mean()
helper_batch_local = model._mean_agent_pairwise_distance(states)
flattened_cross_batch = torch.pdist(states.reshape(-1, config.local_dim), p=2).mean()
check("Batch-local diversity helper matches manual calculation", bool(torch.allclose(helper_batch_local, manual_batch_local, atol=1e-6)), (helper_batch_local, manual_batch_local))
check("Batch-local diversity avoids flattened cross-batch calculation", bool(not torch.allclose(helper_batch_local, flattened_cross_batch, atol=1e-6)), (helper_batch_local, flattened_cross_batch))

clustered = torch.zeros(1, config.agent_count, config.local_dim)
clustered[0, 1, 0] = 0.01
clustered[0, 2, 0] = 0.02
clustered[0, 3, 0] = 0.03
repulsion = model._pairwise_repulsion_field(clustered)
check("Pairwise repulsion field is active for clustered agents", bool(torch.norm(repulsion) > 0), repulsion)
check("Pairwise repulsion preserves zero-sum internal pressure", bool(torch.allclose(repulsion.sum(dim=1), torch.zeros(1, config.local_dim), atol=1e-5)), repulsion.sum(dim=1))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
before = model.z_cubed_state.detach().clone()
metrics = model.train_step(optimizer, x[:8], target=y[:8], update_recurrent_state=True)
after = model.z_cubed_state.detach().clone()

check("Train step returns metrics", isinstance(metrics, dict) and "loss_total" in metrics, metrics)
check("Train step loss finite", metrics.get("loss_total", float("nan")) == metrics.get("loss_total", float("nan")), metrics)
check("Persistent Z³ state advances", bool(torch.norm(after - before) > 0), (before, after))

stream_model = Z3NeuralDynamics(config)
stream_optimizer = torch.optim.AdamW(stream_model.parameters(), lr=1e-3)
stream_before = stream_model.z_cubed_state.detach().clone()
window_metrics = stream_model.train_sequence_window(stream_optimizer, real_stream, truncation_steps=2, commit_recurrent_state=True, add_noise=False)
stream_after = stream_model.z_cubed_state.detach().clone()
check("Truncated BPTT window returns metrics", isinstance(window_metrics, dict) and "window_loss" in window_metrics, window_metrics)
check("Truncated BPTT chunk count is bounded", window_metrics.get("truncated_bptt_chunks") == 2.0, window_metrics)
check("Truncated BPTT commits detached recurrent state", bool(torch.norm(stream_after - stream_before) > 0 and not stream_model.z_cubed_state.requires_grad), stream_model.z_cubed_state)

projection = model.public_projection(output)
check("Public projection exposes z_cubed_state", "z_cubed_state" in projection, projection)
check("Public projection exposes phi", "phi" in projection and 0.0 <= projection["phi"] <= 1.0, projection)
check("Public projection exposes learning metrics", "learning" in projection, projection)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
