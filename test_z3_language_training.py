"""
Z³ language training adapter smoke tests.
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
print("TESTING Z³ LANGUAGE TRAINING ADAPTER")
print("=" * 60)

try:
    import torch
    from core.z3_language_training import (
        LanguageAdapterConfig,
        LanguageEmbeddingAdapter,
        build_language_embedding_stream,
        split_language_corpus,
        train_z3_on_language_window,
    )
    from core.z3_neural_dynamics import Z3NeuralConfig, Z3NeuralDynamics
except ModuleNotFoundError as exc:
    skip("PyTorch language training adapter", str(exc))
    print("=" * 60)
    print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
    raise SystemExit(0)


torch.manual_seed(17)
text = (
    "Z3 observes a stream of meaning. "
    "Each local agent carries a partial hypothesis. "
    "Language supplies continuity, surprise, memory, and return paths. "
    "The global observer integrates novelty without collapsing diversity."
)
texts = [text, text.replace("meaning", "dialogue").replace("global", "recursive")]

config = LanguageAdapterConfig(input_dim=8, window_size=6, stride=3)
adapter = LanguageEmbeddingAdapter(config)
stream = adapter.encode_text(text)
batch_stream = adapter.encode_texts(texts)
x, y = adapter.prepare_training_pairs(texts)

check("Single language stream has Z3 input dimension", stream.dim() == 2 and stream.shape[-1] == 8, stream.shape)
check("Single language stream has at least two steps", stream.shape[0] >= 2, stream.shape)
check("Batched language stream has batch dimension", tuple(batch_stream.shape[:1]) == (2,), batch_stream.shape)
check("Training pairs flatten next-step language targets", x.shape == y.shape and x.shape[-1] == 8 and x.shape[0] > 0, (x.shape, y.shape))
check("Language vectors are finite", bool(torch.isfinite(batch_stream).all()), batch_stream)
check("Language vectors are normalized", bool(torch.allclose(torch.norm(stream, dim=-1), torch.ones(stream.shape[0]), atol=1e-5)), torch.norm(stream, dim=-1))

sentences = split_language_corpus(text)
check("Corpus splitter returns sentence-like units", len(sentences) >= 2, sentences)

convenience_single = build_language_embedding_stream(text, input_dim=8, window_size=6, stride=3)
convenience_batch = build_language_embedding_stream(texts, input_dim=8, window_size=6, stride=3)
check("Convenience single stream shape", convenience_single.dim() == 2 and convenience_single.shape[-1] == 8, convenience_single.shape)
check("Convenience batch stream shape", convenience_batch.dim() == 3 and convenience_batch.shape[0] == 2, convenience_batch.shape)

z3_config = Z3NeuralConfig(
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
model = Z3NeuralDynamics(z3_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
before = model.z_cubed_state.detach().clone()
metrics = train_z3_on_language_window(
    model,
    optimizer,
    texts,
    truncation_steps=2,
    window_size=6,
    stride=3,
    commit_recurrent_state=True,
    add_noise=False,
)
after = model.z_cubed_state.detach().clone()

check("Language window training returns metrics", isinstance(metrics, dict) and "window_loss" in metrics, metrics)
check("Language window loss is finite", metrics.get("window_loss", float("nan")) == metrics.get("window_loss", float("nan")), metrics)
check("Language training advances persistent Z3 state", bool(torch.norm(after - before) > 0), (before, after))

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
