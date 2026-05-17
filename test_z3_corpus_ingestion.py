"""
Z³ corpus ingestion stream smoke tests.

These tests prove that language corpus text is emitted as real observations and
that the neural ingestor consumes those observations through the existing
Z3NeuralDynamics language-training path when PyTorch is installed.
"""
import asyncio
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
print("TESTING Z³ CORPUS INGESTION STREAM")
print("=" * 60)

os.environ["LANGUAGE_TRAINING_TEXT"] = (
    "Z3 receives a real corpus stream, not a placeholder. "
    "Each sentence becomes a language observation with raw text attached. "
    "The neural ingestor buffers the text and trains the z cubed runtime. "
    "This confirms that corpus ingestion is wired end to end."
)
os.environ["LANGUAGE_TRAINING_BATCH_SIZE"] = "2"
os.environ["Z3_CORPUS_MIN_WORDS"] = "3"
os.environ["DISABLE_REMOTE_CORPUS_STREAM"] = "1"

from agents.plugins.language_stream_plugin import LanguageStreamPlugin


async def exercise_plugin():
    plugin = LanguageStreamPlugin()
    await plugin.initialize()
    observations = await plugin.fetch()
    return plugin, observations


plugin, observations = asyncio.run(exercise_plugin())
check("Language plugin emits corpus observations", len(observations) > 0, observations)
if observations:
    obs, domain = observations[0]
    check("Corpus observation uses language domain", domain == "language:corpus", domain)
    check("Corpus observation carries raw text", isinstance(obs.get("text"), str) and len(obs["text"]) > 0, obs)
    check("Corpus observation has numeric value", isinstance(obs.get("value"), float), obs)
check("Language plugin reports active stream count", plugin.stream_count() > 0, plugin.status())

try:
    import torch  # noqa: F401
    from core.z3_corpus_ingestion import Z3CorpusIngestionConfig, Z3CorpusNeuralIngestor
except ModuleNotFoundError as exc:
    skip("Neural corpus ingestor", str(exc))
else:
    config = Z3CorpusIngestionConfig(
        enabled=True,
        batch_size=2,
        min_words=3,
        max_buffer_texts=8,
        max_train_batches_per_flush=1,
        window_size=6,
        stride=3,
        truncation_steps=2,
        checkpoint_every_steps=0,
    )
    ingestor = Z3CorpusNeuralIngestor(config)
    if not ingestor.available:
        skip("Neural corpus ingestor availability", ingestor.snapshot().get("last_error"))
    else:
        before = ingestor.model.z_cubed_state.detach().clone()
        for obs, _domain in observations[:2]:
            ingestor.observe_text(obs["text"], metadata=obs)
        ingestor.flush(force=True)
        after = ingestor.model.z_cubed_state.detach().clone()
        status = ingestor.snapshot()
        check("Neural ingestor sees corpus text", status["texts_seen"] >= 1, status)
        check("Neural ingestor performs training", status["trained_steps"] >= 1, status)
        check("Neural ingestor returns metrics", bool(status["last_metrics"]), status)
        check("Corpus ingestion advances z³ neural state", bool(torch.norm(after - before) > 0), status)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
