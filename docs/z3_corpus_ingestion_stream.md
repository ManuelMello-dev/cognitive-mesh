# Z³ Corpus Ingestion Stream

The z³ corpus ingestion stream is now a live runtime path, not only a recommendation or helper script. The stream begins in `LanguageStreamPlugin`, emits `language:corpus` observations containing raw text, and is consumed by `Z3CorpusNeuralIngestor` inside `DistributedCognitiveCore.ingest()`. When PyTorch is installed, those text observations are buffered and trained into `Z3NeuralDynamics` through `train_z3_on_language_window()`.

## Runtime path

| Stage | File | Responsibility |
|---|---|---|
| Corpus source | `agents/plugins/language_stream_plugin.py` | Reads inline text, a local corpus file, or a Hugging Face streaming dataset. |
| Observation ingestion | `core/distributed_core.py` | Receives `language:corpus` observations and forwards their `text` field into the neural ingestor. |
| Neural training bridge | `core/z3_corpus_ingestion.py` | Buffers text batches, calls `train_z3_on_language_window()`, updates z³ recurrent state, and checkpoints the neural runtime. |
| Existing language adapter | `core/z3_language_training.py` | Converts raw language windows into `[steps, input_dim]` or `[batch, steps, input_dim]` tensors. |
| Existing neural runtime | `core/z3_neural_dynamics.py` | Performs the actual z³ / z-prime neural state update and truncated BPTT training. |

## Default corpus source

If no local corpus text is configured, the plugin attempts to use the Hugging Face dataset stream:

```text
LANGUAGE_TRAINING_DATASET=manu/project_gutenberg
LANGUAGE_TRAINING_DATASET_SPLIT=en
LANGUAGE_TRAINING_TEXT_FIELD=text
```

This can be overridden for other corpora:

```text
LANGUAGE_TRAINING_DATASET=PleIAs/common_corpus
LANGUAGE_TRAINING_DATASET_SPLIT=train
LANGUAGE_TRAINING_TEXT_FIELD=text
```

For local operation without Hugging Face, set either:

```text
LANGUAGE_TRAINING_TEXT="direct text used for smoke testing"
```

or:

```text
LANGUAGE_TRAINING_CORPUS_PATH=/path/to/corpus.txt
```

## Neural ingestion controls

The neural ingestion bridge is enabled by default, but it requires the optional neural dependencies in `requirements-neural.txt`. If PyTorch is unavailable, the system does **not** pretend training is happening; the ingestor exposes `available=false` and records the import error in its status.

| Variable | Default | Meaning |
|---|---:|---|
| `DISABLE_Z3_CORPUS_INGESTION` | unset | Set to `1` to disable neural corpus ingestion. |
| `Z3_CORPUS_TRAIN_BATCH_SIZE` | `LANGUAGE_TRAINING_BATCH_SIZE` or `8` | Number of text samples per neural training update. |
| `Z3_CORPUS_MIN_WORDS` | `24` | Minimum words required before text is accepted into the neural buffer. |
| `Z3_CORPUS_WINDOW_SIZE` | `24` | Token window size used by the language adapter. |
| `Z3_CORPUS_STRIDE` | `12` | Token stride used by the language adapter. |
| `Z3_CORPUS_TRUNCATION_STEPS` | `16` | Truncated BPTT length for z³ sequence training. |
| `Z3_CORPUS_CHECKPOINT_PATH` | `outputs/z3_neural_language.pt` | Neural checkpoint path for the corpus-trained z³ runtime. |

## Validation

Run the smoke test with:

```bash
python3.11 test_z3_corpus_ingestion.py
```

On lightweight environments without PyTorch, the test verifies the corpus observation stream and skips only the neural-training assertion. On neural environments with PyTorch installed, it additionally verifies that corpus text advances the persistent z³ neural state.
