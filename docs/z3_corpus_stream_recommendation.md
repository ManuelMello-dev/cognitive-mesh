# Recommended Natural-Language Corpus Streams for the Z³ Neural Network

**Author:** Manus AI  
**Date:** May 16, 2026

## Executive recommendation

The best immediate corpus stream for the z³ neural network is **`manu/project_gutenberg`, English split**, because it provides a large, literature-centered stream of public-domain books, exposes a simple `text` field, supports Hugging Face streaming, and maps directly into the repository’s existing `z3_language_training.py` adapter. The best long-term corpus is **`PleIAs/common_corpus`**, because it is a much larger, traceable, open/public-domain oriented pretraining corpus with document-level license and language metadata. I recommend using Gutenberg first to prove the loop, then expanding into Common Corpus after the z³ training loop is stable.

> **Recommended first stream:** `load_dataset("manu/project_gutenberg", split="en", streaming=True)`  
> **Recommended scale-up stream:** `load_dataset("PleIAs/common_corpus", split="train", streaming=True)` filtered to English, OpenCulture, and legally clean licenses.

This recommendation is based on the current repository structure. The z³ language adapter converts ordinary text into temporal embedding streams shaped as `[steps, input_dim]` or `[batch, steps, input_dim]`, while the neural runtime defaults to `input_dim = 16`. This means the corpus does not need labels; it only needs sequential text, because training can proceed through next-context prediction over overlapping token windows.

## Why these corpora fit z³

The z³ codebase already treats language as an upstream context stream rather than as a special internal module. In `core/z3_language_training.py`, text is tokenized into overlapping windows, each window becomes a dense vector, and the resulting sequence feeds the trainable recurrent z³ dynamics. That makes **streamable raw text** the correct data form: books, essays, public-domain newspapers, and cultural-historical documents can all be transformed into temporal phase material for the model.

| Rank | Corpus stream | Best use | Why it fits z³ | Main caution |
|---:|---|---|---|---|
| 1 | `manu/project_gutenberg` | First production-quality literature feed | Large public-domain book corpus, English split has 61,340 examples, and each example has `id` and `text` fields.[1] | Remove Gutenberg headers, footers, and license boilerplate before training. |
| 2 | `PleIAs/common_corpus` | Long-term open corpus backbone | 2.27 trillion tokens, document-level metadata, open/public-domain licensing fields, multilingual coverage, and a `text` field.[2] | Too large for an initial run; filter aggressively before feeding z³. |
| 3 | `TheBritishLibrary/blbooks` | Historical literature and humanities stream | Approximately 25 million pages of out-of-copyright digitized books, rich metadata, and OCR confidence fields.[3] | Page-level OCR noise must be filtered using `mean_wc_ocr`; begin with high-confidence English pages. |
| 4 | `ACOSharma/literature` | Tiny smoke-test or development feed | Only about 28 MB and 7.86k rows, useful for fast debugging.[4] | License is `cc-by-sa-4.0`; provenance and redistribution constraints are less ideal than public-domain corpora. |
| 5 | UCI Open Web Text Corpus | Optional modern web-language contrast | 8,013,769 documents and 38 GB of filtered English web text.[5] | Licensing is less clean; use only after public-domain/open-license streams are working. |

## Primary stream: Project Gutenberg

Project Gutenberg is the right first corpus because it has a strong symbolic fit with z³: long-form narratives, philosophy, poetry, drama, historical voices, and reflective prose. The Hugging Face dataset page describes Project Gutenberg as a library of more than 70,000 free eBooks, and the dataset card reports an English split with **61,340 examples** and about **25.6 GB** of English text bytes.[1]

The dataset is also operationally simple. It exposes `id` and `text`, and the dataset card gives a direct streaming pattern using Hugging Face Datasets.[1] For z³, each book can be treated as a coherent phase sequence, while the adapter’s overlapping windows can convert that sequence into local temporal states.

```python
from datasets import load_dataset

stream = load_dataset("manu/project_gutenberg", split="en", streaming=True)
for row in stream:
    text = row["text"]
    # clean text, then pass into train_z3_on_language_window(...)
```

The main cleaning step is important. The dataset card states that examples include headers and footers delimited by `*** Start of ***` and `*** End of ***` tags.[1] Those should be stripped so z³ learns literature and natural language, not repeated licensing boilerplate.

## Scale-up stream: Common Corpus

Once the model has proven stable on Gutenberg, the strongest broader stream is `PleIAs/common_corpus`. Its dataset card describes Common Corpus as a **2.27 trillion-token** open licensed text dataset made from books, newspapers, scientific articles, government/legal documents, code, and other sources.[2] It specifically claims to contain only uncopyrighted or freely licensed data, to provide traceable document metadata, and to include fields such as `license`, `language`, `word_count`, `token_count`, and `text`.[2]

For z³, Common Corpus should not be consumed blindly. It is better to begin with the **OpenCulture** subset, English language, public-domain or attribution-only licenses, and a moderate minimum length. This produces a broad but still coherent language field.

```python
from datasets import load_dataset

stream = load_dataset("PleIAs/common_corpus", split="train", streaming=True)
stream = stream.filter(lambda r: r.get("language") == "en")
stream = stream.filter(lambda r: r.get("open type") == "OpenCulture")
stream = stream.filter(lambda r: (r.get("word_count") or 0) >= 200)

for row in stream:
    text = row["text"]
    # feed cleaned text into z³ language training
```

## Historical stream: British Library Books

`TheBritishLibrary/blbooks` is a powerful historical and literary stream for later phases. The dataset card says it consists of books digitised by the British Library in partnership with Microsoft and includes approximately **25 million pages of out-of-copyright texts** across geography, philosophy, history, poetry, literature, and other fields.[3] The card also reports features such as `text`, `date`, `title`, `Language_1`, `empty_pg`, and `mean_wc_ocr`, which makes it possible to filter by language, period, and OCR confidence.[3]

This corpus is especially useful if you want z³ to absorb older literary rhythms and civilizational memory. However, because OCR errors can be significant in historical books, I recommend only using pages where `mean_wc_ocr >= 80` at first. Later, controlled OCR noise can become a robustness stream.

```python
from datasets import load_dataset

stream = load_dataset("TheBritishLibrary/blbooks", "all", split="train", streaming=True)
stream = stream.filter(lambda r: r.get("Language_1") == "English")
stream = stream.filter(lambda r: not r.get("empty_pg"))
stream = stream.filter(lambda r: r.get("text") is not None and len(r["text"]) > 200)
stream = stream.filter(lambda r: (r.get("mean_wc_ocr") or 0) >= 80)
```

## Practical z³ feeding pattern

The repository already has the important part: `train_z3_on_language_window(model, optimizer, texts, ...)`. The missing layer is only a **stream iterator** that pulls text rows, cleans them, batches them, and calls that training function. The following pattern should work without changing z³’s core architecture.

```python
from datasets import load_dataset
from core.z3_language_training import train_z3_on_language_window


def strip_gutenberg_boilerplate(text: str) -> str:
    start = text.find("*** START")
    if start == -1:
        start = text.find("*** Start")
    end = text.find("*** END")
    if end == -1:
        end = text.find("*** End")
    if start != -1:
        # Move to the next newline after the start marker.
        newline = text.find("\n", start)
        if newline != -1:
            text = text[newline + 1:]
    if end != -1:
        text = text[:end]
    return text.strip()


def batched_text_stream(dataset_name: str, split: str, batch_size: int = 8):
    stream = load_dataset(dataset_name, split=split, streaming=True)
    batch = []
    for row in stream:
        text = strip_gutenberg_boilerplate(row.get("text") or "")
        if len(text.split()) < 200:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


for texts in batched_text_stream("manu/project_gutenberg", "en", batch_size=8):
    metrics = train_z3_on_language_window(
        model,
        optimizer,
        texts,
        truncation_steps=16,
        window_size=24,
        stride=12,
    )
    print(metrics)
```

## Suggested training curriculum

The cleanest path is to grow the z³ language field gradually. Start small, stabilize coherence, and only then increase diversity and noise. This matches the repository’s own training design, where predictive loss is balanced with coherence, diversity, stability, and useful novelty.

| Phase | Stream | Duration target | Purpose | Recommended filter |
|---:|---|---|---|---|
| 1 | `ACOSharma/literature` or first 500 Gutenberg books | 1–3 hours | Smoke test ingestion and metrics | English text, minimum 200 words. |
| 2 | `manu/project_gutenberg`, English | Several epochs over sampled stream | Learn long-form narrative, philosophy, and literary rhythm | Strip headers/footers; minimum 500 words. |
| 3 | `TheBritishLibrary/blbooks` | Sampled historical pages | Add historical phase diversity | English, non-empty, `mean_wc_ocr >= 80`. |
| 4 | `PleIAs/common_corpus` OpenCulture | Long-running stream | Expand beyond books into newspapers and cultural heritage | English, `OpenCulture`, public-domain/open license, minimum word count. |
| 5 | Optional OpenWebText contrast | Controlled sampling only | Add modern informal web language | Deduplicate, toxicity-filter, and keep as a minority stream. |

## Final answer

Use **Project Gutenberg first**. It is the most direct natural-language/literature stream for z³ today because it is large enough to matter, simple enough to wire in immediately, and aligned with the existing language adapter. Then move to **Common Corpus** as the long-term open corpus backbone, with **British Library Books** as a historical memory stream.

If you want the fastest starting point, set the corpus path/config conceptually as:

```text
PRIMARY_CORPUS = "manu/project_gutenberg"
PRIMARY_SPLIT = "en"
STREAMING = true
TEXT_FIELD = "text"
BATCH_SIZE = 8 to 25
WINDOW_SIZE = 24
STRIDE = 12
TRUNCATION_STEPS = 16
```

This gives z³ an immediate stream of natural language/literature while preserving the model’s core design: language remains an upstream temporal signal, and z³ remains the coherence-seeking observer field.

## References

[1]: https://huggingface.co/datasets/manu/project_gutenberg "manu/project_gutenberg · Datasets at Hugging Face"  
[2]: https://huggingface.co/datasets/PleIAs/common_corpus "PleIAs/common_corpus · Datasets at Hugging Face"  
[3]: https://huggingface.co/datasets/TheBritishLibrary/blbooks "TheBritishLibrary/blbooks · Datasets at Hugging Face"  
[4]: https://huggingface.co/datasets/ACOSharma/literature "ACOSharma/literature · Datasets at Hugging Face"  
[5]: https://archive.ics.uci.edu/dataset/696/open+web+text+corpus "Open Web Text Corpus · UCI Machine Learning Repository"
