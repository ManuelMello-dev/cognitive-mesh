"""
z3_corpus_streamer.py
=====================

Reusable streaming helpers for feeding public natural-language and literature
corpora into the existing z³ language training adapter in cognitive-mesh.

Install dependency if needed:
    sudo pip3 install datasets

Example use inside the cognitive-mesh project:
    from z3_corpus_streamer import gutenberg_batches
    from core.z3_language_training import train_z3_on_language_window

    for texts in gutenberg_batches(batch_size=8, max_batches=100):
        metrics = train_z3_on_language_window(model, optimizer, texts)
        print(metrics)
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, Iterator, List, Optional

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None


_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_SPACE_RE = re.compile(r"\s+")


def require_datasets() -> None:
    """Raise a clear error if Hugging Face Datasets is not installed."""
    if load_dataset is None:  # pragma: no cover
        raise ModuleNotFoundError("Install Hugging Face Datasets with: sudo pip3 install datasets") from _DATASETS_IMPORT_ERROR


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable document order."""
    return _SPACE_RE.sub(" ", (text or "").replace("\x00", " ")).strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove common Project Gutenberg header/footer markers from a book."""
    text = text or ""
    start_match = _START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    else:
        # Fallback for dataset variants with less standardized capitalization.
        marker = text.lower().find("*** start")
        if marker != -1:
            newline = text.find("\n", marker)
            if newline != -1:
                text = text[newline + 1:]

    end_match = _END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]
    else:
        marker = text.lower().find("*** end")
        if marker != -1:
            text = text[:marker]
    return normalize_text(text)


def stream_text_batches(
    dataset_name: str,
    *,
    split: str = "train",
    config: Optional[str] = None,
    text_field: str = "text",
    batch_size: int = 8,
    min_words: int = 200,
    cleaner: Optional[Callable[[str], str]] = normalize_text,
    row_filter: Optional[Callable[[Dict], bool]] = None,
    max_batches: Optional[int] = None,
) -> Iterator[List[str]]:
    """Yield batches of cleaned text rows from a Hugging Face streaming dataset."""
    require_datasets()
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if min_words < 1:
        raise ValueError("min_words must be >= 1")

    if config:
        stream = load_dataset(dataset_name, config, split=split, streaming=True)
    else:
        stream = load_dataset(dataset_name, split=split, streaming=True)

    emitted = 0
    batch: List[str] = []
    for row in stream:
        if row_filter is not None and not row_filter(row):
            continue
        text = row.get(text_field) or ""
        text = cleaner(text) if cleaner else text
        if len(text.split()) < min_words:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            emitted += 1
            if max_batches is not None and emitted >= max_batches:
                return
            batch = []

    if batch and (max_batches is None or emitted < max_batches):
        yield batch


def gutenberg_batches(batch_size: int = 8, max_batches: Optional[int] = None) -> Iterator[List[str]]:
    """Primary recommended z³ literature stream: Project Gutenberg English books."""
    return stream_text_batches(
        "manu/project_gutenberg",
        split="en",
        batch_size=batch_size,
        min_words=500,
        cleaner=strip_gutenberg_boilerplate,
        max_batches=max_batches,
    )


def common_corpus_openculture_batches(batch_size: int = 8, max_batches: Optional[int] = None) -> Iterator[List[str]]:
    """Long-term open/public-domain cultural stream from Common Corpus."""
    def keep(row: Dict) -> bool:
        language_ok = row.get("language") == "en"
        culture_ok = row.get("open type") == "OpenCulture"
        words_ok = (row.get("word_count") or 0) >= 300
        return language_ok and culture_ok and words_ok

    return stream_text_batches(
        "PleIAs/common_corpus",
        split="train",
        batch_size=batch_size,
        min_words=300,
        cleaner=normalize_text,
        row_filter=keep,
        max_batches=max_batches,
    )


def british_library_clean_batches(batch_size: int = 8, max_batches: Optional[int] = None) -> Iterator[List[str]]:
    """Historical out-of-copyright book stream filtered for English and OCR confidence."""
    def keep(row: Dict) -> bool:
        english = row.get("Language_1") == "English"
        non_empty = not row.get("empty_pg") and bool(row.get("text"))
        ocr_ok = (row.get("mean_wc_ocr") or 0.0) >= 80.0
        return english and non_empty and ocr_ok

    return stream_text_batches(
        "TheBritishLibrary/blbooks",
        config="all",
        split="train",
        batch_size=batch_size,
        min_words=100,
        cleaner=normalize_text,
        row_filter=keep,
        max_batches=max_batches,
    )
