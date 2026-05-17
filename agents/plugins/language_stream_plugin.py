"""
LanguageStreamPlugin
====================
Feeds real corpus language observations into the cognitive mesh.

The plugin supports three sources, in priority order:

1. ``LANGUAGE_TRAINING_TEXT`` for direct inline smoke tests.
2. ``LANGUAGE_TRAINING_CORPUS_PATH`` for local text files.
3. ``LANGUAGE_TRAINING_DATASET`` / ``Z3_CORPUS_DATASET`` for Hugging Face
   streaming datasets, defaulting to Project Gutenberg English.

Each fetched text segment is emitted as a ``language:corpus`` observation. The
DistributedCognitiveCore now consumes those observations through the live
``Z3CorpusNeuralIngestor``, so this plugin is no longer merely descriptive; it is
an actual corpus ingestion stream into the z³ neural runtime when PyTorch is
available.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agents.provider_base import DataPlugin

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SPACE_RE = re.compile(r"\s+")
_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


class LanguageStreamPlugin(DataPlugin):
    """Real-language data source for language-first Z³ training and testing."""

    name = "language_stream"
    domain = "language:corpus"

    def __init__(self) -> None:
        self.corpus_path = os.getenv("LANGUAGE_TRAINING_CORPUS_PATH", "").strip()
        self.inline_text = os.getenv("LANGUAGE_TRAINING_TEXT", "").strip()
        self.batch_size = int(os.getenv("LANGUAGE_TRAINING_BATCH_SIZE", "25"))
        self.min_words = int(os.getenv("LANGUAGE_STREAM_MIN_WORDS", os.getenv("Z3_CORPUS_MIN_WORDS", "24")))
        self.dataset_name = os.getenv("LANGUAGE_TRAINING_DATASET", os.getenv("Z3_CORPUS_DATASET", "manu/project_gutenberg")).strip()
        self.dataset_config = os.getenv("LANGUAGE_TRAINING_DATASET_CONFIG", os.getenv("Z3_CORPUS_DATASET_CONFIG", "")).strip()
        self.dataset_split = os.getenv("LANGUAGE_TRAINING_DATASET_SPLIT", os.getenv("Z3_CORPUS_DATASET_SPLIT", "en")).strip()
        self.text_field = os.getenv("LANGUAGE_TRAINING_TEXT_FIELD", os.getenv("Z3_CORPUS_TEXT_FIELD", "text")).strip()
        self.disable_remote = os.getenv("DISABLE_REMOTE_CORPUS_STREAM", "").lower() in ("1", "true", "yes")
        self._segments: List[str] = []
        self._offset = 0
        self._dataset_iter: Optional[Iterator[Dict[str, Any]]] = None
        self._source = "idle"
        self._last_error = ""
        self._total_emitted = 0

    async def initialize(self) -> None:
        text = self._load_text()
        if text:
            self._segments = self._split_units(text)
            self._source = "inline_text" if self.inline_text else "local_file"
            return

        if self.dataset_name and not self.disable_remote:
            self._initialize_dataset_stream()

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        observations: List[Tuple[Dict[str, Any], str]] = []
        if self._segments:
            for _ in range(min(self.batch_size, len(self._segments))):
                segment = self._segments[self._offset % len(self._segments)]
                if len(segment.split()) >= self.min_words:
                    observations.append((self.text_to_observation(segment, source=self._source), self.domain))
                    self._total_emitted += 1
                self._offset += 1
            return observations

        if self._dataset_iter is None:
            return []

        attempts = 0
        max_attempts = max(self.batch_size * 10, 25)
        while len(observations) < self.batch_size and attempts < max_attempts:
            attempts += 1
            try:
                row = next(self._dataset_iter)
            except StopIteration:
                self._dataset_iter = None
                self._source = "exhausted"
                break
            except Exception as exc:
                self._last_error = str(exc)
                break
            text = self._clean_dataset_text(str(row.get(self.text_field) or ""))
            if len(text.split()) < self.min_words:
                continue
            observation = self.text_to_observation(text, source=f"dataset:{self.dataset_name}")
            observation["dataset"] = self.dataset_name
            observation["dataset_split"] = self.dataset_split
            if row.get("id") is not None:
                observation["corpus_id"] = str(row.get("id"))
            if row.get("title") is not None:
                observation["title"] = str(row.get("title"))
            observations.append((observation, self.domain))
            self._total_emitted += 1
        return observations

    def stream_count(self) -> int:
        if self._segments:
            return len(self._segments)
        if self._dataset_iter is not None:
            return 1
        return 0

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self._source,
            "dataset": self.dataset_name if self._dataset_iter is not None else None,
            "dataset_split": self.dataset_split,
            "segments": len(self._segments),
            "total_emitted": self._total_emitted,
            "last_error": self._last_error,
        }

    def _load_text(self) -> str:
        if self.inline_text:
            return self.inline_text
        if self.corpus_path:
            path = Path(self.corpus_path).expanduser()
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8", errors="ignore")
            self._last_error = f"Corpus path not found: {path}"
        return ""

    def _initialize_dataset_stream(self) -> None:
        try:
            from datasets import load_dataset  # type: ignore
        except ModuleNotFoundError as exc:
            self._last_error = f"Hugging Face datasets package unavailable: {exc}"
            self._source = "remote_unavailable"
            return

        try:
            kwargs = {"split": self.dataset_split, "streaming": True}
            if self.dataset_config:
                dataset = load_dataset(self.dataset_name, self.dataset_config, **kwargs)
            else:
                dataset = load_dataset(self.dataset_name, **kwargs)
            self._dataset_iter = iter(dataset)
            self._source = f"dataset:{self.dataset_name}:{self.dataset_split}"
        except Exception as exc:
            self._last_error = f"Dataset stream failed: {exc}"
            self._dataset_iter = None
            self._source = "remote_failed"

    @staticmethod
    def _split_units(text: str) -> List[str]:
        return [unit.strip() for unit in _SENTENCE_RE.split(text or "") if unit.strip()]

    @staticmethod
    def _clean_dataset_text(text: str) -> str:
        text = text or ""
        start_match = _START_RE.search(text)
        if start_match:
            text = text[start_match.end():]
        end_match = _END_RE.search(text)
        if end_match:
            text = text[:end_match.start()]
        return _SPACE_RE.sub(" ", text.replace("\x00", " ")).strip()

    @staticmethod
    def text_to_observation(text: str, *, source: str = "language") -> Dict[str, Any]:
        tokens = [token for token in _TOKEN_RE.findall(text or "") if token.strip()]
        digest = hashlib.blake2b((text or "").encode("utf-8"), digest_size=12).hexdigest()
        token_count = len(tokens)
        unique_ratio = len(set(t.lower() for t in tokens)) / max(1, token_count)
        punctuation_count = sum(1 for token in tokens if not any(ch.isalnum() for ch in token))
        alpha_count = sum(1 for token in tokens if any(ch.isalpha() for ch in token))
        mean_token_length = sum(len(token) for token in tokens) / max(1, token_count)
        return {
            "entity_id": f"language_{digest}",
            "value": float(token_count),
            "secondary_value": float(unique_ratio),
            "timestamp": time.time(),
            "source": source,
            "text": text,
            "token_count": token_count,
            "unique_ratio": unique_ratio,
            "punctuation_ratio": punctuation_count / max(1, token_count),
            "alpha_ratio": alpha_count / max(1, token_count),
            "mean_token_length": mean_token_length,
        }
