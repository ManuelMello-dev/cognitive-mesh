"""
LanguageStreamPlugin
====================
Feeds real language observations into the cognitive mesh.

The plugin is intentionally corpus-driven. It reads actual text supplied through
``LANGUAGE_TRAINING_CORPUS_PATH`` or ``LANGUAGE_TRAINING_TEXT`` and converts each
sentence-like unit into a generic observation under the ``language:corpus``
domain. When no corpus is configured it remains idle, while the chat endpoint can
still inject user messages as ``language:chat`` observations for interaction and
testing.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.provider_base import DataPlugin

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class LanguageStreamPlugin(DataPlugin):
    """Real-language data source for language-first Z³ training and testing."""

    name = "language_stream"
    domain = "language:corpus"

    def __init__(self) -> None:
        self.corpus_path = os.getenv("LANGUAGE_TRAINING_CORPUS_PATH", "").strip()
        self.inline_text = os.getenv("LANGUAGE_TRAINING_TEXT", "").strip()
        self.batch_size = int(os.getenv("LANGUAGE_TRAINING_BATCH_SIZE", "25"))
        self._segments: List[str] = []
        self._offset = 0

    async def initialize(self) -> None:
        text = self._load_text()
        self._segments = self._split_units(text)

    async def fetch(self) -> List[Tuple[Dict[str, Any], str]]:
        if not self._segments:
            return []
        observations: List[Tuple[Dict[str, Any], str]] = []
        for _ in range(min(self.batch_size, len(self._segments))):
            segment = self._segments[self._offset % len(self._segments)]
            observations.append((self.text_to_observation(segment, source="language_corpus"), self.domain))
            self._offset += 1
        return observations

    def _load_text(self) -> str:
        if self.corpus_path:
            path = Path(self.corpus_path).expanduser()
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8", errors="ignore")
        return self.inline_text

    @staticmethod
    def _split_units(text: str) -> List[str]:
        return [unit.strip() for unit in _SENTENCE_RE.split(text or "") if unit.strip()]

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
