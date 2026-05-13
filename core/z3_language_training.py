"""
Z³ language training adapter
============================

This module converts ordinary language streams into dense temporal vectors that
can train ``Z3NeuralDynamics`` through its existing embedding-stream contract.
It deliberately keeps language outside the Z³ core: text becomes an upstream
context stream, and the neural runtime remains modality-agnostic.

The default encoder is lightweight and deterministic so smoke tests can run
offline. Production deployments can replace ``LanguageEmbeddingAdapter.encode``
with embeddings from a transformer, memory system, or external model as long as
the returned tensor has shape ``[steps, input_dim]`` or
``[batch, steps, input_dim]``.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - exercised when torch is installed.
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - graceful import behavior.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

if torch is not None:  # pragma: no cover - exercised when torch is installed.
    from core.z3_neural_dynamics import prepare_embedding_pairs
else:  # pragma: no cover - graceful behavior without torch.
    prepare_embedding_pairs = None  # type: ignore[assignment]


_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class LanguageAdapterConfig:
    """Configuration for deterministic language-to-vector conversion."""

    input_dim: int = 16
    window_size: int = 24
    stride: int = 12
    min_tokens_per_step: int = 1
    lowercase: bool = True
    l2_normalize: bool = True
    include_position: bool = True

    def __post_init__(self) -> None:
        if self.input_dim < 4:
            raise ValueError("input_dim must be at least 4 so lexical and rhythm features can coexist")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.min_tokens_per_step < 1:
            raise ValueError("min_tokens_per_step must be >= 1")


class LanguageEmbeddingAdapter:
    """Build Z³-compatible embedding streams from language.

    The adapter treats each text as an ordered stream of overlapping token
    windows. Every window becomes one dense vector. This preserves enough
    temporal structure for next-context training while avoiding any dependency
    on a specific large language model provider.
    """

    def __init__(self, config: Optional[LanguageAdapterConfig] = None) -> None:
        if torch is None:  # pragma: no cover - depends on local environment.
            raise ModuleNotFoundError("PyTorch is required for language training tensors") from _TORCH_IMPORT_ERROR
        self.config = config or LanguageAdapterConfig()

    def encode_text(self, text: str) -> "torch.Tensor":
        """Encode a single document or conversation transcript as ``[steps, input_dim]``."""
        tokens = self._tokenize(text)
        if len(tokens) < self.config.min_tokens_per_step:
            raise ValueError("text does not contain enough tokens to produce a language stream")

        windows = self._windows(tokens)
        vectors = [self._encode_window(window, index, len(windows)) for index, window in enumerate(windows)]
        stream = torch.stack(vectors, dim=0)
        if stream.shape[0] < 2:
            # Z³ sequence training needs at least one (x_t, x_t+1) pair. Duplicate
            # very short streams with a small deterministic position perturbation.
            duplicate = stream.clone()
            if self.config.include_position:
                duplicate[:, -1] = 1.0
            stream = torch.cat([stream, duplicate], dim=0)
        return stream

    def encode_texts(self, texts: Sequence[str]) -> "torch.Tensor":
        """Encode multiple texts as a padded batched stream ``[batch, steps, input_dim]``.

        Shorter streams are padded by repeating their final state. Repetition is
        preferable to zero-padding here because Z³ should experience sustained
        context rather than artificial silence at the end of shorter documents.
        """
        if not texts:
            raise ValueError("texts must contain at least one text item")
        streams = [self.encode_text(text) for text in texts]
        max_steps = max(stream.shape[0] for stream in streams)
        padded = []
        for stream in streams:
            if stream.shape[0] < max_steps:
                pad = stream[-1:, :].expand(max_steps - stream.shape[0], -1)
                stream = torch.cat([stream, pad], dim=0)
            padded.append(stream)
        return torch.stack(padded, dim=0)

    def prepare_training_pairs(self, texts: Sequence[str]) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return flattened next-step pairs for ``Z3NeuralDynamics.train_step``."""
        if prepare_embedding_pairs is None:  # pragma: no cover - depends on local environment.
            raise ModuleNotFoundError("PyTorch is required for language training pairs") from _TORCH_IMPORT_ERROR
        return prepare_embedding_pairs(self.encode_texts(texts))

    def _tokenize(self, text: str) -> List[str]:
        text = text or ""
        if self.config.lowercase:
            text = text.lower()
        return [token for token in _TOKEN_RE.findall(text) if token.strip()]

    def _windows(self, tokens: Sequence[str]) -> List[List[str]]:
        windows: List[List[str]] = []
        size = self.config.window_size
        stride = self.config.stride
        for start in range(0, len(tokens), stride):
            window = list(tokens[start : start + size])
            if len(window) >= self.config.min_tokens_per_step:
                windows.append(window)
            if start + size >= len(tokens):
                break
        return windows or [list(tokens)]

    def _encode_window(self, tokens: Sequence[str], index: int, total_windows: int) -> "torch.Tensor":
        cfg = self.config
        vector = torch.zeros(cfg.input_dim, dtype=torch.float32)
        lexical_dims = max(1, cfg.input_dim - 4)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % lexical_dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 / math.sqrt(max(1, len(tokens)))
            vector[bucket] += sign * weight

        lengths = [len(token) for token in tokens]
        alpha_count = sum(1 for token in tokens if any(char.isalpha() for char in token))
        punct_count = sum(1 for token in tokens if not any(char.isalnum() for char in token))
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        mean_length = sum(lengths) / max(1, len(lengths))

        tail = cfg.input_dim - 4
        vector[tail] = min(1.0, mean_length / 12.0)
        vector[tail + 1] = unique_ratio
        vector[tail + 2] = punct_count / max(1, len(tokens))
        vector[tail + 3] = alpha_count / max(1, len(tokens))

        if cfg.include_position and cfg.input_dim >= 2:
            position = index / max(1, total_windows - 1)
            vector[-2] = math.sin(position * math.pi)
            vector[-1] = math.cos(position * math.pi)

        if cfg.l2_normalize:
            norm = torch.norm(vector, p=2).clamp_min(1e-6)
            vector = vector / norm
        return vector


def split_language_corpus(text: str) -> List[str]:
    """Split raw corpus text into non-empty sentence-like units."""
    units = [unit.strip() for unit in _SENTENCE_RE.split(text or "") if unit.strip()]
    return units or ([text.strip()] if text and text.strip() else [])


def build_language_embedding_stream(
    texts: str | Sequence[str],
    *,
    input_dim: int,
    window_size: int = 24,
    stride: int = 12,
) -> "torch.Tensor":
    """Convenience function for producing Z³ language embedding streams.

    ``texts`` may be a single transcript/document or a sequence of documents. A
    single string returns ``[steps, input_dim]``; a sequence returns
    ``[batch, steps, input_dim]``.
    """
    adapter = LanguageEmbeddingAdapter(
        LanguageAdapterConfig(input_dim=input_dim, window_size=window_size, stride=stride)
    )
    if isinstance(texts, str):
        return adapter.encode_text(texts)
    return adapter.encode_texts(list(texts))


def train_z3_on_language_window(
    model,
    optimizer,
    texts: str | Sequence[str],
    *,
    truncation_steps: int = 16,
    window_size: int = 24,
    stride: int = 12,
    commit_recurrent_state: bool = True,
    add_noise: bool = True,
) -> dict[str, float]:
    """Train ``Z3NeuralDynamics`` on a language stream using truncated BPTT."""
    stream = build_language_embedding_stream(
        texts,
        input_dim=model.config.input_dim,
        window_size=window_size,
        stride=stride,
    )
    return model.train_sequence_window(
        optimizer,
        stream,
        truncation_steps=truncation_steps,
        commit_recurrent_state=commit_recurrent_state,
        add_noise=add_noise,
    )
