"""
Z³ corpus ingestion stream
==========================

This module is the live bridge between corpus text and the trainable Z³ neural
runtime. It is intentionally separate from corpus discovery plugins: plugins
supply raw language observations, while this ingestor buffers those observations
and trains ``Z3NeuralDynamics`` through the existing language adapter.

The implementation is optional-dependency safe. If PyTorch is unavailable, the
mesh still runs, but the ingestor reports itself as unavailable instead of
pretending that neural corpus training is happening.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

try:  # pragma: no cover - depends on deployment image.
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # Prefer package imports when tests import from repository root.
    from core.z3_language_training import train_z3_on_language_window
    from core.z3_neural_dynamics import Z3NeuralConfig, Z3NeuralDynamics
except Exception:  # pragma: no cover - fallback for core-on-path runtime imports.
    try:
        from z3_language_training import train_z3_on_language_window  # type: ignore
        from z3_neural_dynamics import Z3NeuralConfig, Z3NeuralDynamics  # type: ignore
    except Exception as exc:  # pragma: no cover
        train_z3_on_language_window = None  # type: ignore[assignment]
        Z3NeuralConfig = None  # type: ignore[assignment]
        Z3NeuralDynamics = None  # type: ignore[assignment]
        _IMPORT_ERROR = exc
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class Z3CorpusIngestionConfig:
    """Runtime configuration for neural corpus ingestion."""

    enabled: bool = True
    batch_size: int = 8
    min_words: int = 24
    max_buffer_texts: int = 256
    max_train_batches_per_flush: int = 1
    window_size: int = 24
    stride: int = 12
    truncation_steps: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_path: str = "outputs/z3_neural_language.pt"
    checkpoint_every_steps: int = 25
    mode: str = "balanced"

    @classmethod
    def from_env(cls) -> "Z3CorpusIngestionConfig":
        disabled = os.getenv("DISABLE_Z3_CORPUS_INGESTION", "").lower() in ("1", "true", "yes")
        return cls(
            enabled=not disabled,
            batch_size=int(os.getenv("Z3_CORPUS_TRAIN_BATCH_SIZE", os.getenv("LANGUAGE_TRAINING_BATCH_SIZE", "8"))),
            min_words=int(os.getenv("Z3_CORPUS_MIN_WORDS", "24")),
            max_buffer_texts=int(os.getenv("Z3_CORPUS_BUFFER_TEXTS", "256")),
            max_train_batches_per_flush=int(os.getenv("Z3_CORPUS_MAX_TRAIN_BATCHES_PER_FLUSH", "1")),
            window_size=int(os.getenv("Z3_CORPUS_WINDOW_SIZE", "24")),
            stride=int(os.getenv("Z3_CORPUS_STRIDE", "12")),
            truncation_steps=int(os.getenv("Z3_CORPUS_TRUNCATION_STEPS", "16")),
            learning_rate=float(os.getenv("Z3_CORPUS_LEARNING_RATE", "0.001")),
            weight_decay=float(os.getenv("Z3_CORPUS_WEIGHT_DECAY", "0.0001")),
            checkpoint_path=os.getenv("Z3_CORPUS_CHECKPOINT_PATH", "outputs/z3_neural_language.pt"),
            checkpoint_every_steps=int(os.getenv("Z3_CORPUS_CHECKPOINT_EVERY_STEPS", "25")),
            mode=os.getenv("Z3_CORPUS_NEURAL_MODE", "balanced").strip().lower() or "balanced",
        )


class Z3CorpusNeuralIngestor:
    """Buffer corpus text and apply real Z³ neural training updates."""

    def __init__(self, config: Optional[Z3CorpusIngestionConfig] = None) -> None:
        self.config = config or Z3CorpusIngestionConfig.from_env()
        self._lock = threading.RLock()
        self._buffer: Deque[str] = deque(maxlen=max(1, self.config.max_buffer_texts))
        self._last_metrics: Dict[str, float] = {}
        self._last_error: str = ""
        self._trained_steps = 0
        self._texts_seen = 0
        self._last_train_time = 0.0
        self.model = None
        self.optimizer = None

        if self.config.enabled:
            self._initialize_runtime()

    @property
    def available(self) -> bool:
        """Return true only when a real trainable neural runtime exists."""
        return self.config.enabled and self.model is not None and self.optimizer is not None

    def _initialize_runtime(self) -> None:
        if torch is None:
            self._last_error = f"PyTorch unavailable: {_TORCH_IMPORT_ERROR}"
            return
        if _IMPORT_ERROR is not None or Z3NeuralDynamics is None or Z3NeuralConfig is None:
            self._last_error = f"Z3 neural imports unavailable: {_IMPORT_ERROR}"
            return

        checkpoint = Path(self.config.checkpoint_path)
        try:
            if checkpoint.exists():
                self.model = Z3NeuralDynamics.load_checkpoint(checkpoint, map_location="cpu")
            else:
                if self.config.mode == "internal_coherence":
                    neural_config = Z3NeuralConfig.internal_coherence()
                elif self.config.mode == "predictive_runtime":
                    neural_config = Z3NeuralConfig.predictive_runtime()
                else:
                    neural_config = Z3NeuralConfig.balanced()
                self.model = Z3NeuralDynamics(neural_config)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard.
            self.model = None
            self.optimizer = None
            self._last_error = f"Z3 corpus neural initialization failed: {exc}"

    def observe_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Accept one corpus text sample and train if enough buffered text exists."""
        if not self.config.enabled:
            return self.snapshot()
        text = (text or "").strip()
        if len(text.split()) < self.config.min_words:
            return self.snapshot()
        with self._lock:
            self._buffer.append(text)
            self._texts_seen += 1
        return self.flush(force=False)

    def observe_texts(self, texts: Iterable[str]) -> Dict[str, Any]:
        """Accept multiple corpus samples and train if the configured batch is ready."""
        for text in texts:
            self.observe_text(text)
        return self.flush(force=False)

    def flush(self, *, force: bool = False) -> Dict[str, Any]:
        """Run one or more training updates from the buffered corpus text."""
        if not self.available:
            return self.snapshot()
        trained_now = 0
        with self._lock:
            while len(self._buffer) >= self.config.batch_size or (force and self._buffer):
                if trained_now >= self.config.max_train_batches_per_flush:
                    break
                batch = []
                while self._buffer and len(batch) < self.config.batch_size:
                    batch.append(self._buffer.popleft())
                if not batch:
                    break
                try:
                    metrics = train_z3_on_language_window(
                        self.model,
                        self.optimizer,
                        batch,
                        truncation_steps=self.config.truncation_steps,
                        window_size=self.config.window_size,
                        stride=self.config.stride,
                        commit_recurrent_state=True,
                        add_noise=True,
                    )
                    self._last_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                    self._trained_steps += 1
                    self._last_train_time = time.time()
                    self._last_error = ""
                    trained_now += 1
                    if self.config.checkpoint_every_steps > 0 and self._trained_steps % self.config.checkpoint_every_steps == 0:
                        self.save_checkpoint()
                except Exception as exc:
                    # Put the batch back at the front so no corpus signal is silently lost.
                    for item in reversed(batch):
                        self._buffer.appendleft(item)
                    self._last_error = f"Z3 corpus training failed: {exc}"
                    break
        return self.snapshot()

    def save_checkpoint(self) -> bool:
        """Persist the neural corpus state if a model is available."""
        if not self.available:
            return False
        try:
            path = Path(self.config.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_checkpoint(path)
            return True
        except Exception as exc:  # pragma: no cover
            self._last_error = f"Z3 corpus checkpoint failed: {exc}"
            return False

    def snapshot(self) -> Dict[str, Any]:
        """Expose ingestion state for runtime diagnostics and the public cache."""
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "available": self.available,
                "buffer_size": len(self._buffer),
                "texts_seen": self._texts_seen,
                "trained_steps": self._trained_steps,
                "last_train_time": self._last_train_time,
                "last_error": self._last_error,
                "checkpoint_path": self.config.checkpoint_path,
                "batch_size": self.config.batch_size,
                "window_size": self.config.window_size,
                "stride": self.config.stride,
                "truncation_steps": self.config.truncation_steps,
                "last_metrics": dict(self._last_metrics),
            }
