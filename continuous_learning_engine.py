"""
Continuous Learning Engine
Online learning from data streams with pattern mining and auto-adaptation
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import pickle

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Track learning performance"""
    accuracy: float = 0.5
    samples_processed: int = 0
    patterns_discovered: int = 0
    adaptations: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        return {
            'accuracy': self.accuracy,
            'samples_processed': self.samples_processed,
            'patterns_discovered': self.patterns_discovered,
            'adaptations': self.adaptations,
            'last_update': self.last_update.isoformat()
        }


@dataclass
class Pattern:
    """A discovered pattern in the data"""
    pattern_id: str
    centroid: np.ndarray
    examples: List[Dict[str, Any]]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    hit_count: int = 0

    def to_dict(self):
        return {
            'pattern_id': self.pattern_id,
            'confidence': self.confidence,
            'example_count': len(self.examples),
            'hit_count': self.hit_count,
            'created_at': self.created_at.isoformat()
        }


class ContinuousLearningEngine:
    """
    Online learning engine that continuously learns from data streams.
    Features:
    - Incremental learning (no batch retraining)
    - Pattern mining and discovery
    - Auto-adaptation to distribution shifts
    - Short-term and long-term memory
    """

    def __init__(
        self,
        feature_dim: int = 50,
        learning_rate: float = 0.01,
        pattern_mining: bool = True,
        auto_adapt: bool = True,
        memory_size: int = 1000
    ):
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.pattern_mining = pattern_mining
        self.auto_adapt = auto_adapt

        # Online model weights
        self.weights = np.random.randn(feature_dim) * 0.01
        self.bias = 0.0

        # Feature mapping
        self.feature_names: List[str] = []
        self.feature_index: Dict[str, int] = {}

        # Memory
        self.short_term_memory: deque = deque(maxlen=memory_size)
        self.long_term_patterns: Dict[str, Pattern] = {}
        self.pattern_counter = 0

        # Metrics
        self.metrics = LearningMetrics()

        # Prediction history for accuracy tracking
        self.prediction_history: deque = deque(maxlen=100)

        # Distribution tracking for drift detection
        self.feature_means: Dict[str, float] = defaultdict(float)
        self.feature_vars: Dict[str, float] = defaultdict(lambda: 1.0)
        self.sample_count = 0

        logger.info(
            f"ContinuousLearningEngine initialized: "
            f"dim={feature_dim}, lr={learning_rate}, "
            f"pattern_mining={pattern_mining}, auto_adapt={auto_adapt}"
        )

    def _encode_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Encode observation dict into feature vector"""
        vector = np.zeros(self.feature_dim)

        for key, value in observation.items():
            if not isinstance(value, (int, float, bool)):
                continue

            if key not in self.feature_index:
                if len(self.feature_names) < self.feature_dim:
                    idx = len(self.feature_names)
                    self.feature_names.append(key)
                    self.feature_index[key] = idx
                else:
                    continue

            idx = self.feature_index[key]
            vector[idx] = float(value)

        return vector

    def _update_distribution(self, observation: Dict[str, Any]):
        """Track feature distributions for drift detection"""
        self.sample_count += 1
        alpha = 1.0 / self.sample_count

        for key, value in observation.items():
            if isinstance(value, (int, float)):
                old_mean = self.feature_means[key]
                self.feature_means[key] = old_mean + alpha * (value - old_mean)
                self.feature_vars[key] = (
                    self.feature_vars[key] + alpha * ((value - old_mean) ** 2 - self.feature_vars[key])
                )

    def process_observation(
        self,
        observation: Dict[str, Any],
        outcome: Optional[float] = None,
        feedback: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a single observation through the learning pipeline.
        Returns prediction and any discovered patterns.
        """
        # Encode
        x = self._encode_observation(observation)

        # Predict
        prediction = float(np.dot(self.weights[:len(x)], x[:len(self.weights)]) + self.bias)
        prediction = 1.0 / (1.0 + np.exp(-np.clip(prediction, -10, 10)))  # sigmoid

        result = {
            'prediction': prediction,
            'pattern_id': None,
            'is_novel': False
        }

        # Learn from outcome if provided
        if outcome is not None:
            error = outcome - prediction
            # SGD update
            grad = error * x[:len(self.weights)]
            self.weights[:len(x)] += self.learning_rate * grad
            self.bias += self.learning_rate * error

            # Track accuracy
            correct = (prediction > 0.5) == (outcome > 0.5)
            self.prediction_history.append(1.0 if correct else 0.0)
            if len(self.prediction_history) > 0:
                self.metrics.accuracy = sum(self.prediction_history) / len(self.prediction_history)

        # Store in memory
        self.short_term_memory.append(observation)
        self.metrics.samples_processed += 1

        # Update distribution tracking
        self._update_distribution(observation)

        # Pattern mining
        if self.pattern_mining and self.metrics.samples_processed % 10 == 0:
            pattern = self._mine_patterns(observation, x)
            if pattern:
                result['pattern_id'] = pattern.pattern_id
                result['is_novel'] = True

        # Auto-adaptation (drift detection)
        if self.auto_adapt and self.metrics.samples_processed % 50 == 0:
            self._check_and_adapt()

        self.metrics.last_update = datetime.now()
        return result

    def _mine_patterns(self, observation: Dict[str, Any], vector: np.ndarray) -> Optional[Pattern]:
        """Mine for patterns in recent observations"""
        # Check if observation matches existing pattern
        best_match = None
        best_similarity = 0.0

        for pattern in self.long_term_patterns.values():
            similarity = self._cosine_similarity(vector, pattern.centroid)
            if similarity > best_similarity and similarity > 0.8:
                best_similarity = similarity
                best_match = pattern

        if best_match:
            # Update existing pattern
            best_match.hit_count += 1
            best_match.confidence = min(1.0, best_match.confidence + 0.01)
            alpha = 1.0 / best_match.hit_count
            best_match.centroid = best_match.centroid * (1 - alpha) + vector * alpha
            if len(best_match.examples) < 20:
                best_match.examples.append(observation)
            return best_match
        else:
            # Create new pattern if we have enough recent data
            if len(self.short_term_memory) >= 5:
                pattern_id = f"pattern_{self.pattern_counter}"
                self.pattern_counter += 1

                pattern = Pattern(
                    pattern_id=pattern_id,
                    centroid=vector.copy(),
                    examples=[observation],
                    confidence=0.3,
                    hit_count=1
                )

                self.long_term_patterns[pattern_id] = pattern
                self.metrics.patterns_discovered += 1
                logger.debug(f"Discovered new pattern: {pattern_id}")
                return pattern

        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _check_and_adapt(self):
        """Check for distribution drift and adapt"""
        if len(self.short_term_memory) < 20:
            return

        recent = list(self.short_term_memory)[-20:]
        recent_means = defaultdict(float)

        for obs in recent:
            for key, value in obs.items():
                if isinstance(value, (int, float)):
                    recent_means[key] += value / len(recent)

        # Check for significant drift
        drift_detected = False
        for key, recent_mean in recent_means.items():
            if key in self.feature_means:
                global_mean = self.feature_means[key]
                global_std = max(np.sqrt(self.feature_vars[key]), 1e-6)
                z_score = abs(recent_mean - global_mean) / global_std

                if z_score > 3.0:  # Significant drift
                    drift_detected = True
                    break

        if drift_detected:
            # Increase learning rate temporarily
            self.learning_rate = min(self.learning_rate * 1.5, 0.1)
            self.metrics.adaptations += 1
            logger.info(f"Distribution drift detected. Adapted learning rate to {self.learning_rate:.4f}")
        else:
            # Decay learning rate back
            self.learning_rate = max(self.learning_rate * 0.99, 0.001)

    def get_insights(self) -> Dict[str, Any]:
        """Get learning engine insights"""
        return {
            'metrics': self.metrics.to_dict(),
            'total_patterns': len(self.long_term_patterns),
            'memory_size': len(self.short_term_memory),
            'feature_count': len(self.feature_names),
            'learning_rate': self.learning_rate,
            'top_patterns': [
                p.to_dict() for p in sorted(
                    self.long_term_patterns.values(),
                    key=lambda p: p.hit_count,
                    reverse=True
                )[:5]
            ]
        }

    def save_state(self, filepath: str):
        """Save engine state to disk"""
        try:
            state = {
                'weights': self.weights,
                'bias': self.bias,
                'feature_names': self.feature_names,
                'feature_index': self.feature_index,
                'metrics': self.metrics.to_dict(),
                'pattern_counter': self.pattern_counter,
                'learning_rate': self.learning_rate,
                'sample_count': self.sample_count
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Learning state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    def load_state(self, filepath: str):
        """Load engine state from disk"""
        try:
            import os
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)
                self.weights = state['weights']
                self.bias = state['bias']
                self.feature_names = state['feature_names']
                self.feature_index = state['feature_index']
                self.pattern_counter = state.get('pattern_counter', 0)
                self.learning_rate = state.get('learning_rate', self.learning_rate)
                self.sample_count = state.get('sample_count', 0)
                logger.info(f"Learning state loaded from {filepath}")
        except Exception as e:
            logger.warning(f"Could not load learning state: {e}")
