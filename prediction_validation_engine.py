"""
Universal State-Change Prediction & Validation Engine
======================================================
Predicts and validates directional changes in ANY continuous variable,
not just financial prices.  The engine is domain-agnostic: it tracks
"streams" identified by (entity_id, domain) pairs and predicts whether
the primary observable value will increase, decrease, or remain stable
over the next N observations.

Design:
  1. Tracks value history per stream (any scalar: price, temperature,
     CPU load, humidity, sentiment score, etc.)
  2. Makes directional predictions over an N-tick HORIZON
  3. Validates predictions after N ticks have elapsed
  4. Uses ADAPTIVE dead zones based on observed per-stream volatility
  5. Updates rule confidence based on predictive accuracy
  6. Tracks accuracy over rolling time windows
  7. Feeds accuracy back into PHI / SIGMA calculation

The engine accepts observations with a mandatory `value` field and an
optional `entity_id` field (defaults to the domain string).  Legacy
observations that contain `price` and `symbol` are transparently mapped
to `value` and `entity_id` for backward compatibility.
"""

import logging
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
from contracts import PredictionOutput

logger = logging.getLogger(__name__)


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"   # was CRITICAL — renamed to be domain-neutral


# Backward-compatible alias so existing code that imports Direction.CRITICAL still works
Direction.CRITICAL = Direction.STABLE  # type: ignore[attr-defined]


def _extract_entity_and_value(observation: Dict[str, Any], domain: str):
    """
    Extract a generic (entity_id, value, secondary_value) tuple from any
    observation dict.  Supports:
      - Generic:   {'entity_id': ..., 'value': ...}
      - Financial: {'symbol': ..., 'price': ..., 'volume': ...}
      - IoT:       {'sensor_id': ..., 'reading': ...}
      - Any other: first numeric field is treated as value
    """
    # Generic schema (preferred)
    if 'value' in observation:
        entity_id = observation.get('entity_id') or observation.get('id') or domain
        value = observation['value']
        secondary = observation.get('secondary_value', observation.get('volume', 0))
        return str(entity_id), float(value), float(secondary)

    # Financial legacy schema
    if 'price' in observation:
        entity_id = observation.get('symbol') or domain
        value = observation['price']
        secondary = observation.get('volume', 0)
        return str(entity_id), float(value), float(secondary)

    # IoT / sensor schema
    if 'reading' in observation:
        entity_id = observation.get('sensor_id') or observation.get('entity_id') or domain
        value = observation['reading']
        secondary = observation.get('secondary', 0)
        return str(entity_id), float(value), float(secondary)

    # Fallback: first numeric value found
    for k, v in observation.items():
        if k not in ('timestamp', 'time', 'ts') and isinstance(v, (int, float)):
            entity_id = observation.get('entity_id') or observation.get('id') or domain
            return str(entity_id), float(v), 0.0

    return None, None, None


@dataclass
class Prediction:
    """A single prediction made by the mesh about any stream"""
    prediction_id: str
    entity_id: str          # generic entity (was: symbol)
    domain: str
    direction: Direction
    confidence: float
    basis: str
    predicted_at: float
    predicted_value: float  # value at time of prediction (was: predicted_price)
    horizon: int = 5
    ticks_remaining: int = 5
    dead_zone_pct: float = 0.0
    target_value: Optional[float] = None
    actual_value: Optional[float] = None
    actual_direction: Optional[Direction] = None
    validated: bool = False
    max_post_signal_move_pct: float = 0.0
    correct: Optional[bool] = None
    validation_time: Optional[float] = None

    # ── Legacy property aliases for backward compatibility ──
    @property
    def symbol(self) -> str:
        return self.entity_id

    @property
    def predicted_price(self) -> float:
        return self.predicted_value

    @property
    def actual_price(self) -> Optional[float]:
        return self.actual_value

    @actual_price.setter
    def actual_price(self, v):
        self.actual_value = v


@dataclass
class StreamHistory:
    """Value history and prediction state for a single observable stream"""
    entity_id: str
    domain: str
    values: deque = field(default_factory=lambda: deque(maxlen=200))
    secondary_values: deque = field(default_factory=lambda: deque(maxlen=200))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=200))
    pending_prediction: Optional[Prediction] = None
    total_predictions: int = 0
    correct_predictions: int = 0
    recent_accuracy: deque = field(default_factory=lambda: deque(maxlen=50))

    # ── Legacy aliases ──
    @property
    def symbol(self) -> str:
        return self.entity_id

    @property
    def prices(self) -> deque:
        return self.values

    @property
    def volumes(self) -> deque:
        return self.secondary_values

    @property
    def accuracy(self) -> float:
        if not self.recent_accuracy:
            return 0.5
        return sum(self.recent_accuracy) / len(self.recent_accuracy)

    @property
    def value_trend(self) -> Optional[Direction]:
        """Short-term directional trend from last 5 observations"""
        if len(self.values) < 3:
            return None
        recent = list(self.values)[-5:]
        if len(recent) < 2:
            return None
        change = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        vol = self.tick_volatility
        threshold = max(vol * 2, 0.001)
        if change > threshold:
            return Direction.UP
        elif change < -threshold:
            return Direction.DOWN
        return Direction.STABLE

    # Backward-compat alias
    @property
    def price_trend(self) -> Optional[Direction]:
        return self.value_trend

    @property
    def tick_volatility(self) -> float:
        """Per-tick volatility: std-dev of tick-to-tick returns"""
        if len(self.values) < 5:
            return 0.005
        recent = list(self.values)[-30:]
        if len(recent) < 3:
            return 0.005
        returns = [
            (recent[i] - recent[i - 1]) / max(abs(recent[i - 1]), 1e-8)
            for i in range(1, len(recent))
        ]
        if not returns:
            return 0.005
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return max(math.sqrt(variance), 0.0001)

    @property
    def volatility(self) -> float:
        return self.tick_volatility

    @property
    def momentum(self) -> float:
        """Value momentum as a z-score (-3 to +3)"""
        if len(self.values) < 5:
            return 0.0
        recent = list(self.values)[-10:]
        if len(recent) < 2:
            return 0.0
        raw_change = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        vol = self.tick_volatility
        n = len(recent) - 1
        expected_noise = vol * math.sqrt(n) if n > 0 else vol
        if expected_noise < 1e-10:
            return 0.0
        z_score = raw_change / expected_noise
        return max(-3.0, min(3.0, z_score))


# Backward-compat alias
SymbolHistory = StreamHistory


@dataclass
class RulePerformance:
    """Track how well a specific rule predicts state changes"""
    rule_id: str
    predictions_made: int = 0
    correct_predictions: int = 0
    recent_accuracy: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def accuracy(self) -> float:
        if not self.recent_accuracy:
            return 0.5
        return sum(self.recent_accuracy) / len(self.recent_accuracy)


class PredictionValidationEngine:
    """
    Universal state-change predictor.
    Makes predictions about ANY continuous variable and validates them
    against reality to create a closed feedback loop.

    Key design: predictions and validations operate on the SAME timescale.
    A prediction made over a 5-tick momentum window is validated after 5 ticks.
    The dead zone is adaptive: 1× per-tick sigma × sqrt(horizon).
    """

    SHORT_HORIZON = 5
    MEDIUM_HORIZON = 10
    LONG_HORIZON = 20
    CRITICAL_MOVE_THRESHOLD_PCT = 3.0
    CRITICAL_MOVE_VALIDATION_WINDOW_HOURS = 24

    def __init__(self):
        # Per-stream tracking (keyed by entity_id)
        self.streams: Dict[str, StreamHistory] = {}

        # Backward-compat alias
        self.symbols = self.streams

        self.all_predictions: deque = deque(maxlen=5000)
        self.prediction_counter = 0
        self.rule_performance: Dict[str, RulePerformance] = defaultdict(
            lambda: RulePerformance(rule_id="unknown")
        )
        self.accuracy_history: deque = deque(maxlen=500)
        self.phi_history: deque = deque(maxlen=500)
        self.sigma_history: deque = deque(maxlen=500)
        self.total_predictions = 0
        self.total_correct = 0
        self.total_validated = 0

        logger.info("PredictionValidationEngine initialized (universal state-change mode)")

    def observe(self, observation: Dict[str, Any], domain: str) -> Optional[Dict[str, Any]]:
        """Alias for record_observation"""
        return self.record_observation(observation, domain)

    def record_observation(self, observation: Dict[str, Any], domain: str) -> Optional[Dict[str, Any]]:
        """
        Record a new observation for any stream.
        Validates pending predictions and makes new ones.
        """
        entity_id, value, secondary = _extract_entity_and_value(observation, domain)
        if entity_id is None or value is None:
            return None
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None

        if entity_id not in self.streams:
            self.streams[entity_id] = StreamHistory(entity_id=entity_id, domain=domain)

        history = self.streams[entity_id]
        history.values.append(value)
        history.secondary_values.append(secondary)
        history.timestamps.append(time.time())

        validation_result = None
        if history.pending_prediction and not history.pending_prediction.validated:
            pred = history.pending_prediction
            pred.ticks_remaining -= 1
            if pred.ticks_remaining <= 0:
                validation_result = self._validate_prediction(pred, value)

        if history.pending_prediction is None or history.pending_prediction.validated:
            new_prediction = self._make_prediction(history)
            if new_prediction:
                history.pending_prediction = new_prediction
                self.all_predictions.append(new_prediction)

        return validation_result

    def _validate_prediction(self, prediction: Prediction, actual_value: float) -> Dict[str, Any]:
        """Validate a prediction against reality after the horizon has elapsed."""
        prediction.actual_value = actual_value
        prediction.validated = True
        prediction.validation_time = time.time()

        value_change_pct = (actual_value - prediction.predicted_value) / max(
            abs(prediction.predicted_value), 1e-8
        )

        # Max post-signal move for STABLE/CRITICAL predictions
        if prediction.direction == Direction.STABLE:
            history = self.streams.get(prediction.entity_id)
            if history:
                relevant_values = []
                window_end = prediction.predicted_at + (self.CRITICAL_MOVE_VALIDATION_WINDOW_HOURS * 3600)
                for i in range(len(history.timestamps) - 1, -1, -1):
                    ts = history.timestamps[i]
                    if prediction.predicted_at <= ts <= window_end:
                        relevant_values.append(history.values[i])
                    elif ts < prediction.predicted_at:
                        break
                relevant_values.reverse()
                if relevant_values:
                    min_v = min(relevant_values)
                    max_v = max(relevant_values)
                    move_low = abs((min_v - prediction.predicted_value) / max(abs(prediction.predicted_value), 1e-8))
                    move_high = abs((max_v - prediction.predicted_value) / max(abs(prediction.predicted_value), 1e-8))
                    prediction.max_post_signal_move_pct = max(move_low, move_high)

        dz = prediction.dead_zone_pct
        if value_change_pct > dz:
            prediction.actual_direction = Direction.UP
        elif value_change_pct < -dz:
            prediction.actual_direction = Direction.DOWN
        else:
            prediction.actual_direction = Direction.STABLE

        partial_score = 0.0
        if prediction.direction == prediction.actual_direction:
            prediction.correct = True
            partial_score = 1.0
        elif prediction.direction == Direction.STABLE:
            significant_move_threshold = self.CRITICAL_MOVE_THRESHOLD_PCT / 100.0
            if abs(value_change_pct) >= significant_move_threshold:
                prediction.correct = True
                partial_score = 1.0
            else:
                prediction.correct = False
                partial_score = 0.0
        elif (prediction.direction == Direction.UP and value_change_pct > 0) or \
             (prediction.direction == Direction.DOWN and value_change_pct < 0):
            partial_score = 0.5
        else:
            partial_score = 0.0

        entity_id = prediction.entity_id
        if entity_id in self.streams:
            history = self.streams[entity_id]
            history.total_predictions += 1
            if prediction.correct:
                history.correct_predictions += 1
            history.recent_accuracy.append(partial_score)

        if prediction.basis:
            perf = self.rule_performance[prediction.basis]
            perf.rule_id = prediction.basis
            perf.predictions_made += 1
            if prediction.correct:
                perf.correct_predictions += 1
            perf.recent_accuracy.append(partial_score)

        self.total_predictions += 1
        self.total_validated += 1
        if prediction.correct:
            self.total_correct += 1

        global_accuracy = self.total_correct / max(self.total_validated, 1)
        self.accuracy_history.append((time.time(), global_accuracy))

        return {
            "entity_id": entity_id,
            "domain": prediction.domain,
            "predicted": prediction.direction.value,
            "actual": prediction.actual_direction.value,
            "correct": prediction.correct,
            "partial_score": round(partial_score, 2),
            "confidence": prediction.confidence,
            "basis": prediction.basis,
            "predicted_value": prediction.predicted_value,
            "actual_value": actual_value,
            "value_change_pct": round(value_change_pct * 100, 4),
            "dead_zone_pct": round(prediction.dead_zone_pct * 100, 4),
            "horizon": prediction.horizon,
            "stream_accuracy": self.streams[entity_id].accuracy if entity_id in self.streams else 0.5,
            "global_accuracy": global_accuracy,
        }

    def _make_prediction(self, history: StreamHistory) -> Optional[Prediction]:
        """
        Make a prediction for the next N ticks based on:
          1. Momentum z-score
          2. Short-term trend
          3. Mean reversion
          4. Secondary-value confirmation (volume, attention, etc.)
        """
        if len(history.values) < 5:
            return None

        values = list(history.values)
        current_value = values[-1]
        tick_vol = history.tick_volatility
        horizon = self.SHORT_HORIZON
        dead_zone = tick_vol * math.sqrt(horizon)

        # Signal 1: Momentum z-score
        momentum_z = history.momentum
        if momentum_z > 1.0:
            momentum_signal = Direction.UP
        elif momentum_z < -1.0:
            momentum_signal = Direction.DOWN
        else:
            momentum_signal = Direction.STABLE

        # Signal 2: Short-term trend
        trend = history.value_trend or Direction.STABLE

        # Signal 3: Mean reversion
        deviation = 0.0
        mean_rev_signal = Direction.STABLE
        if len(values) >= 20:
            sma_20 = sum(values[-20:]) / 20
            deviation = (current_value - sma_20) / max(abs(sma_20), 1e-8)
            dev_threshold = tick_vol * math.sqrt(20) * 2
            if deviation > dev_threshold:
                mean_rev_signal = Direction.DOWN
            elif deviation < -dev_threshold:
                mean_rev_signal = Direction.UP

        # Signal 4: Secondary-value confirmation (volume / attention spike)
        secondary_spike = False
        if len(history.secondary_values) >= 5:
            recent_sec = list(history.secondary_values)[-5:]
            avg_sec = sum(recent_sec) / len(recent_sec)
            secondary_spike = recent_sec[-1] > avg_sec * 1.5 if avg_sec > 0 else False

        signal_scores = {Direction.UP: 0.0, Direction.DOWN: 0.0, Direction.STABLE: 0.0}
        signal_scores[momentum_signal] += 0.35
        if abs(momentum_z) > 1.5:
            signal_scores[momentum_signal] += 0.05
        signal_scores[trend] += 0.25
        signal_scores[mean_rev_signal] += 0.25
        if secondary_spike:
            dominant = max(signal_scores, key=signal_scores.get)
            signal_scores[dominant] += 0.15
        else:
            signal_scores[Direction.STABLE] += 0.10

        predicted_direction = max(signal_scores, key=signal_scores.get)
        raw_confidence = signal_scores[predicted_direction]
        stream_accuracy = history.accuracy
        adjusted_confidence = raw_confidence * 0.6 + stream_accuracy * 0.4

        if abs(momentum_z) > 1.0 and signal_scores[momentum_signal] >= signal_scores.get(trend, 0):
            basis = f"momentum_z{momentum_z:+.2f}"
        elif abs(deviation) > 0 and signal_scores[mean_rev_signal] > signal_scores.get(trend, 0):
            basis = f"mean_reversion_{deviation:+.4f}"
        else:
            basis = f"trend_{trend.value}"

        self.prediction_counter += 1
        return Prediction(
            prediction_id=f"pred_{self.prediction_counter}",
            entity_id=history.entity_id,
            domain=history.domain,
            direction=predicted_direction,
            confidence=round(adjusted_confidence, 4),
            basis=basis,
            predicted_at=time.time(),
            predicted_value=current_value,
            horizon=horizon,
            ticks_remaining=horizon,
            dead_zone_pct=dead_zone,
        )

    # ──────────────────────────────────────────
    # PHI / SIGMA Calculation
    # ──────────────────────────────────────────

    def calculate_real_phi(self) -> float:
        """
        PHI (Global Coherence):
          40% prediction accuracy
          30% prediction consistency (low variance)
          30% cross-stream agreement
        """
        if self.total_validated == 0:
            return 0.5

        accuracy = self.total_correct / max(self.total_validated, 1)

        if len(self.accuracy_history) >= 5:
            recent_acc = [a for _, a in list(self.accuracy_history)[-20:]]
            mean_acc = sum(recent_acc) / len(recent_acc)
            variance = sum((a - mean_acc) ** 2 for a in recent_acc) / len(recent_acc)
            stability = max(0.0, 1.0 - math.sqrt(variance) * 3)
        else:
            stability = 0.5

        if len(self.streams) >= 2:
            directions = [
                s.pending_prediction.direction
                for s in self.streams.values()
                if s.pending_prediction and not s.pending_prediction.validated
            ]
            if len(directions) >= 2:
                from collections import Counter
                counts = Counter(directions)
                most_common_count = counts.most_common(1)[0][1]
                agreement = most_common_count / len(directions)
            else:
                agreement = 0.5
        else:
            agreement = 0.5

        phi = accuracy * 0.4 + stability * 0.3 + agreement * 0.3
        phi = max(0.0, min(1.0, phi))
        self.phi_history.append((time.time(), phi))
        return phi

    def calculate_real_sigma(self) -> float:
        """
        SIGMA (Noise Level):
          40% prediction error rate
          30% average stream volatility
          30% accuracy variance
        """
        if self.total_validated == 0:
            return 0.5

        error_rate = 1.0 - (self.total_correct / max(self.total_validated, 1))

        volatilities = [s.volatility for s in self.streams.values() if len(s.values) >= 5]
        avg_volatility = sum(volatilities) / max(len(volatilities), 1) if volatilities else 0.0
        vol_noise = min(1.0, avg_volatility * 50)

        if len(self.accuracy_history) >= 5:
            recent_acc = [a for _, a in list(self.accuracy_history)[-20:]]
            mean_acc = sum(recent_acc) / len(recent_acc)
            variance = sum((a - mean_acc) ** 2 for a in recent_acc) / len(recent_acc)
            acc_variance = min(1.0, math.sqrt(variance) * 4)
        else:
            acc_variance = 0.5

        sigma = error_rate * 0.4 + vol_noise * 0.3 + acc_variance * 0.3
        sigma = max(0.0, min(1.0, sigma))
        self.sigma_history.append((time.time(), sigma))
        return sigma

    # ──────────────────────────────────────────
    # Rule Confidence Feedback
    # ──────────────────────────────────────────

    def get_rule_confidence_adjustments(self) -> Dict[str, float]:
        """Return confidence adjustments for rules based on predictive accuracy."""
        return {
            rule_id: perf.accuracy
            for rule_id, perf in self.rule_performance.items()
            if perf.predictions_made >= 5
        }

    # ──────────────────────────────────────────
    # Insights & Reporting
    # ──────────────────────────────────────────

    def get_insights(self) -> Dict[str, Any]:
        """Get prediction engine insights"""
        stream_accuracies = {}
        for eid, history in self.streams.items():
            stream_accuracies[eid] = {
                "accuracy": round(history.accuracy, 4),
                "total_predictions": history.total_predictions,
                "correct": history.correct_predictions,
                "trend": history.value_trend.value if history.value_trend else "unknown",
                "momentum_z": round(history.momentum, 4),
                "tick_volatility": round(history.tick_volatility, 6),
                "observation_count": len(history.values),
                "pending_horizon": (
                    history.pending_prediction.ticks_remaining
                    if history.pending_prediction and not history.pending_prediction.validated
                    else None
                ),
            }

        accuracy_trend = [
            {"time": ts, "accuracy": round(acc, 4)}
            for ts, acc in list(self.accuracy_history)[-20:]
        ]
        phi_trend = [
            {"time": ts, "phi": round(phi, 4)}
            for ts, phi in list(self.phi_history)[-20:]
        ]
        sigma_trend = [
            {"time": ts, "sigma": round(sigma, 4)}
            for ts, sigma in list(self.sigma_history)[-20:]
        ]

        top_rules = sorted(
            [(rid, perf) for rid, perf in self.rule_performance.items() if perf.predictions_made >= 3],
            key=lambda x: x[1].accuracy,
            reverse=True
        )[:10]

        recent_preds = []
        for pred in list(self.all_predictions)[-10:]:
            recent_preds.append({
                "entity_id": pred.entity_id,
                "domain": pred.domain,
                "direction": pred.direction.value,
                "confidence": round(pred.confidence, 4),
                "correct": pred.correct,
                "validated": pred.validated,
                "basis": pred.basis,
                "horizon": pred.horizon,
                "dead_zone_pct": round(pred.dead_zone_pct * 100, 4),
            })

        return {
            "total_predictions": self.total_predictions,
            "total_validated": self.total_validated,
            "total_correct": self.total_correct,
            "global_accuracy": round(self.total_correct / max(self.total_validated, 1), 4),
            "streams_tracked": len(self.streams),
            "symbols_tracked": len(self.streams),  # backward-compat alias
            "stream_accuracies": stream_accuracies,
            "symbol_accuracies": stream_accuracies,  # backward-compat alias
            "accuracy_trend": accuracy_trend,
            "phi_trend": phi_trend,
            "sigma_trend": sigma_trend,
            "top_rules": [
                {"rule_id": rid, "accuracy": round(perf.accuracy, 4), "predictions": perf.predictions_made}
                for rid, perf in top_rules
            ],
            "recent_predictions": recent_preds,
            "phi": round(self.calculate_real_phi(), 4),
            "sigma": round(self.calculate_real_sigma(), 4),
        }

    def get_active_predictions_as_outputs(self) -> List[PredictionOutput]:
        """Return all pending (unvalidated) predictions as structured PredictionOutput contracts."""
        outputs = []
        for history in self.symbols.values():
            pred = history.pending_prediction
            if pred and not pred.validated:
                outputs.append(PredictionOutput(
                    prediction_id=pred.prediction_id,
                    symbol=pred.symbol,
                    domain=pred.domain,
                    direction=pred.direction.value,
                    confidence=pred.confidence,
                    basis=pred.basis,
                    horizon=pred.horizon,
                    ticks_remaining=pred.ticks_remaining,
                    predicted_price=pred.predicted_price,
                    dead_zone_pct=pred.dead_zone_pct,
                ))
        return outputs
