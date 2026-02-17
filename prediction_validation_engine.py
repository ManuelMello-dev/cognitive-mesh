"""
Prediction & Validation Engine
================================
The missing piece: the mesh must predict, then validate against reality.

This engine:
1. Tracks price history per symbol
2. Makes directional predictions (up/down/stable) based on learned rules + patterns
3. Validates predictions against actual outcomes on the NEXT observation
4. Updates rule confidence based on predictive accuracy (not support count)
5. Tracks accuracy over rolling time windows
6. Feeds accuracy back into PHI calculation

This is what makes the system ACTUALLY LEARN — not just accumulate.
"""

import logging
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class Prediction:
    """A single prediction made by the mesh"""
    prediction_id: str
    symbol: str
    domain: str
    direction: Direction
    confidence: float          # How confident the mesh is (0-1)
    basis: str                 # What rule/pattern generated this
    predicted_at: float        # timestamp
    predicted_price: float     # price at time of prediction
    target_price: Optional[float] = None   # what we expect
    actual_price: Optional[float] = None   # what actually happened
    actual_direction: Optional[Direction] = None
    validated: bool = False
    correct: Optional[bool] = None
    validation_time: Optional[float] = None


@dataclass
class SymbolHistory:
    """Price history and prediction state for a single symbol"""
    symbol: str
    domain: str
    prices: deque = field(default_factory=lambda: deque(maxlen=200))
    volumes: deque = field(default_factory=lambda: deque(maxlen=200))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=200))
    pending_prediction: Optional[Prediction] = None
    total_predictions: int = 0
    correct_predictions: int = 0
    recent_accuracy: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def accuracy(self) -> float:
        if not self.recent_accuracy:
            return 0.5  # neutral baseline
        return sum(self.recent_accuracy) / len(self.recent_accuracy)

    @property
    def price_trend(self) -> Optional[Direction]:
        """Calculate short-term price trend from last 5 observations"""
        if len(self.prices) < 3:
            return None
        recent = list(self.prices)[-5:]
        if len(recent) < 2:
            return None
        change = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        if change > 0.005:
            return Direction.UP
        elif change < -0.005:
            return Direction.DOWN
        return Direction.STABLE

    @property
    def volatility(self) -> float:
        """Calculate recent price volatility"""
        if len(self.prices) < 3:
            return 0.0
        recent = list(self.prices)[-20:]
        if len(recent) < 2:
            return 0.0
        returns = [(recent[i] - recent[i-1]) / max(abs(recent[i-1]), 1e-8)
                    for i in range(1, len(recent))]
        if not returns:
            return 0.0
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    @property
    def momentum(self) -> float:
        """Calculate price momentum (-1 to 1)"""
        if len(self.prices) < 5:
            return 0.0
        recent = list(self.prices)[-10:]
        if len(recent) < 2:
            return 0.0
        change = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        return max(-1.0, min(1.0, change * 10))  # scale to -1..1


@dataclass
class RulePerformance:
    """Track how well a specific rule predicts"""
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
    Makes predictions and validates them against reality.
    This is the feedback loop that makes the mesh actually learn.
    """

    def __init__(self):
        # Per-symbol tracking
        self.symbols: Dict[str, SymbolHistory] = {}

        # All predictions (for audit trail)
        self.all_predictions: deque = deque(maxlen=5000)
        self.prediction_counter = 0

        # Rule performance tracking
        self.rule_performance: Dict[str, RulePerformance] = defaultdict(
            lambda: RulePerformance(rule_id="unknown")
        )

        # Global accuracy tracking over time windows
        self.accuracy_history: deque = deque(maxlen=500)  # (timestamp, accuracy)
        self.phi_history: deque = deque(maxlen=500)        # (timestamp, phi)
        self.sigma_history: deque = deque(maxlen=500)      # (timestamp, sigma)

        # Aggregate stats
        self.total_predictions = 0
        self.total_correct = 0
        self.total_validated = 0

        logger.info("PredictionValidationEngine initialized")

    def record_observation(self, observation: Dict[str, Any], domain: str) -> Optional[Dict[str, Any]]:
        """
        Record a new observation. If there's a pending prediction for this symbol,
        validate it. Then make a new prediction.

        Returns validation result if a prediction was validated, else None.
        """
        symbol = observation.get('symbol', '')
        price = observation.get('price')
        volume = observation.get('volume', 0)

        if not symbol or price is None:
            return None

        try:
            price = float(price)
        except (ValueError, TypeError):
            return None

        # Initialize symbol history if new
        if symbol not in self.symbols:
            self.symbols[symbol] = SymbolHistory(symbol=symbol, domain=domain)

        history = self.symbols[symbol]

        # STEP 1: Validate pending prediction (if any)
        validation_result = None
        if history.pending_prediction and not history.pending_prediction.validated:
            validation_result = self._validate_prediction(history.pending_prediction, price)

        # STEP 2: Record the observation
        history.prices.append(price)
        history.volumes.append(volume)
        history.timestamps.append(time.time())

        # STEP 3: Make a new prediction for the NEXT observation
        new_prediction = self._make_prediction(history)
        if new_prediction:
            history.pending_prediction = new_prediction
            self.all_predictions.append(new_prediction)

        return validation_result

    def _validate_prediction(self, prediction: Prediction, actual_price: float) -> Dict[str, Any]:
        """Validate a prediction against reality"""
        prediction.actual_price = actual_price
        prediction.validated = True
        prediction.validation_time = time.time()

        # Determine actual direction
        if actual_price > prediction.predicted_price * 1.002:
            prediction.actual_direction = Direction.UP
        elif actual_price < prediction.predicted_price * 0.998:
            prediction.actual_direction = Direction.DOWN
        else:
            prediction.actual_direction = Direction.STABLE

        # Check if prediction was correct
        prediction.correct = (prediction.direction == prediction.actual_direction)

        # Update symbol accuracy
        symbol = prediction.symbol
        if symbol in self.symbols:
            history = self.symbols[symbol]
            history.total_predictions += 1
            if prediction.correct:
                history.correct_predictions += 1
            history.recent_accuracy.append(1.0 if prediction.correct else 0.0)

        # Update rule performance
        if prediction.basis:
            perf = self.rule_performance[prediction.basis]
            perf.rule_id = prediction.basis
            perf.predictions_made += 1
            if prediction.correct:
                perf.correct_predictions += 1
            perf.recent_accuracy.append(1.0 if prediction.correct else 0.0)

        # Update global stats
        self.total_predictions += 1
        self.total_validated += 1
        if prediction.correct:
            self.total_correct += 1

        # Record accuracy snapshot
        global_accuracy = self.total_correct / max(self.total_validated, 1)
        self.accuracy_history.append((time.time(), global_accuracy))

        return {
            "symbol": symbol,
            "predicted": prediction.direction.value,
            "actual": prediction.actual_direction.value,
            "correct": prediction.correct,
            "confidence": prediction.confidence,
            "basis": prediction.basis,
            "predicted_price": prediction.predicted_price,
            "actual_price": actual_price,
            "price_change_pct": (actual_price - prediction.predicted_price) / max(abs(prediction.predicted_price), 1e-8) * 100,
            "symbol_accuracy": self.symbols[symbol].accuracy if symbol in self.symbols else 0.5,
            "global_accuracy": global_accuracy,
        }

    def _make_prediction(self, history: SymbolHistory) -> Optional[Prediction]:
        """
        Make a prediction for the next observation based on:
        1. Price momentum (short-term trend)
        2. Volatility (uncertainty)
        3. Pattern matching (learned patterns)
        4. Mean reversion (if extended)
        """
        if len(history.prices) < 2:
            return None

        prices = list(history.prices)
        current_price = prices[-1]

        # --- Signal 1: Momentum ---
        momentum = history.momentum
        momentum_signal = Direction.UP if momentum > 0.05 else (
            Direction.DOWN if momentum < -0.05 else Direction.STABLE
        )

        # --- Signal 2: Short-term trend ---
        trend = history.price_trend
        if trend is None:
            trend = Direction.STABLE

        # --- Signal 3: Mean reversion ---
        if len(prices) >= 20:
            sma_20 = sum(prices[-20:]) / 20
            deviation = (current_price - sma_20) / max(abs(sma_20), 1e-8)
            if deviation > 0.03:
                mean_rev_signal = Direction.DOWN  # extended above mean
            elif deviation < -0.03:
                mean_rev_signal = Direction.UP    # extended below mean
            else:
                mean_rev_signal = Direction.STABLE
        else:
            mean_rev_signal = Direction.STABLE

        # --- Signal 4: Volume confirmation ---
        if len(history.volumes) >= 5:
            recent_vol = list(history.volumes)[-5:]
            avg_vol = sum(recent_vol) / len(recent_vol)
            vol_spike = recent_vol[-1] > avg_vol * 1.5 if avg_vol > 0 else False
        else:
            vol_spike = False

        # --- Combine signals with weights ---
        # Momentum and trend are primary; mean reversion is secondary
        signal_scores = {Direction.UP: 0.0, Direction.DOWN: 0.0, Direction.STABLE: 0.0}

        # Momentum (weight: 0.4)
        signal_scores[momentum_signal] += 0.4

        # Trend (weight: 0.3)
        signal_scores[trend] += 0.3

        # Mean reversion (weight: 0.2)
        signal_scores[mean_rev_signal] += 0.2

        # Volume confirmation (weight: 0.1) — amplifies the dominant signal
        if vol_spike:
            dominant = max(signal_scores, key=signal_scores.get)
            signal_scores[dominant] += 0.1
        else:
            signal_scores[Direction.STABLE] += 0.1

        # Pick the strongest signal
        predicted_direction = max(signal_scores, key=signal_scores.get)
        raw_confidence = signal_scores[predicted_direction]

        # Adjust confidence by historical accuracy for this symbol
        symbol_accuracy = history.accuracy
        adjusted_confidence = raw_confidence * 0.6 + symbol_accuracy * 0.4

        # Determine basis (what drove this prediction)
        if signal_scores[momentum_signal] >= signal_scores[trend]:
            basis = f"momentum_{momentum:.3f}"
        elif signal_scores[mean_rev_signal] > signal_scores[trend]:
            basis = f"mean_reversion_{deviation:.3f}" if len(prices) >= 20 else "trend"
        else:
            basis = f"trend_{trend.value}"

        self.prediction_counter += 1
        prediction = Prediction(
            prediction_id=f"pred_{self.prediction_counter}",
            symbol=history.symbol,
            domain=history.domain,
            direction=predicted_direction,
            confidence=round(adjusted_confidence, 4),
            basis=basis,
            predicted_at=time.time(),
            predicted_price=current_price,
        )

        return prediction

    # ──────────────────────────────────────────
    # PHI / SIGMA Calculation (Real)
    # ──────────────────────────────────────────

    def calculate_real_phi(self) -> float:
        """
        PHI (Global Coherence) — based on REAL metrics:
        1. Prediction accuracy (40%) — are we actually right?
        2. Concept stability (30%) — are predictions consistent?
        3. Cross-symbol agreement (30%) — do different symbols tell the same story?
        """
        if self.total_validated == 0:
            return 0.5  # neutral baseline until we have data

        # Component 1: Global prediction accuracy
        accuracy = self.total_correct / max(self.total_validated, 1)

        # Component 2: Prediction consistency (low variance = high stability)
        if len(self.accuracy_history) >= 5:
            recent_acc = [a for _, a in list(self.accuracy_history)[-20:]]
            mean_acc = sum(recent_acc) / len(recent_acc)
            variance = sum((a - mean_acc) ** 2 for a in recent_acc) / len(recent_acc)
            stability = max(0.0, 1.0 - math.sqrt(variance) * 3)
        else:
            stability = 0.5

        # Component 3: Cross-symbol agreement
        if len(self.symbols) >= 2:
            directions = []
            for sym in self.symbols.values():
                if sym.pending_prediction:
                    directions.append(sym.pending_prediction.direction)
            if len(directions) >= 2:
                # Count most common direction
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

        # Record for trend tracking
        self.phi_history.append((time.time(), phi))

        return phi

    def calculate_real_sigma(self) -> float:
        """
        SIGMA (Noise Level) — based on REAL metrics:
        1. Prediction error rate (40%) — how often are we wrong?
        2. Price volatility across symbols (30%)
        3. Accuracy variance (30%) — is accuracy stable or erratic?
        """
        if self.total_validated == 0:
            return 0.5  # neutral baseline

        # Component 1: Error rate
        error_rate = 1.0 - (self.total_correct / max(self.total_validated, 1))

        # Component 2: Average volatility across symbols
        volatilities = [s.volatility for s in self.symbols.values() if len(s.prices) >= 5]
        avg_volatility = sum(volatilities) / max(len(volatilities), 1) if volatilities else 0.0
        # Normalize volatility (typical market vol is 0.01-0.05)
        vol_noise = min(1.0, avg_volatility * 20)

        # Component 3: Accuracy variance
        if len(self.accuracy_history) >= 5:
            recent_acc = [a for _, a in list(self.accuracy_history)[-20:]]
            mean_acc = sum(recent_acc) / len(recent_acc)
            variance = sum((a - mean_acc) ** 2 for a in recent_acc) / len(recent_acc)
            acc_variance = min(1.0, math.sqrt(variance) * 4)
        else:
            acc_variance = 0.5

        sigma = error_rate * 0.4 + vol_noise * 0.3 + acc_variance * 0.3
        sigma = max(0.0, min(1.0, sigma))

        # Record for trend tracking
        self.sigma_history.append((time.time(), sigma))

        return sigma

    # ──────────────────────────────────────────
    # Rule Confidence Feedback
    # ──────────────────────────────────────────

    def get_rule_confidence_adjustments(self) -> Dict[str, float]:
        """
        Return confidence adjustments for rules based on their predictive accuracy.
        Rules that predict well get boosted; rules that fail get penalized.
        """
        adjustments = {}
        for rule_id, perf in self.rule_performance.items():
            if perf.predictions_made >= 5:
                # Accuracy-based confidence
                adjustments[rule_id] = perf.accuracy
        return adjustments

    # ──────────────────────────────────────────
    # Insights & Reporting
    # ──────────────────────────────────────────

    def get_insights(self) -> Dict[str, Any]:
        """Get prediction engine insights"""
        # Per-symbol accuracy
        symbol_accuracies = {}
        for sym, history in self.symbols.items():
            symbol_accuracies[sym] = {
                "accuracy": round(history.accuracy, 4),
                "total_predictions": history.total_predictions,
                "correct": history.correct_predictions,
                "trend": history.price_trend.value if history.price_trend else "unknown",
                "momentum": round(history.momentum, 4),
                "volatility": round(history.volatility, 6),
                "price_count": len(history.prices),
            }

        # Accuracy trend (last 20 data points)
        accuracy_trend = []
        if self.accuracy_history:
            for ts, acc in list(self.accuracy_history)[-20:]:
                accuracy_trend.append({"time": ts, "accuracy": round(acc, 4)})

        # PHI trend
        phi_trend = []
        if self.phi_history:
            for ts, phi in list(self.phi_history)[-20:]:
                phi_trend.append({"time": ts, "phi": round(phi, 4)})

        # SIGMA trend
        sigma_trend = []
        if self.sigma_history:
            for ts, sigma in list(self.sigma_history)[-20:]:
                sigma_trend.append({"time": ts, "sigma": round(sigma, 4)})

        # Top performing rules
        top_rules = sorted(
            [(rid, perf) for rid, perf in self.rule_performance.items() if perf.predictions_made >= 3],
            key=lambda x: x[1].accuracy,
            reverse=True
        )[:10]

        # Recent predictions
        recent_preds = []
        for pred in list(self.all_predictions)[-10:]:
            recent_preds.append({
                "symbol": pred.symbol,
                "direction": pred.direction.value,
                "confidence": round(pred.confidence, 4),
                "correct": pred.correct,
                "validated": pred.validated,
                "basis": pred.basis,
            })

        return {
            "total_predictions": self.total_predictions,
            "total_validated": self.total_validated,
            "total_correct": self.total_correct,
            "global_accuracy": round(self.total_correct / max(self.total_validated, 1), 4),
            "symbols_tracked": len(self.symbols),
            "symbol_accuracies": symbol_accuracies,
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
