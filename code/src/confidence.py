"""Confidence composition, calibration, and calibration diagnostics.

Confidence estimates the probability that the selected *action* is correct.  The
fallback is intentionally class-specific: notify=[.55,.94], digest=[.52,.90],
and mute=[.58,.96].  These bands reflect the different cost/precision profiles
of interrupts, deferrals, and high-precision safety/opt-out suppression.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class ConfidenceSignals:
    rule_strength: float
    model_probability_margin: float
    neighbor_similarity: float
    neighbor_agreement: float
    historical_evidence_quality: float
    context_completeness: float
    media_extraction_confidence: float
    component_conflict: float

    def clipped(self) -> "ConfidenceSignals":
        return ConfidenceSignals(**{key: float(np.clip(value, 0, 1)) for key, value in asdict(self).items()})


CLASS_BANDS = {"notify": (.55, .94), "digest": (.52, .90), "mute": (.58, .96)}


class ConfidenceCalibrator:
    """Compose signals and optionally map the result using held-out outcomes."""

    def __init__(self) -> None:
        self.model: IsotonicRegression | LogisticRegression | None = None
        self.method = "class-specific-bands"

    @staticmethod
    def raw_score(signals: ConfidenceSignals) -> float:
        s = signals.clipped()
        support = (
            .30 * s.rule_strength
            + .13 * s.model_probability_margin
            + .12 * s.neighbor_similarity
            + .12 * s.neighbor_agreement
            + .11 * s.historical_evidence_quality
            + .10 * s.context_completeness
            + .12 * s.media_extraction_confidence
        )
        # Conflict is a direct penalty; uncertain media also makes missing context
        # more consequential than either signal would be independently.
        support -= .22 * s.component_conflict
        support -= .08 * (1 - s.media_extraction_confidence) * (1 - s.context_completeness)
        return float(np.clip(support, 0, 1))

    def fit(self, signals: Sequence[ConfidenceSignals], correct: Sequence[bool]) -> "ConfidenceCalibrator":
        """Fit only from out-of-sample predictions supplied by the caller.

        Isotonic needs at least 80 examples and 20 examples of each outcome;
        sigmoid calibration needs 30 and 8 of each.  Smaller or one-sided sets
        retain the documented deterministic class bands.
        """
        x = np.asarray([self.raw_score(item) for item in signals], dtype=float)
        y = np.asarray(correct, dtype=int)
        positives, negatives = int(y.sum()), int(len(y) - y.sum())
        self.model = None
        self.method = "class-specific-bands"
        if len(y) >= 80 and min(positives, negatives) >= 20:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=.01, y_max=.99).fit(x, y)
            self.method = "isotonic"
        elif len(y) >= 30 and min(positives, negatives) >= 8:
            self.model = LogisticRegression(C=1.0, solver="lbfgs", random_state=0).fit(x.reshape(-1, 1), y)
            self.method = "sigmoid"
        return self

    def predict(self, action: str, signals: ConfidenceSignals) -> float:
        raw = self.raw_score(signals)
        if self.model is None:
            low, high = CLASS_BANDS[action]
            value = low + (high - low) * raw
        elif self.method == "isotonic":
            value = float(self.model.predict([raw])[0])
        else:
            value = float(self.model.predict_proba([[raw]])[0, 1])
        return round(float(np.clip(value, .01, .99)), 4)


def calibration_metrics(correct: Iterable[bool], confidence: Iterable[float], n_bins: int = 10) -> dict[str, object]:
    """Return Brier score, log loss, ECE, and equal-width reliability bins."""
    y = np.asarray(list(correct), dtype=int)
    p = np.clip(np.asarray(list(confidence), dtype=float), 1e-6, 1 - 1e-6)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        mask = (p >= lower) & ((p < upper) if index < n_bins - 1 else (p <= upper))
        if not mask.any():
            continue
        accuracy, mean_confidence, count = float(y[mask].mean()), float(p[mask].mean()), int(mask.sum())
        ece += count / len(y) * abs(accuracy - mean_confidence)
        bins.append({"lower": lower, "upper": upper, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "expected_calibration_error": float(ece),
        "reliability_bins": bins,
    }
