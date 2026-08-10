"""Canonical predictive and benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from .config import LABELS, QualityGatesConfig, RELEASE_METRICS_SCHEMA


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    loss: float
    recall_by_class: tuple[float, ...]
    precision_by_class: tuple[float, ...]
    f1_by_class: tuple[float, ...]
    support_by_class: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    partition: str = "PublicTest"
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RELEASE_METRICS_SCHEMA,
            "partition": self.partition,
            "dtype": self.dtype,
            "labels": list(LABELS),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "loss": self.loss,
            "precision_by_class": dict(zip(LABELS, self.precision_by_class)),
            "recall_by_class": dict(zip(LABELS, self.recall_by_class)),
            "f1_by_class": dict(zip(LABELS, self.f1_by_class)),
            "support_by_class": dict(zip(LABELS, self.support_by_class)),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }

    @property
    def margin(self) -> float:
        return self.predictive_margin(QualityGatesConfig())

    def predictive_margin(self, gates: QualityGatesConfig) -> float:
        return min(
            self.accuracy - gates.accuracy,
            self.macro_f1 - gates.macro_f1,
            *(value - gates.min_recall for value in self.recall_by_class),
        )


@dataclass(frozen=True)
class GateResult:
    passed: bool
    observed: Mapping[str, Any]
    limits: Mapping[str, Any]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "observed": dict(self.observed),
            "limits": dict(self.limits),
            "reasons": list(self.reasons),
        }


def metrics_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray | None = None,
    *,
    loss: float | None = None,
    partition: str = "PublicTest",
    dtype: str = "float32",
) -> Metrics:
    actual = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("prediction arrays must have equal lengths")
    labels = list(range(4))
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    if loss is None:
        if probabilities is not None:
            loss = float(log_loss(actual, np.asarray(probabilities), labels=labels))
        else:
            loss = float(np.mean(actual != predicted)) if len(actual) else 0.0
    return Metrics(
        float(report["accuracy"]),
        float(report["macro avg"]["f1-score"]),
        float(loss),
        tuple(float(report[str(i)]["recall"]) for i in labels),
        tuple(float(report[str(i)]["precision"]) for i in labels),
        tuple(float(report[str(i)]["f1-score"]) for i in labels),
        tuple(int(report[str(i)]["support"]) for i in labels),
        tuple(tuple(int(value) for value in row) for row in matrix.tolist()),
        partition,
        dtype,
    )


def evaluate_predictive_gate(metrics: Metrics, gates: QualityGatesConfig) -> GateResult:
    reasons: list[str] = []
    if metrics.accuracy < gates.accuracy:
        reasons.append("accuracy")
    if metrics.macro_f1 < gates.macro_f1:
        reasons.append("macro_f1")
    if any(value < gates.min_recall for value in metrics.recall_by_class):
        reasons.append("per_class_recall")
    return GateResult(
        not reasons,
        {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "min_recall": min(metrics.recall_by_class, default=0.0),
            "margin": metrics.predictive_margin(gates),
        },
        {"accuracy": gates.accuracy, "macro_f1": gates.macro_f1, "min_recall": gates.min_recall},
        tuple(reasons),
    )


def order_candidates(candidates: Sequence[Mapping[str, Any]], *, limit: int = 2) -> tuple[Mapping[str, Any], ...]:
    by_margin = sorted(candidates, key=lambda value: (-float(value["median_margin"]), str(value["recipe_id"])))
    if len(by_margin) > 1 and abs(float(by_margin[0]["median_margin"]) - float(by_margin[1]["median_margin"])) < 0.005:
        by_margin = sorted(
            candidates,
            key=lambda value: (
                float(value.get("median_p95_ms", 0.0)),
                int(value.get("median_rss_increment_bytes", 0)),
                int(value.get("median_onnx_bytes", 0)),
                str(value["recipe_id"]),
            ),
        )
    return tuple(by_margin[:limit])
