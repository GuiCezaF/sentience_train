"""Canonical metrics, gates, and deterministic report writers.

Training, candidate selection, and release use this module so that every
artifact describes the same labels, class order, and confusion-matrix shape.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from .config import LABELS, QualityGatesConfig


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

    @property
    def macro_precision(self) -> float:
        return float(mean(self.precision_by_class)) if self.precision_by_class else 0.0

    @property
    def macro_recall(self) -> float:
        return float(mean(self.recall_by_class)) if self.recall_by_class else 0.0

    @property
    def min_recall(self) -> float:
        return min(self.recall_by_class, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "loss": self.loss,
            "precision_by_class": dict(zip(LABELS, self.precision_by_class)),
            "recall_by_class": dict(zip(LABELS, self.recall_by_class)),
            "f1_by_class": dict(zip(LABELS, self.f1_by_class)),
            "support_by_class": dict(zip(LABELS, self.support_by_class)),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "labels": list(LABELS),
            "class_names": list(LABELS),
            "partition": self.partition,
            "dtype": self.dtype,
        }


def metrics_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: np.ndarray | None = None,
    *,
    loss: float | None = None,
    dtype: str = "float32",
    partition: str = "PublicTest",
) -> Metrics:
    """Build complete metrics using the canonical four-class order."""
    actual = np.asarray(y_true, dtype=np.int64).ravel()
    predicted = np.asarray(y_pred, dtype=np.int64).ravel()
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same number of samples")
    labels = list(range(len(LABELS)))
    matrix = confusion_matrix(actual, predicted, labels=labels)
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    if loss is None:
        if y_score is None:
            loss = float(np.mean(actual != predicted)) if len(actual) else 0.0
        else:
            probabilities = np.asarray(y_score, dtype=np.float64)
            if probabilities.ndim != 2 or probabilities.shape != (
                len(actual),
                len(labels),
            ):
                raise ValueError("y_score must have shape (samples, 4)")
            probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-15)
            loss = float(log_loss(actual, probabilities, labels=labels))
    return Metrics(
        accuracy=float(report["accuracy"]),
        macro_f1=float(report["macro avg"]["f1-score"]),
        loss=float(loss),
        recall_by_class=tuple(float(report[str(index)]["recall"]) for index in labels),
        precision_by_class=tuple(float(report[str(index)]["precision"]) for index in labels),
        f1_by_class=tuple(float(report[str(index)]["f1-score"]) for index in labels),
        support_by_class=tuple(int(report[str(index)]["support"]) for index in labels),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix.tolist()),
        partition=partition,
        dtype=dtype,
    )


@dataclass(frozen=True)
class ResourceMetrics:
    tflite_bytes: int
    arena_with_margin_bytes: int
    macs: int
    arena_total_bytes: int | None = None
    arena_head_bytes: int | None = None
    arena_temporary_bytes: int | None = None
    arena_tail_bytes: int | None = None
    input_contract: Mapping[str, Any] = field(default_factory=dict)
    output_contract: Mapping[str, Any] = field(default_factory=dict)
    operators: tuple[Mapping[str, Any], ...] = ()
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tflite_bytes": self.tflite_bytes,
            "arena_with_margin_bytes": self.arena_with_margin_bytes,
            "arena_total_bytes": self.arena_total_bytes,
            "arena_head_bytes": self.arena_head_bytes,
            "arena_temporary_bytes": self.arena_temporary_bytes,
            "arena_tail_bytes": self.arena_tail_bytes,
            "macs": self.macs,
            "input_contract": dict(self.input_contract),
            "output_contract": dict(self.output_contract),
            "operators": [dict(item) for item in self.operators],
            "sha256": self.sha256,
        }


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


def evaluate_gates(
    metrics: Metrics,
    gates: QualityGatesConfig,
    *,
    reference: Metrics | None = None,
    resources: ResourceMetrics | None = None,
) -> GateResult:
    """Apply inclusive predictive, parity, and host-resource gates."""
    delta = None if reference is None else reference.accuracy - metrics.accuracy
    observed: dict[str, Any] = {
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "recall_by_class": dict(zip(LABELS, metrics.recall_by_class)),
        "accuracy_delta": delta,
    }
    limits: dict[str, Any] = {
        "accuracy": {"minimum": gates.accuracy},
        "macro_f1": {"minimum": gates.macro_f1},
        "recall_by_class": {"minimum": gates.recall},
        "accuracy_delta": {"maximum": gates.max_accuracy_delta},
    }
    reasons: list[str] = []
    if metrics.accuracy < gates.accuracy:
        reasons.append(f"accuracy {metrics.accuracy:.6f} < {gates.accuracy:.6f}")
    if metrics.macro_f1 < gates.macro_f1:
        reasons.append(f"macro_f1 {metrics.macro_f1:.6f} < {gates.macro_f1:.6f}")
    for name, value in zip(LABELS, metrics.recall_by_class):
        if value < gates.recall:
            reasons.append(f"recall[{name}] {value:.6f} < {gates.recall:.6f}")
    if delta is not None and delta - gates.max_accuracy_delta > 1e-12:
        reasons.append(f"accuracy_delta {delta:.6f} > {gates.max_accuracy_delta:.6f}")
    if resources is not None:
        observed["resources"] = resources.to_dict()
        limits["resources"] = {
            "tflite_bytes": {"maximum": gates.tflite_bytes},
            "arena_with_margin_bytes": {"maximum": gates.arena_with_margin_bytes},
            "macs": {"maximum": gates.macs},
        }
        if resources.tflite_bytes > gates.tflite_bytes:
            reasons.append(f"tflite_bytes {resources.tflite_bytes} > {gates.tflite_bytes}")
        if resources.arena_with_margin_bytes > gates.arena_with_margin_bytes:
            reasons.append(
                f"arena_with_margin_bytes {resources.arena_with_margin_bytes} > {gates.arena_with_margin_bytes}"
            )
        if resources.macs > gates.macs:
            reasons.append(f"macs {resources.macs} > {gates.macs}")
    return GateResult(not reasons, observed, limits, tuple(reasons))


def evaluate_resource_gates(
    resource: Mapping[str, Any],
    gates: QualityGatesConfig,
) -> GateResult:
    """Apply host resource limits while preserving pending hardware evidence."""
    observed: dict[str, Any] = {
        "tflite_bytes": resource.get("tflite_bytes", resource.get("model_bytes")),
        "arena_with_margin_bytes": resource.get("arena_with_margin_bytes"),
        "macs": resource.get("macs"),
        "arena_measurement": "host_estimate",
        "esp32_s3_arena": "pending_external",
        "esp32_s3_latency_p95": "pending_external",
    }
    limits: dict[str, Any] = {
        "tflite_bytes": {"maximum": gates.tflite_bytes},
        "arena_with_margin_bytes": {"maximum": gates.arena_with_margin_bytes},
        "macs": {"maximum": gates.macs},
    }
    reasons: list[str] = []
    for key, limit in (
        ("tflite_bytes", gates.tflite_bytes),
        ("arena_with_margin_bytes", gates.arena_with_margin_bytes),
        ("macs", gates.macs),
    ):
        value = observed[key]
        if not isinstance(value, int):
            reasons.append(f"{key} is missing")
        elif value > limit:
            reasons.append(f"{key} {value} > {limit}")
    return GateResult(not reasons, observed, limits, tuple(reasons))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_json(path: Path, metrics: Metrics) -> None:
    write_json(path, metrics.to_dict())


def confusion_document(
    metrics_by_model: Mapping[str, Metrics | None],
) -> dict[str, Any]:
    first = next((metrics for metrics in metrics_by_model.values() if metrics is not None), None)
    return {
        "labels": list(LABELS),
        "partition": None if first is None else first.partition,
        "matrices": {
            name: None if metrics is None else [list(row) for row in metrics.confusion_matrix]
            for name, metrics in metrics_by_model.items()
        },
    }


def write_confusion_reports(
    directory: Path,
    metrics_by_model: Mapping[str, Metrics | None],
    *,
    stem: str = "confusion-matrix",
) -> tuple[Path, Path, Path]:
    """Write named-matrix JSON, long CSV, and a byte-stable PNG."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    csv_path = directory / f"{stem}.csv"
    png_path = directory / f"{stem}.png"
    write_json(json_path, confusion_document(metrics_by_model))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "actual", "predicted", "count"))
        for name, metrics in metrics_by_model.items():
            matrix = metrics.confusion_matrix if metrics is not None else ((0,) * len(LABELS),) * len(LABELS)
            for actual, row in zip(LABELS, matrix):
                for predicted, count in zip(LABELS, row):
                    writer.writerow((name, actual, predicted, int(count)))
    _write_confusion_png(png_path, metrics_by_model)
    return json_path, csv_path, png_path


def _write_confusion_png(path: Path, metrics_by_model: Mapping[str, Metrics | None]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised only without export deps
        raise RuntimeError(f"Matplotlib/Agg is unavailable: {exc}") from exc
    items = tuple(metrics_by_model.items())
    figure, axes = plt.subplots(
        1,
        max(1, len(items)),
        figsize=(4.0 * max(1, len(items)), 3.6),
        dpi=120,
        squeeze=False,
    )
    for axis, (name, metrics) in zip(axes[0], items):
        matrix = np.asarray(metrics.confusion_matrix if metrics is not None else ((0,) * len(LABELS),) * len(LABELS))
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        for row in range(len(LABELS)):
            for column in range(len(LABELS)):
                axis.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                    color="black",
                )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Software": "sentience-train", "Date": None})
    plt.close(figure)
