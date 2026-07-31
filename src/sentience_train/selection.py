"""Pure selection policy, metrics, gates, and immutable result types.

The orchestration engine uses this module for decisions so they can be tested
without a dataset, a GPU, or the LiteRT deployment toolchain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from .config import LABELS, QualityGatesConfig, WIDTHS


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
    def min_recall(self) -> float:
        return min(self.recall_by_class, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "loss": self.loss,
            "recall_by_class": dict(zip(LABELS, self.recall_by_class)),
            "precision_by_class": dict(zip(LABELS, self.precision_by_class)),
            "f1_by_class": dict(zip(LABELS, self.f1_by_class)),
            "support_by_class": dict(zip(LABELS, self.support_by_class)),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
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
    """Build complete selection metrics from one allowed partition."""
    actual = np.asarray(y_true, dtype=np.int64).ravel()
    predicted = np.asarray(y_pred, dtype=np.int64).ravel()
    labels = list(range(len(LABELS)))
    matrix = confusion_matrix(actual, predicted, labels=labels)
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    if loss is None:
        if y_score is None:
            # This path is useful for deterministic synthetic tests.  It is
            # intentionally a finite proxy and is never used for calibration.
            loss = float(np.mean(actual != predicted)) if len(actual) else 0.0
        else:
            probabilities = np.asarray(y_score, dtype=np.float64)
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
        return {"passed": self.passed, "observed": dict(self.observed), "limits": dict(self.limits), "reasons": list(self.reasons)}


def evaluate_gates(
    metrics: Metrics,
    gates: QualityGatesConfig,
    *,
    reference: Metrics | None = None,
    resources: ResourceMetrics | None = None,
) -> GateResult:
    """Apply inclusive quality/resource gates, recording every observation."""
    delta = None if reference is None else reference.accuracy - metrics.accuracy
    observed: dict[str, Any] = {
        "accuracy": metrics.accuracy,
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
            reasons.append(f"arena_with_margin_bytes {resources.arena_with_margin_bytes} > {gates.arena_with_margin_bytes}")
        if resources.macs > gates.macs:
            reasons.append(f"macs {resources.macs} > {gates.macs}")
    return GateResult(not reasons, observed, limits, tuple(reasons))


@dataclass(frozen=True)
class Aggregate:
    mean: float
    std: float
    minimum: float
    maximum: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "min": self.minimum, "max": self.maximum, "count": self.count}


def aggregate(values: Iterable[float]) -> Aggregate:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot aggregate an empty sequence")
    return Aggregate(mean(values), stdev(values) if len(values) > 1 else 0.0, min(values), max(values), len(values))


def aggregate_metrics(results: Sequence["SeedResult"]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot aggregate no seed results")
    int8 = [result for result in results if result.int8_metrics is not None]
    return {
        "accuracy": aggregate(result.int8_metrics.accuracy for result in int8).to_dict(),
        "macro_f1": aggregate(result.int8_metrics.macro_f1 for result in int8).to_dict(),
        "loss": aggregate(result.int8_metrics.loss for result in int8).to_dict(),
        "precision_by_class": {
            label: aggregate(result.int8_metrics.precision_by_class[index] for result in int8).to_dict()
            for index, label in enumerate(LABELS)
        },
        "f1_by_class": {
            label: aggregate(result.int8_metrics.f1_by_class[index] for result in int8).to_dict()
            for index, label in enumerate(LABELS)
        },
        "recall_by_class": {
            label: aggregate(result.int8_metrics.recall_by_class[index] for result in int8).to_dict()
            for index, label in enumerate(LABELS)
        },
        "fp32_accuracy": aggregate(result.fp32_metrics.accuracy for result in results).to_dict(),
        "fp32_macro_f1": aggregate(result.fp32_metrics.macro_f1 for result in results).to_dict(),
        "fp32_loss": aggregate(result.fp32_metrics.loss for result in results).to_dict(),
        "fp32_recall_by_class": {
            label: aggregate(result.fp32_metrics.recall_by_class[index] for result in results).to_dict()
            for index, label in enumerate(LABELS)
        },
        "accuracy_delta": aggregate(result.accuracy_delta for result in results if result.accuracy_delta is not None).to_dict(),
        "resources": {
            "tflite_bytes": aggregate(result.resources.tflite_bytes for result in results if result.resources is not None).to_dict(),
            "arena_with_margin_bytes": aggregate(result.resources.arena_with_margin_bytes for result in results if result.resources is not None).to_dict(),
            "macs": aggregate(result.resources.macs for result in results if result.resources is not None).to_dict(),
        },
    }


@dataclass(frozen=True)
class SeedResult:
    width: float
    seed: int
    fp32_metrics: Metrics
    fp32_gate: GateResult
    int8_metrics: Metrics | None = None
    int8_gate: GateResult | None = None
    resources: ResourceMetrics | None = None
    accuracy_delta: float | None = None
    route: str | None = None
    status: str = "pending"
    transition_reasons: tuple[str, ...] = ()
    checkpoint_fp32: str | None = None
    checkpoint_qat: str | None = None
    tflite: str | None = None
    history: str | None = None
    qat_history: str | None = None
    confusion_matrix_json: str | None = None
    confusion_matrix_csv: str | None = None
    best_epoch: int = 0

    @property
    def fp32(self) -> Metrics:
        return self.fp32_metrics

    @property
    def int8(self) -> Metrics | None:
        return self.int8_metrics

    @property
    def delta(self) -> float | None:
        return self.accuracy_delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "seed": self.seed,
            "fp32": {"metrics": self.fp32_metrics.to_dict(), "gates": self.fp32_gate.to_dict()},
            "int8": None if self.int8_metrics is None else {"metrics": self.int8_metrics.to_dict(), "gates": self.int8_gate.to_dict() if self.int8_gate else None},
            "resources": None if self.resources is None else self.resources.to_dict(),
            "accuracy_delta": self.accuracy_delta,
            "route": self.route,
            "status": self.status,
            "best_epoch": self.best_epoch,
            "transition_reasons": list(self.transition_reasons),
            "artifacts": {"checkpoint_fp32": self.checkpoint_fp32, "checkpoint_qat": self.checkpoint_qat, "tflite": self.tflite, "history": self.history, "qat_history": self.qat_history, "confusion_matrix_json": self.confusion_matrix_json, "confusion_matrix_csv": self.confusion_matrix_csv},
        }


@dataclass(frozen=True)
class WidthAggregate:
    width: float
    seeds: tuple[SeedResult, ...]
    aggregates: Mapping[str, Any]
    representative_seed: int | None
    representative: SeedResult | None
    gates: GateResult
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "status": self.status,
            "reason": self.reason,
            "representative_seed": self.representative_seed,
            "aggregates": dict(self.aggregates),
            "gates": self.gates.to_dict(),
            "seeds": [seed.to_dict() for seed in self.seeds],
        }


@dataclass(frozen=True)
class FrozenRun:
    run_id: str
    directory: Path
    width: float
    seed: int
    checkpoint: Path
    int8_artifact: Path
    route: str
    metrics: Metrics
    status: str = "frozen"
    best_epoch: int = 0

    @property
    def best_macro_f1(self) -> float:
        return self.metrics.macro_f1

    @property
    def best_loss(self) -> float:
        return self.metrics.loss

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "directory": str(self.directory), "status": self.status, "width": self.width, "seed": self.seed, "checkpoint": str(self.checkpoint), "int8_artifact": str(self.int8_artifact), "route": self.route, "metrics": self.metrics.to_dict()}


class SelectionFailed(RuntimeError):
    """Raised after an immutable run containing the complete diagnosis exists."""

    def __init__(self, directory: Path, reason: str = "all candidates failed selection") -> None:
        self.directory = directory
        self.diagnosis = directory / "candidate-comparison.json"
        self.reason = reason
        super().__init__(f"selection failed; diagnosis: {self.diagnosis}")


class EarlyStopping:
    """Macro-F1 early stopping with loss tie-break and checkpoint restoration."""

    def __init__(self, patience: int = 12, min_delta: float = 0.002) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.stale = 0

    def update(self, epoch: int, macro_f1: float, validation_loss: float) -> bool:
        score_improved = macro_f1 >= self.best_score + self.min_delta
        tied_better = macro_f1 == self.best_score and validation_loss < self.best_loss
        improved = self.best_epoch == 0 or score_improved or tied_better
        if improved:
            self.best_score = float(macro_f1)
            self.best_loss = float(validation_loss)
            self.best_epoch = int(epoch)
            self.stale = 0
        else:
            self.stale += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stale >= self.patience


def representative_seed(results: Sequence[SeedResult]) -> int:
    available = [result for result in results if result.int8_metrics is not None]
    if not available:
        raise ValueError("cannot choose representative without INT8 metrics")
    ordered = sorted(available, key=lambda item: (item.int8_metrics.macro_f1, item.seed))
    median_value = ordered[len(ordered) // 2].int8_metrics.macro_f1
    return min(result.seed for result in available if result.int8_metrics.macro_f1 == median_value)


def should_run_qat(
    *,
    ptq_conversion_failed: bool,
    accuracy_delta: float | None,
    max_accuracy_delta: float = 0.02,
) -> bool:
    """QAT has exactly two triggers: conversion failure or accuracy loss."""
    return ptq_conversion_failed or (accuracy_delta is not None and accuracy_delta > max_accuracy_delta)


def width_order(widths: Iterable[float] = WIDTHS) -> tuple[float, ...]:
    expected = tuple(float(width) for width in widths)
    return tuple(width for width in WIDTHS if width in expected)


def select_first_passing(widths: Sequence[WidthAggregate]) -> WidthAggregate | None:
    """Return the first passed width in canonical order, never the largest score."""
    by_width = {item.width: item for item in widths}
    for width in WIDTHS:
        candidate = by_width.get(width)
        if candidate is not None and candidate.status == "passed":
            return candidate
    return None


def dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
