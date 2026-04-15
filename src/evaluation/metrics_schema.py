from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def load_evaluation_metrics_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _dtype_block(doc: dict[str, Any], dtype_key: str = "float32") -> dict[str, Any]:
    return doc["test"][dtype_key]


def validate_evaluation_metrics(
    doc: dict[str, Any],
    *,
    dtype_key: str = "float32",
    expected_version: int = 6,
) -> list[str]:
    """
    Validates structure and internal consistency of the artifact (matrix x report x subgroups).
    Accumulates errors.
    """
    errors: list[str] = []

    if "version" not in doc:
        errors.append("missing required field: version")
    elif doc["version"] != expected_version:
        errors.append(
            f"expected version {expected_version}, got {doc['version']!r}"
        )

    if "test" not in doc:
        errors.append("missing required field: test")
        return errors

    test = doc["test"]
    if dtype_key not in test:
        errors.append(f"test is missing precision block {dtype_key!r}")
        return errors

    block = test[dtype_key]
    required = (
        "confusion_matrix",
        "classification_report",
        "loss",
        "support",
        "weighted_average",
        "subgroup_metrics",
        "class_names",
    )
    for key in required:
        if key not in block:
            errors.append(f"block {dtype_key} is missing required field: {key}")

    if errors:
        return errors

    cm = np.asarray(block["confusion_matrix"], dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        errors.append(
            f"confusion_matrix must be square 2D, got shape {cm.shape!r}"
        )
        return errors

    n = cm.shape[0]
    class_names = block["class_names"]
    if len(class_names) != n:
        errors.append(
            f"class_names ({len(class_names)}) does not match matrix order ({n})"
        )

    rep = block["classification_report"]
    for i in range(n):
        key = str(i)
        if key not in rep:
            errors.append(f"classification_report missing key {key!r}")
            continue
        row_sum = int(cm[i].sum())
        sup = rep[key].get("support")
        if sup != row_sum:
            errors.append(
                f"support for class {key} ({sup}) != matrix row sum ({row_sum})"
            )

    if "accuracy" in rep:
        acc_file = float(rep["accuracy"])
        acc_cm = float(accuracy_from_confusion_matrix(cm))
        if not np.isclose(acc_file, acc_cm, rtol=0, atol=1e-9):
            errors.append(
                f"accuracy in report ({acc_file}) differs from matrix ({acc_cm})"
            )

    total_support = int(cm.sum())
    if block["support"] != total_support:
        errors.append(
            f"global support ({block['support']}) != matrix sum ({total_support})"
        )

    wa = block["weighted_average"]
    wrep = rep.get("weighted avg", {})
    for metric in ("precision", "recall", "f1-score"):
        if metric not in wa or metric not in wrep:
            errors.append(f"weighted metric missing: {metric}")
        elif not np.isclose(float(wa[metric]), float(wrep[metric]), rtol=0, atol=1e-9):
            errors.append(
                f"weighted_average.{metric} ({wa[metric]}) != classification_report weighted avg ({wrep[metric]})"
            )

    sg = block["subgroup_metrics"]
    if "per_key" not in sg or "class_name" not in sg["per_key"]:
        errors.append("subgroup_metrics.per_key.class_name is missing")
    else:
        errors.extend(
            _validate_subgroup_rows(cm, class_names, sg["per_key"]["class_name"])
        )

    return errors


def _validate_subgroup_rows(
    cm: np.ndarray,
    class_names: list[str],
    class_name_block: dict[str, Any],
) -> list[str]:
    """Ensures each subgroup repeats only the corresponding row of the global matrix."""
    errors: list[str] = []
    per_group = class_name_block.get("per_group", {})
    for idx, name in enumerate(class_names):
        if name not in per_group:
            errors.append(f"subgroup_metrics missing group {name!r}")
            continue
        gcm = np.asarray(per_group[name]["confusion_matrix"], dtype=np.int64)
        if gcm.shape != cm.shape:
            errors.append(
                f"subgroup {name!r}: matrix {gcm.shape} != global {cm.shape}"
            )
            continue
        if not np.array_equal(gcm[idx], cm[idx]):
            errors.append(
                f"subgroup {name!r}: row {idx} differs from global matrix"
            )
        mask = np.ones(cm.shape[0], dtype=bool)
        mask[idx] = False
        if int(gcm[mask].sum()) != 0:
            errors.append(
                f"subgroup {name!r}: expected zeroed rows except for index {idx}"
            )
        sub_rep = per_group[name].get("classification_report", {})
        key = str(idx)
        if key in sub_rep and sub_rep[key].get("support") != int(cm[idx].sum()):
            errors.append(
                f"subgroup {name!r}: support in {key} does not match global row sum"
            )
    return errors


def accuracy_from_confusion_matrix(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.int64)
    return float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0


def labels_from_confusion_matrix(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cm = np.asarray(cm, dtype=np.int64)
    y_true: list[int] = []
    y_pred: list[int] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = int(cm[i, j])
            if c:
                y_true.extend([i] * c)
                y_pred.extend([j] * c)
    return np.array(y_true), np.array(y_pred)


def classification_report_matches_matrix(
    cm: np.ndarray,
    report: dict[str, Any],
    *,
    atol: float = 1e-6,
) -> list[str]:
    errors: list[str] = []
    y_true, y_pred = labels_from_confusion_matrix(cm)
    computed = classification_report(
        y_true, y_pred, labels=list(range(cm.shape[0])), output_dict=True, zero_division=0
    )

    for i in range(cm.shape[0]):
        key = str(i)
        for metric in ("precision", "recall", "f1-score"):
            a = float(report[key][metric])
            b = float(computed[key][metric])
            if not np.isclose(a, b, rtol=0, atol=atol):
                errors.append(
                    f"{key}.{metric}: file={a}, computed={b}"
                )

    for agg in ("macro avg", "weighted avg"):
        for metric in ("precision", "recall", "f1-score"):
            a = float(report[agg][metric])
            b = float(computed[agg][metric])
            if not np.isclose(a, b, rtol=0, atol=atol):
                errors.append(
                    f"{agg}.{metric}: file={a}, computed={b}"
                )

    if "accuracy" in report:
        a = float(report["accuracy"])
        b = float(computed["accuracy"])
        if not np.isclose(a, b, rtol=0, atol=atol):
            errors.append(f"accuracy: file={a}, computed={b}")

    return errors


def _subgroup_metrics_from_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Build `subgroup_metrics` in the shape expected by validation: one matrix per
    class with only that row copied from the global matrix and all other rows zeroed.
    """
    cm = np.asarray(cm, dtype=np.int64)
    per_group: dict[str, Any] = {}
    for idx, name in enumerate(class_names):
        gcm = np.zeros_like(cm)
        gcm[idx] = cm[idx]
        per_group[name] = {
            "confusion_matrix": gcm.astype(int).tolist(),
            "classification_report": {},
        }
    return {
        "facetted": None,
        "per_key": {"class_name": {"per_group": per_group}},
    }


def _renormalize_proba_rows(y_score: np.ndarray) -> np.ndarray:
    """
    Force each row to sum to 1 in float64.

    Avoids sklearn `UserWarning` in `log_loss` when probabilities arrive as float32
    (e.g. PyTorch `softmax`) and row sums drift slightly away from 1.0.
    """
    p = np.asarray(y_score, dtype=np.float64)
    denom = np.maximum(p.sum(axis=1, keepdims=True), 1e-15)
    return p / denom


def _classification_report_json_safe(rep: dict[str, Any]) -> dict[str, Any]:
    """Convert sklearn `classification_report` dict values to JSON-serializable scalars."""
    out: dict[str, Any] = {}
    for k, v in rep.items():
        if isinstance(v, dict):
            out[k] = _classification_report_json_safe(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def build_evaluation_metrics_document(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_score: np.ndarray,
    *,
    version: int = 6,
    dtype_key: str = "float32",
) -> dict[str, Any]:
    """
    Build a version-6 Sentience / Edge Impulse-style metrics document from labels,
    discrete predictions, and per-sample probability rows.

    Multiclass loss is `log_loss`; ROC AUC uses one-vs-rest when all classes are
    present in `y_true` (otherwise `roc_auc_score` may return NaN).
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    n_classes = len(class_names)
    y_score = _renormalize_proba_rows(y_score)
    if y_score.ndim != 2 or y_score.shape[1] != n_classes:
        raise ValueError(
            f"y_score must be (n_samples, {n_classes}), got {y_score.shape}"
        )
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=None,
        output_dict=True,
        zero_division=0,
    )
    rep = _classification_report_json_safe(rep)
    loss = float(log_loss(y_true, y_score, labels=labels))
    try:
        roc = float(
            roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                labels=labels,
            )
        )
    except ValueError:
        roc = float("nan")

    wa = rep["weighted avg"]
    block: dict[str, Any] = {
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": rep,
        "roc_auc": roc,
        "loss": loss,
        "support": int(cm.sum()),
        "weighted_average": {
            "precision": float(wa["precision"]),
            "recall": float(wa["recall"]),
            "f1-score": float(wa["f1-score"]),
        },
        "subgroup_metrics": _subgroup_metrics_from_confusion_matrix(cm, class_names),
        "class_names": list(class_names),
    }
    return {"version": version, "test": {dtype_key: block}}


def build_metrics_summary(doc: dict[str, Any], dtype_key: str = "float32") -> dict[str, Any]:
    block = _dtype_block(doc, dtype_key)
    cm = np.asarray(block["confusion_matrix"], dtype=np.int64)
    rep = block["classification_report"]
    return {
        "version": doc.get("version"),
        "dtype": dtype_key,
        "class_names": list(block["class_names"]),
        "support_total": int(block["support"]),
        "accuracy": float(rep.get("accuracy", accuracy_from_confusion_matrix(cm))),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "roc_auc": block.get("roc_auc"),
        "loss": float(block["loss"]),
        "per_class_f1": {
            str(i): float(rep[str(i)]["f1-score"]) for i in range(cm.shape[0])
        },
    }


def format_metrics_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "## Metrics Summary (evaluated artifact)",
        "",
        f"- **JSON Version:** {summary.get('version')}",
        f"- **Numeric Precision:** {summary.get('dtype')}",
        f"- **Samples (support):** {summary.get('support_total')}",
        f"- **Accuracy:** {summary.get('accuracy'):.6f}",
        f"- **Macro F1:** {summary.get('macro_f1'):.6f}",
        f"- **Weighted F1:** {summary.get('weighted_f1'):.6f}",
        f"- **Loss:** {summary.get('loss'):.6f}",
        f"- **ROC AUC:** {summary.get('roc_auc')}",
        "",
        "### F1 per class",
        "",
        "| Class | F1 |",
        "|-------|----|",
    ]
    for k, v in summary.get("per_class_f1", {}).items():
        name = ""
        names = summary.get("class_names") or []
        if names and k.isdigit() and int(k) < len(names):
            name = names[int(k)]
        label = f"{k} ({name})" if name else k
        lines.append(f"| {label} | {v:.6f} |")
    return "\n".join(lines)
