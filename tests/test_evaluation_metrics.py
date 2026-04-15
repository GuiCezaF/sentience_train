from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from evaluation import (
    accuracy_from_confusion_matrix,
    build_evaluation_metrics_document,
    build_metrics_summary,
    classification_report_matches_matrix,
    load_evaluation_metrics_json,
    validate_evaluation_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

# Same order as `ImageFolder` with class folders angry/happy/neutral/sad and `config.CLASS_NAMES`.
_CLASS_NAMES = ["angry", "happy", "neutral", "sad"]


@pytest.fixture(scope="session")
def evaluation_metrics_doc() -> dict:
    """
    Run the checkpoint when available; otherwise use the synthetic bundle.
    Writes JSON and metadata under `reports/` for the session report.
    """
    from _prediction_capture import collect_predictions_for_metrics_tests

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = collect_predictions_for_metrics_tests()
    doc = build_evaluation_metrics_document(
        bundle.y_true,
        bundle.y_pred,
        _CLASS_NAMES,
        bundle.y_score,
    )
    (REPORTS_DIR / "evaluation_run_meta.txt").write_text(
        f"prediction_source={bundle.source}\n",
        encoding="utf-8",
    )
    out = REPORTS_DIR / "evaluation_metrics.generated.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def test_metrics_json_roundtrip_through_loader(
    tmp_path: Path, evaluation_metrics_doc: dict
) -> None:
    p = tmp_path / "eval_metrics.json"
    p.write_text(json.dumps(evaluation_metrics_doc), encoding="utf-8")
    loaded = load_evaluation_metrics_json(p)
    assert loaded["version"] == evaluation_metrics_doc["version"]
    assert validate_evaluation_metrics(loaded) == []


def test_document_has_expected_top_level_keys(evaluation_metrics_doc: dict) -> None:
    assert evaluation_metrics_doc["version"] == 6
    assert "test" in evaluation_metrics_doc
    assert "float32" in evaluation_metrics_doc["test"]


def test_structural_validation_empty_errors(evaluation_metrics_doc: dict) -> None:
    errs = validate_evaluation_metrics(evaluation_metrics_doc)
    assert errs == [], errs


def test_matrix_is_square_and_row_sums_match_support(
    evaluation_metrics_doc: dict,
) -> None:
    block = evaluation_metrics_doc["test"]["float32"]
    cm = np.asarray(block["confusion_matrix"])
    assert cm.shape[0] == cm.shape[1] == 4
    rep = block["classification_report"]
    for i in range(4):
        assert cm[i].sum() == rep[str(i)]["support"]


def test_accuracy_matches_confusion_matrix(evaluation_metrics_doc: dict) -> None:
    block = evaluation_metrics_doc["test"]["float32"]
    cm = np.asarray(block["confusion_matrix"])
    acc_rep = float(block["classification_report"]["accuracy"])
    acc_cm = accuracy_from_confusion_matrix(cm)
    assert np.isclose(acc_rep, acc_cm)


def test_classification_report_matches_sklearn(evaluation_metrics_doc: dict) -> None:
    block = evaluation_metrics_doc["test"]["float32"]
    cm = np.asarray(block["confusion_matrix"])
    rep = block["classification_report"]
    mismatches = classification_report_matches_matrix(cm, rep, atol=1e-5)
    assert mismatches == [], mismatches


def test_metrics_summary_bounds_and_support(evaluation_metrics_doc: dict) -> None:
    s = build_metrics_summary(evaluation_metrics_doc)
    cm = np.asarray(evaluation_metrics_doc["test"]["float32"]["confusion_matrix"])
    assert s["class_names"] == ["angry", "happy", "neutral", "sad"]
    assert s["support_total"] == int(cm.sum())
    assert s["dtype"] == "float32"
    assert 0.0 <= float(s["accuracy"]) <= 1.0
    assert 0.0 <= float(s["macro_f1"]) <= 1.0


def test_wrong_version_is_rejected(evaluation_metrics_doc: dict) -> None:
    doc = {**evaluation_metrics_doc, "version": 99}
    errs = validate_evaluation_metrics(doc, expected_version=6)
    assert any("expected version" in e for e in errs)


def test_document_without_test_key_rejected() -> None:
    errs = validate_evaluation_metrics({"version": 6})
    assert any("test" in e.lower() for e in errs)


def test_weighted_average_matches_report(evaluation_metrics_doc: dict) -> None:
    block = evaluation_metrics_doc["test"]["float32"]
    wa = block["weighted_average"]
    w = block["classification_report"]["weighted avg"]
    for k in ("precision", "recall", "f1-score"):
        assert np.isclose(wa[k], w[k])


def test_roc_auc_is_finite_and_in_unit_interval(evaluation_metrics_doc: dict) -> None:
    auc = evaluation_metrics_doc["test"]["float32"].get("roc_auc")
    assert auc is not None
    v = float(auc)
    assert not math.isnan(v)
    assert 0.0 <= v <= 1.0


def test_subgroup_facetted_is_null(evaluation_metrics_doc: dict) -> None:
    assert (
        evaluation_metrics_doc["test"]["float32"]["subgroup_metrics"]["facetted"]
        is None
    )


def test_subgroup_per_class_rows_match_global_matrix(
    evaluation_metrics_doc: dict,
) -> None:
    block = evaluation_metrics_doc["test"]["float32"]
    cm = np.asarray(block["confusion_matrix"])
    per = block["subgroup_metrics"]["per_key"]["class_name"]["per_group"]
    for idx, name in enumerate(block["class_names"]):
        gcm = np.asarray(per[name]["confusion_matrix"])
        assert np.array_equal(gcm[idx], cm[idx])
        mask = np.ones(4, dtype=bool)
        mask[idx] = False
        assert int(gcm[mask].sum()) == 0
