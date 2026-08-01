from __future__ import annotations

from pathlib import Path

import numpy as np

from sentience_train.metrics import write_confusion_reports
from sentience_train.metrics import metrics_from_predictions


def test_release_metrics_keep_canonical_class_order_and_exact_support() -> None:
    metrics = metrics_from_predictions([0, 1, 2, 3], [0, 2, 2, 3], np.eye(4), partition="PrivateTest", dtype="int8")

    assert metrics.to_dict()["class_names"] == ["angry", "happy", "neutral", "sad"]
    assert metrics.support_by_class == (1, 1, 1, 1)
    assert metrics.confusion_matrix == (
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )
    assert metrics.partition == "PrivateTest"
    assert list(metrics.to_dict()["recall_by_class"]) == [
        "angry",
        "happy",
        "neutral",
        "sad",
    ]


def test_confusion_matrix_rendering_is_byte_deterministic(tmp_path: Path) -> None:
    metrics = metrics_from_predictions([0, 1, 2, 3], [0, 1, 2, 3], np.eye(4), partition="PrivateTest", dtype="int8")
    first = write_confusion_reports(tmp_path / "first", {"reference": metrics, "tflite": metrics})
    second = write_confusion_reports(tmp_path / "second", {"reference": metrics, "tflite": metrics})

    assert first[2].read_bytes() == second[2].read_bytes()
    assert first[1].read_text(encoding="utf-8").splitlines()[0] == "model,actual,predicted,count"
