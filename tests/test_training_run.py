from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch
import yaml
import pytest

from sentience_train.config import load_config
from sentience_train.engine import run_training
from sentience_train.metrics import Metrics, ResourceMetrics
from sentience_train.selection import SelectionFailed


def _write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        for usage, count in (("Training", 512), ("PublicTest", 4), ("PrivateTest", 4)):
            for index in range(count):
                emotion = (0, 3, 6, 4)[index % 4]
                writer.writerow([emotion, " ".join([str(index % 2)] * 2304), usage])


def _write_config(path: Path, *, accuracy: float = 0.0) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "csv_path": "fer.csv",
                    "classes": ["angry", "happy", "neutral", "sad"],
                },
                "model": {"num_classes": 4},
                "training": {
                    "batch_size": 512,
                    "epochs": 1,
                    "device": "cpu",
                    "num_workers": 0,
                    "scheduler": "cosine",
                },
                "selection": {
                    "widths": [0.25, 0.35, 0.50],
                    "seeds": [42, 43, 44],
                    "representative": "median_int8_macro_f1_then_lowest_seed",
                },
                "quality_gates": {
                    "accuracy": accuracy,
                    "macro_f1": accuracy,
                    "recall": accuracy,
                    "max_accuracy_delta": 0.02,
                    "tflite_bytes": 100,
                    "arena_with_margin_bytes": 10,
                    "macs": 10,
                },
                "quantization": {
                    "calibration_seed": 42,
                    "calibration_examples": 512,
                    "qat": {"enabled": True, "epochs": 1, "lr": 0.00001},
                },
                "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001},
                "early_stopping": {"patience": 12, "min_delta": 0.002},
                "augmentation": {
                    "horizontal_flip_probability": 0.0,
                    "rotation_degrees": 0,
                    "translate_fraction": 0,
                    "scale_min": 1,
                    "scale_max": 1,
                    "brightness": 0,
                    "contrast": 0,
                },
                "run": {"directory": "runs"},
            }
        ),
        encoding="utf-8",
    )


def _tiny_model(_model_name: str, num_classes: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48 * 48, num_classes))


def _fake_metrics() -> Metrics:
    return Metrics(
        0.8,
        0.7,
        0.2,
        (0.6,) * 4,
        (0.7,) * 4,
        (0.7,) * 4,
        (1,) * 4,
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype="int8",
    )


def _quantize(model, calibration, destination, route, width, seed):  # type: ignore[no-untyped-def]
    del model, calibration, route, width, seed
    destination.write_bytes(b"synthetic-tflite")
    return {"metrics": _fake_metrics(), "resources": ResourceMetrics(1, 1, 1)}


def test_selection_freezes_first_width_and_uses_explicit_model_factory(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_csv(tmp_path / "fer.csv")
    _write_config(tmp_path / "config.yaml")
    config = load_config(tmp_path / "config.yaml")
    caplog.set_level(logging.INFO, logger="sentience_train.engine")
    names: list[tuple[str, int]] = []

    def factory(model_name: str, num_classes: int) -> torch.nn.Module:
        names.append((model_name, num_classes))
        return _tiny_model(model_name, num_classes)

    frozen = run_training(config, model_factory=factory, quantization_runner=_quantize)
    assert frozen.status == "frozen"
    assert frozen.width == 0.25
    assert names == [("ds_cnn_w025", 4)] * 3
    manifest = json.loads((frozen.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert manifest["csv"]["private_test_access"] is False
    assert "teacher" not in manifest
    assert "Training device: cpu (requested: cpu)" in caplog.text
    assert "FP32 w=0.25 seed=42" in capsys.readouterr().err
    selection_log = [
        json.loads(line)
        for line in (frozen.directory / "logs" / "selection.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert selection_log[0]["device"] == "cpu"
    assert selection_log[0]["requested_device"] == "cpu"


def test_all_candidates_fail_with_immutable_diagnosis_and_no_private_test(
    tmp_path: Path,
) -> None:
    _write_csv(tmp_path / "fer.csv")
    _write_config(tmp_path / "config.yaml", accuracy=1.0)
    config = load_config(tmp_path / "config.yaml")
    with pytest.raises(SelectionFailed) as failure:
        run_training(config, model_factory=_tiny_model, quantization_runner=_quantize)
    manifest = json.loads((failure.value.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "selection_failed"
    assert manifest["private_test_access"] is False
    assert not (failure.value.directory / "teacher").exists()
    assert failure.value.diagnosis.is_file()
