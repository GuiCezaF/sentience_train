from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import yaml

from sentience_train.config import load_config
from sentience_train.engine import run_training
from sentience_train.metrics import Metrics, ResourceMetrics


def _write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        raw_labels = (0, 3, 6, 4)
        for usage, count in (("Training", 512), ("PublicTest", 4), ("PrivateTest", 4)):
            for index in range(count):
                emotion = raw_labels[index % len(raw_labels)]
                writer.writerow([emotion, " ".join([str(index % 2)] * 2304), usage])


def _config(path: Path) -> None:
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
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "recall": 0.0,
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


def test_selection_freezes_first_width_and_never_reads_private_test(
    tmp_path: Path,
) -> None:
    _write_csv(tmp_path / "fer.csv")
    _config(tmp_path / "config.yaml")
    config = load_config(tmp_path / "config.yaml")
    calls: list[tuple[float, int]] = []

    def model_factory(_model_name, _num_classes):  # type: ignore[no-untyped-def]
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48 * 48, 4))

    def quantize(model, calibration, destination, route, width, seed):  # type: ignore[no-untyped-def]
        del model, calibration
        calls.append((width, seed))
        destination.write_bytes(b"synthetic-tflite")
        return {"metrics": _fake_metrics(), "resources": ResourceMetrics(1, 1, 1)}

    frozen = run_training(config, model_factory=model_factory, quantization_runner=quantize)
    assert frozen.status == "frozen"
    assert frozen.width == 0.25
    assert frozen.seed == 42
    assert calls == [(0.25, 42), (0.25, 43), (0.25, 44)]
    manifest = json.loads((frozen.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert manifest["csv"]["private_test_access"] is False
    assert manifest["selected"]["width"] == 0.25
    assert (frozen.directory / "candidate-comparison.json").is_file()
    assert (frozen.directory / "candidate-comparison.md").is_file()
