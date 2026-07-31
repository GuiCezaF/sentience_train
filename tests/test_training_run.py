from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import yaml

from sentience_train.config import load_config
from sentience_train.engine import run_training


def _write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        for usage in ("Training", "PublicTest", "PrivateTest"):
            for emotion in (0, 3, 6, 4):
                writer.writerow([emotion, " ".join([str(emotion)] * 2304), usage])


def _write_config(path: Path) -> None:
    path.write_text(yaml.safe_dump({
        "data": {"csv_path": "fer.csv", "classes": ["angry", "happy", "neutral", "sad"]},
        "model": {"name": "minivgg", "num_classes": 4},
        "training": {"batch_size": 4, "epochs": 1, "seed": 7, "device": "cpu", "num_workers": 0, "scheduler": "cosine"},
        "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001},
        "early_stopping": {"patience": 1, "min_delta": 0.002},
        "augmentation": {"horizontal_flip_probability": 0.0, "rotation_degrees": 0, "translate_fraction": 0, "scale_min": 1, "scale_max": 1, "brightness": 0, "contrast": 0},
        "run": {"directory": "runs"},
    }), encoding="utf-8")


def _tiny_model(_config):  # type: ignore[no-untyped-def]
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48 * 48, 4))


def test_cpu_run_is_immutable_and_private_test_is_not_exposed(tmp_path: Path) -> None:
    _write_csv(tmp_path / "fer.csv")
    _write_config(tmp_path / "config.yaml")
    config = load_config(tmp_path / "config.yaml")
    first = run_training(config, model_factory=_tiny_model)
    second = run_training(config, model_factory=_tiny_model)
    assert first.directory != second.directory
    for run in (first, second):
        assert (run.directory / "config.resolved.yaml").is_file()
        assert (run.directory / "checkpoints" / "best.pt").is_file()
        assert (run.directory / "metrics" / "history.jsonl").is_file()
        assert (run.directory / "metrics" / "validation.json").is_file()
        manifest = json.loads((run.directory / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["csv"]["release_evidence"] is False
        assert manifest["csv"]["partition_counts"]["PrivateTest"] == 4
    a = torch.load(first.directory / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    b = torch.load(second.directory / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]) for key in a["model_state_dict"])
    assert first.best_macro_f1 == second.best_macro_f1
    assert abs(first.best_loss - second.best_loss) <= 1e-8
