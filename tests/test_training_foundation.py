from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from sentience_train.config import ConfigError, load_config
from sentience_train.data import DataValidationError, load_training_data


def _config(tmp_path: Path, csv_path: Path) -> Path:
    document = {
        "data": {"csv_path": csv_path.name, "classes": ["angry", "happy", "neutral", "sad"]},
        "model": {"name": "ds_cnn_w025", "num_classes": 4},
        "training": {"batch_size": 4, "epochs": 1, "seed": 42, "device": "cpu", "num_workers": 0, "scheduler": "cosine"},
        "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001},
        "early_stopping": {"patience": 1, "min_delta": 0.002},
        "augmentation": {"horizontal_flip_probability": 0.5, "rotation_degrees": 10, "translate_fraction": 0.08, "scale_min": 0.95, "scale_max": 1.05, "brightness": 0.15, "contrast": 0.15},
        "run": {"directory": "runs"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "fer.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        pixels = " ".join(["0"] * 2304)
        for usage in ("Training", "PublicTest", "PrivateTest"):
            for emotion in (0, 3, 6, 4, 1):
                writer.writerow([emotion, pixels, usage])
    return path


def test_config_resolves_paths_overrides_and_is_frozen(tmp_path: Path) -> None:
    path = _config(tmp_path, _csv(tmp_path))
    config = load_config(path, {"training.epochs": 3})
    assert config.data.csv_path == (tmp_path / "fer.csv").resolve()
    assert config.training.epochs == 3
    with pytest.raises(FrozenInstanceError):
        config.training.epochs = 4  # type: ignore[misc]


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _config(tmp_path, _csv(tmp_path))
    document = yaml.safe_load(path.read_text())
    document["model"]["typo"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_csv_keeps_private_test_as_provenance_only(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path)
    data = load_training_data(load_config(_config(tmp_path, csv_path)).data)
    assert len(data.train) == 4
    assert len(data.validation) == 4
    assert not hasattr(data, "private_test")
    assert data.provenance.partition_counts["PrivateTest"] == 5
    assert data.provenance.release_evidence is False
    image, label = data.validation[0]
    assert image.shape == (1, 48, 48)
    assert label == 0


def test_csv_reports_invalid_pixel_with_context(tmp_path: Path) -> None:
    path = _csv(tmp_path)
    rows = list(csv.reader(path.open(encoding="utf-8")))
    rows[1][1] = "256 " + " ".join(["0"] * 2303)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(DataValidationError, match="line 2, field pixels"):
        load_training_data(load_config(_config(tmp_path, path)).data)
