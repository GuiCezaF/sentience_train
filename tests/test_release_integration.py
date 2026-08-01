from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import sentience_train._release as release
import sentience_train.export as export
from sentience_train.config import LABELS
from sentience_train.export import RunResolution, export_frozen_run


class _FixedModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        raw = (((value[:, 0, 0, 0] + 1.0) * 127.5).round().long() // 64).clamp(0, 3)
        output = torch.full((value.shape[0], 4), -10.0)
        return output.scatter(1, raw[:, None], 10.0)


class _FakeOnnx:
    def export(self, _model: nn.Module, destination: Path) -> None:
        destination.write_bytes(b"onnx")


class _FakeTflite:
    def inspect(self, _path: Path):
        return export._TfliteInspection(
            export._TensorContract((1, 48, 48, 1), "int8", 1 / 127.5, 0),
            export._TensorContract((1, 4), "int8", 1.0, 0),
            (export._Operator("CONV_2D", 1),),
            ("int8",),
        )

    def predict(self, _path: Path, sample: np.ndarray, _inspection) -> np.ndarray:
        raw = int((((sample[0, 0, 0, 0] + 1.0) * 127.5).round()) // 64)
        result = np.full((1, 4), -10.0, dtype=np.float32)
        result[0, max(0, min(3, raw))] = 10.0
        return result


def _run(tmp_path: Path) -> RunResolution:
    run_directory = tmp_path / "runs" / "run-1"
    run_directory.mkdir(parents=True)
    dataset = tmp_path / "fer.csv"
    values = [0, 64, 128, 192]
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("emotion", "pixels", "Usage"))
        for usage in ("PublicTest", "PrivateTest"):
            for emotion, value in zip((0, 3, 6, 4), values):
                writer.writerow((emotion, " ".join([str(value)] * 2304), usage))
    checkpoint = run_directory / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    tflite = run_directory / "model.tflite"
    tflite.write_bytes(b"tflite")
    public_index = run_directory / "public.index.json"
    public_index.write_text("{}", encoding="utf-8")
    manifest = {
        "configuration": {"data": {"classes": LABELS}, "model": {"num_classes": 4}},
        "selected": {"checkpoint": "checkpoint.pt", "tflite": "model.tflite"},
        "csv": {
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "release_evidence": True,
            "partition_counts": {"PrivateTest": 4},
            "selected_class_counts": {"PrivateTest": dict(zip(LABELS, (1, 1, 1, 1)))},
        },
    }
    return RunResolution(
        "run-1",
        run_directory,
        manifest,
        "a" * 64,
        checkpoint,
        tflite,
        dataset,
        public_index,
        tuple({"offset": 0, "label": i, "line_number": i + 2, "row_sha256": "0" * 64} for i in range(4)),
        0.25,
        42,
        "ptq",
        "ds_cnn_w025",
        {},
        {},
        {
            "tflite_bytes": 7,
            "arena_total_bytes": 100,
            "arena_with_margin_bytes": 120,
            "macs": 100,
        },
    )


@pytest.fixture()
def release_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    run = _run(tmp_path)
    monkeypatch.setattr(
        release,
        "CANONICAL_FER2013_SHA256",
        hashlib.sha256(run.dataset.read_bytes()).hexdigest(),
    )
    model = _FixedModel().eval()
    monkeypatch.setattr(export, "_resolve_frozen_run", lambda _run_id, _runs_root: run)
    monkeypatch.setattr(export, "_restore_model", lambda _run: model)
    monkeypatch.setattr(export, "_onnx_adapter", lambda: _FakeOnnx())
    monkeypatch.setattr(
        export,
        "_validate_onnx",
        lambda _path, _model, _data: {
            "partition": "PublicTest",
            "class_agreement": 1.0,
            "max_absolute_error": 0.0,
            "metrics_pytorch": {},
            "metrics_onnx": {},
        },
    )
    monkeypatch.setattr(export, "_tflite_adapter", lambda: _FakeTflite())
    monkeypatch.setattr(
        export,
        "_tflite_predictions",
        lambda *_args: {
            "partition": "PublicTest",
            "class_agreement": 1.0,
            "accuracy_delta": 0.0,
            "metrics_pytorch": {},
            "metrics_tflite": {},
        },
    )
    monkeypatch.setattr(
        export,
        "_resource_payload",
        lambda copied, _model, _inspection: {
            "model_bytes": copied.tflite.stat().st_size,
            "tflite_bytes": copied.tflite.stat().st_size,
            "arena_with_margin_bytes": 120,
            "macs": 100,
            "arena": {"total_bytes": 100, "with_margin_bytes": 120},
        },
    )
    monkeypatch.setattr(export, "_macs", lambda _model: 100)

    def write_reports(directory, metrics_by_model):  # type: ignore[no-untyped-def]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "confusion-matrix.json").write_text("{}", encoding="utf-8")
        (directory / "confusion-matrix.csv").write_text("model,actual,predicted,count\n", encoding="utf-8")
        (directory / "confusion-matrix.png").write_bytes(b"png")
        return (
            directory / "confusion-matrix.json",
            directory / "confusion-matrix.csv",
            directory / "confusion-matrix.png",
        )

    monkeypatch.setattr(release, "write_confusion_reports", write_reports)
    return tmp_path / "runs", tmp_path / "exports"


def test_complete_release_is_separated_hashed_and_reusable(
    release_fixture: tuple[Path, Path],
) -> None:
    runs, exports = release_fixture

    result = export_frozen_run("run-1", runs_root=runs, exports_root=exports)

    assert result.status == "accepted"
    assert result.test_accessed is True
    assert (result.directory / "onnx" / "emotion_model_fp32.onnx").is_file()
    assert (result.directory / "tflite" / "emotion_model_int8.tflite").is_file()
    assert (result.directory / "validation" / "summary.json").is_file()
    assert json.loads((result.directory / "test" / "metrics.json").read_text())["partition"] == "PrivateTest"
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["status"] == "accepted"
    assert (
        manifest["metrics"]["sha256"]
        == hashlib.sha256((result.directory / "test" / "metrics.json").read_bytes()).hexdigest()
    )
    assert list((exports / "run-1" / ".private-test-access").iterdir())

    reused = export_frozen_run("run-1", runs_root=runs, exports_root=exports)
    assert reused.reused is True
    assert reused.status == "accepted"


def test_adulterated_complete_bundle_cannot_be_released_again(
    release_fixture: tuple[Path, Path],
) -> None:
    runs, exports = release_fixture
    result = export_frozen_run("run-1", runs_root=runs, exports_root=exports)
    (result.directory / "test" / "metrics.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(export.ExportCollisionError):
        export_frozen_run("run-1", runs_root=runs, exports_root=exports)
