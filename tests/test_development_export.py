from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import sentience_train.export as export_module
from sentience_train.export import (
    ExportCollisionError,
    ExportInputError,
    _Operator,
    _TensorContract,
    _TfliteInspection,
    export_development,
    resolve_frozen_run,
)
from sentience_train.selection import Metrics


LABELS = ["angry", "happy", "neutral", "sad"]


def _metrics(dtype: str) -> dict:
    return Metrics(
        accuracy=1.0,
        macro_f1=1.0,
        loss=0.0,
        recall_by_class=(1.0,) * 4,
        precision_by_class=(1.0,) * 4,
        f1_by_class=(1.0,) * 4,
        support_by_class=(1,) * 4,
        confusion_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype=dtype,
    ).to_dict()


def _fixture(tmp_path: Path) -> tuple[Path, Path, nn.Module]:
    runs = tmp_path / "runs"
    run = runs / "run-1"
    run.mkdir(parents=True)
    csv_path = tmp_path / "fer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        for usage in ("Training", "PublicTest", "PrivateTest"):
            for emotion in (0, 3, 6, 4):
                writer.writerow([emotion, " ".join(["0"] * 2304), usage])
    model = nn.Sequential(nn.Flatten(), nn.Linear(48 * 48, 4)).eval()
    checkpoint = run / "checkpoints" / "fp32.pt"
    checkpoint.parent.mkdir()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    tflite = run / "selection.tflite"
    tflite.write_bytes(b"frozen-tflite")
    dataset_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    records = []
    with csv_path.open("rb") as handle:
        offset = len(handle.readline())
        for line_number, raw in enumerate(handle, 2):
            if 6 <= line_number <= 9:
                emotion = int(raw.split(b",", 1)[0])
                records.append({"offset": offset, "label": {0: 0, 3: 1, 6: 2, 4: 3}[emotion], "line_number": line_number, "row_sha256": hashlib.sha256(raw).hexdigest()})
            offset += len(raw)
    index_path = run / "data" / "public_test.index.json"
    index_path.parent.mkdir()
    index_path.write_text(json.dumps({"schema": "public-test-index/v1", "partition": "PublicTest", "labels": LABELS, "dataset": {"path": str(csv_path), "sha256": dataset_hash}, "count": len(records), "records": records}, sort_keys=True), encoding="utf-8")
    (run / "config.resolved.yaml").write_text("data: {}\n", encoding="utf-8")
    resource = {"tflite_bytes": len(tflite.read_bytes()), "arena_total_bytes": 100, "arena_with_margin_bytes": 120, "arena_head_bytes": 10, "arena_temporary_bytes": 20, "arena_tail_bytes": 70, "macs": 9216}
    selected = {"architecture": "DS-CNN", "model_name": "ds_cnn_w025", "width": 0.25, "seed": 42, "route": "ptq", "checkpoint": "checkpoints/fp32.pt", "checkpoint_fp32": "checkpoints/fp32.pt", "tflite": "selection.tflite", "metrics": {"fp32": _metrics("float32"), "int8": _metrics("int8")}, "metrics_fp32": _metrics("float32"), "metrics_int8": _metrics("int8"), "resources": resource}
    manifest = {"schema": "frozen-run/v1", "status": "frozen", "private_test_access": False, "run_id": "run-1", "labels": LABELS, "configuration": {"data": {"classes": LABELS}, "model": {"num_classes": 4}}, "csv": {"path": str(csv_path), "sha256": dataset_hash, "schema": ["emotion", "pixels", "Usage"], "private_test_access": False}, "public_test_index": "data/public_test.index.json", "selected": selected}
    manifest["artifacts"] = {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest() for path in run.rglob("*") if path.is_file()}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return runs, run, model


def test_resolver_rejects_traversal_and_validates_public_index(tmp_path: Path) -> None:
    runs, run, _model = _fixture(tmp_path)
    resolved = resolve_frozen_run("run-1", runs)
    assert len(resolved.index) == 4
    with pytest.raises(ExportInputError):
        resolve_frozen_run("../run-1", runs)
    before = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(run.rglob("*")) if path.is_file())).hexdigest()
    export_module._verify_run_artifacts(run, resolved.manifest)
    after = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(run.rglob("*")) if path.is_file())).hexdigest()
    assert before == after


def test_onnx_bundle_is_dynamic_and_reuses_only_intact_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs, run, model = _fixture(tmp_path)
    monkeypatch.setattr(export_module, "_restore_model", lambda _run: model)
    result = export_development("run-1", "onnx", runs, tmp_path / "exports")
    assert result.reused is False
    assert {path.name for path in result.directory.iterdir()} == {"manifest.json", "labels.txt", "preprocessing.json", "input_contract.json", "output_contract.json", "validation.json", "emotion_model_fp32.onnx"}
    assert json.loads((result.directory / "validation.json").read_text())["partition"] == "PublicTest"
    reused = export_development("run-1", "onnx", runs, tmp_path / "exports")
    assert reused.reused is True
    (result.directory / "emotion_model_fp32.onnx").write_bytes(b"tampered")
    with pytest.raises(ExportCollisionError):
        export_development("run-1", "onnx", runs, tmp_path / "exports")


def test_tflite_is_copied_and_cpp_matches_flatbuffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs, _run, model = _fixture(tmp_path)

    class FakeAdapter:
        def inspect(self, _path: Path) -> _TfliteInspection:
            return _TfliteInspection(_TensorContract((1, 48, 48, 1), "int8", 1 / 127.5, 0), _TensorContract((1, 4), "int8", 1.0, 0), (_Operator("CONV_2D", 1),), ("int8",), None)

        def predict(self, _path: Path, sample: np.ndarray, _inspection: _TfliteInspection) -> np.ndarray:
            with torch.no_grad():
                return model(torch.from_numpy(sample.transpose(0, 3, 1, 2))).numpy()

    monkeypatch.setattr(export_module, "_restore_model", lambda _run: model)
    monkeypatch.setattr(export_module, "_TFLITE_ADAPTER", FakeAdapter())
    result = export_development("run-1", "tflite", runs, tmp_path / "exports")
    assert (result.directory / "emotion_model_int8.tflite").read_bytes() == b"frozen-tflite"
    assert "alignas(16)" in (result.directory / "emotion_model_data.cc").read_text()
    assert "0x66" in (result.directory / "emotion_model_data.cc").read_text()
