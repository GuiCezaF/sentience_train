"""Immutable exports for a validated frozen training run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .artifacts import sha256_file
from .config import LABELS
from .models import build_candidate
from .preflight import ALLOWED_BUILTINS, MAX_OPERATOR_VERSIONS, _macs
from .metrics import Metrics, metrics_from_predictions


EXPORT_SCHEMA = "development-export-bundle/v1"
RELEASE_SCHEMA = "release-bundle/v1"
ACCESS_SCHEMA = "private-test-access/v1"
RELEASE_METRICS_SCHEMA = "release-metrics/v1"
RUN_SCHEMA = "frozen-run/v1"
INDEX_SCHEMA = "public-test-index/v1"
EXPORTER_REVISION = "issue-5-development-export-v1"
ONNX_OPSET = 17
ONNX_ATOL = 1e-5
ONNX_RTOL = 1e-4
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExportError(RuntimeError):
    """Base class for errors with a stable CLI classification."""


class ExportInputError(ExportError):
    """The run or its immutable input contract is incompatible."""


class ExportCollisionError(ExportError):
    """An existing bundle cannot be safely reused."""


class ExportValidationError(ExportError):
    """An adapter, contract, parity, or payload validation failed."""


@dataclass(frozen=True)
class RunResolution:
    run_id: str
    directory: Path
    manifest: Mapping[str, Any]
    source_manifest_sha256: str
    checkpoint: Path
    tflite: Path
    dataset: Path
    public_index: Path
    index: tuple[Mapping[str, Any], ...]
    width: float
    seed: int
    route: str
    model_name: str
    metrics_fp32: Mapping[str, Any]
    metrics_int8: Mapping[str, Any]
    resources: Mapping[str, Any]


@dataclass(frozen=True)
class ExportResult:
    run_id: str
    format: str
    directory: Path
    reused: bool
    identity: str
    status: str = "development"
    test_accessed: bool = False


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportInputError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportInputError(f"JSON artifact {path} must contain an object")
    return value


def _safe_relative(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExportInputError(f"{where} must be a non-empty relative POSIX path")
    candidate = Path(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ExportInputError(f"{where} must stay below the run directory")
    return candidate


def _inside(root: Path, relative: Any, where: str) -> Path:
    path = (root / _safe_relative(relative, where)).resolve()
    if path.parent != root.resolve() and root.resolve() not in path.parents:
        raise ExportInputError(f"{where} escapes the run directory")
    return path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _schema(document: Mapping[str, Any], expected: str, where: str) -> None:
    received = document.get("schema", document.get("schema_version"))
    if received != expected:
        raise ExportInputError(f"{where} has unsupported schema {received!r}; expected {expected!r}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportInputError(f"{where} must be an object")
    return value


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ExportInputError(f"{where} is missing {key!r}")
    return mapping[key]


def _verify_run_artifacts(run: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = _mapping(manifest.get("artifacts"), "manifest.artifacts")
    if not artifacts:
        raise ExportInputError("manifest.artifacts is incomplete")
    for relative, expected in artifacts.items():
        path = _inside(run, relative, "manifest.artifacts path")
        if not path.is_file():
            raise ExportInputError(f"run artifact is missing: {relative}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ExportInputError(f"invalid hash for run artifact {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ExportInputError(f"hash mismatch for run artifact {relative}")


def _load_public_index(
    run: Path, manifest: Mapping[str, Any], dataset: Path
) -> tuple[Path, tuple[Mapping[str, Any], ...]]:
    index_path = _inside(
        run,
        _required(manifest, "public_test_index", "manifest"),
        "manifest.public_test_index",
    )
    index = _json(index_path)
    _schema(index, INDEX_SCHEMA, "public_test_index")
    if index.get("partition") != "PublicTest" or index.get("labels") != list(LABELS):
        raise ExportInputError("PublicTest index has an incompatible partition or labels contract")
    index_dataset = _mapping(_required(index, "dataset", "public_test_index"), "public_test_index.dataset")
    expected_dataset_hash = _required(manifest.get("csv", {}), "sha256", "manifest.csv")
    if index_dataset.get("sha256") != expected_dataset_hash:
        raise ExportInputError("PublicTest index dataset hash does not match the run manifest")
    if Path(str(index_dataset.get("path", ""))).resolve() != dataset.resolve():
        raise ExportInputError("PublicTest index dataset path does not match the run manifest")
    records = _required(index, "records", "public_test_index")
    if not isinstance(records, list) or not records or index.get("count") != len(records):
        raise ExportInputError("PublicTest index is incomplete")
    previous_offset = -1
    normalized: list[Mapping[str, Any]] = []
    for position, record in enumerate(records):
        item = _mapping(record, f"public_test_index.records[{position}]")
        offset = item.get("offset")
        label = item.get("label")
        line_number = item.get("line_number")
        digest = item.get("row_sha256", item.get("sha256"))
        if not isinstance(offset, int) or offset < 0 or offset <= previous_offset:
            raise ExportInputError("PublicTest index offsets must be strictly increasing")
        if not isinstance(label, int) or label not in range(len(LABELS)):
            raise ExportInputError("PublicTest index contains a non-canonical label")
        if not isinstance(line_number, int) or line_number < 2:
            raise ExportInputError("PublicTest index contains an invalid line number")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ExportInputError("PublicTest index contains an invalid row hash")
        previous_offset = offset
        normalized.append(
            {
                "offset": offset,
                "label": label,
                "line_number": line_number,
                "row_sha256": digest,
            }
        )
    return index_path, tuple(normalized)


def _verify_index_rows(dataset: Path, records: Sequence[Mapping[str, Any]]) -> None:
    try:
        with dataset.open("rb") as handle:
            for position, record in enumerate(records):
                handle.seek(int(record["offset"]))
                raw = handle.readline()
                if _sha256_bytes(raw) != record["row_sha256"]:
                    raise ExportInputError(f"PublicTest row {position} hash mismatch")
                try:
                    row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
                    raw_emotion = int(row[0])
                except (
                    UnicodeDecodeError,
                    csv.Error,
                    StopIteration,
                    ValueError,
                    IndexError,
                ) as exc:
                    raise ExportInputError(f"PublicTest row {position} is not a valid FER2013 row") from exc
                mapping = {0: 0, 3: 1, 6: 2, 4: 3}
                if mapping.get(raw_emotion) != record["label"]:
                    raise ExportInputError(f"PublicTest row {position} label mismatch")
                if len(row) != 3 or row[2] != "PublicTest" or len(row[1].split()) != 2304:
                    raise ExportInputError(f"PublicTest row {position} violates the frozen row contract")
    except OSError as exc:
        raise ExportInputError(f"cannot read indexed PublicTest rows from {dataset}: {exc}") from exc


def _validate_frozen_metrics(value: Mapping[str, Any], where: str, dtype: str) -> None:
    if value.get("partition") != "PublicTest" or value.get("dtype") != dtype:
        raise ExportInputError(f"{where} must be complete PublicTest {dtype} metrics")
    if value.get("class_names") != list(LABELS):
        raise ExportInputError(f"{where} has incompatible class names")


def _resolve_frozen_run(run_id: str, runs_root: Path) -> RunResolution:
    """Resolve and fully validate one direct child of runs_root."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ExportInputError("run-id must be a simple direct-child identifier")
    root = Path(runs_root).expanduser().resolve()
    run = root / run_id
    if run.parent != root or run.is_symlink() or not run.is_dir():
        raise ExportInputError(f"run does not exist as a direct child of {root}: {run_id}")
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise ExportInputError(f"run is incomplete: {manifest_path}")
    raw_manifest = manifest_path.read_bytes()
    manifest = _json(manifest_path)
    _schema(manifest, RUN_SCHEMA, "run manifest")
    if manifest.get("status") != "frozen":
        raise ExportInputError("run is not frozen")
    if manifest.get("run_id") != run_id:
        raise ExportInputError("run_id in manifest does not match the requested id")
    if manifest.get("labels") != list(LABELS):
        raise ExportInputError("run labels are not canonical")
    if manifest.get("private_test_access") is not False:
        raise ExportInputError("run has forbidden reserved-partition access")
    configuration = _mapping(_required(manifest, "configuration", "run manifest"), "configuration")
    data_config = _mapping(_required(configuration, "data", "configuration"), "configuration.data")
    if data_config.get("classes") != list(LABELS):
        raise ExportInputError("run configuration has incompatible labels")
    model_config = _mapping(_required(configuration, "model", "configuration"), "configuration.model")
    if model_config.get("num_classes") != len(LABELS):
        raise ExportInputError("run configuration has an incompatible class count")
    selected = _mapping(_required(manifest, "selected", "run manifest"), "selected")
    architecture = selected.get("architecture")
    if architecture not in {"DS-CNN", "DS_CNN", "ds-cnn"}:
        raise ExportInputError("selected architecture is not the frozen DS-CNN family")
    width = selected.get("width")
    if not isinstance(width, (int, float)) or float(width) not in {0.25, 0.35, 0.50}:
        raise ExportInputError("selected width is incompatible")
    seed = selected.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExportInputError("selected seed is invalid")
    route = selected.get("route")
    if route not in {"ptq", "qat"}:
        raise ExportInputError("selected quantization route is invalid")
    model_name = selected.get("model_name", f"ds_cnn_w{int(round(float(width) * 100)):03d}")
    expected_name = f"ds_cnn_w{int(round(float(width) * 100)):03d}"
    if model_name != expected_name:
        raise ExportInputError("selected model name does not match width")
    checkpoint_value = selected.get("checkpoint") or (
        selected.get("checkpoint_qat") if route == "qat" else selected.get("checkpoint_fp32")
    )
    tflite_value = selected.get("tflite") or selected.get("int8_artifact")
    checkpoint = _inside(run, checkpoint_value, "selected.checkpoint")
    tflite = _inside(run, tflite_value, "selected.tflite")
    if not checkpoint.is_file() or not tflite.is_file():
        raise ExportInputError("selected run artifacts are incomplete")
    metrics = _mapping(_required(selected, "metrics", "selected"), "selected.metrics")
    metrics_fp32 = _mapping(selected.get("metrics_fp32", metrics.get("fp32")), "selected.metrics_fp32")
    metrics_int8 = _mapping(selected.get("metrics_int8", metrics.get("int8")), "selected.metrics_int8")
    _validate_frozen_metrics(metrics_fp32, "selected.metrics_fp32", "float32")
    _validate_frozen_metrics(metrics_int8, "selected.metrics_int8", "int8")
    resources = _mapping(_required(selected, "resources", "selected"), "selected.resources")
    resource_arena = resources.get("arena")
    if isinstance(resource_arena, Mapping):
        resources = {
            **resources,
            "arena_total_bytes": resource_arena.get("total_bytes", resources.get("arena_total_bytes")),
            "arena_with_margin_bytes": resource_arena.get(
                "with_margin_bytes", resources.get("arena_with_margin_bytes")
            ),
        }
    for resource_key in (
        "tflite_bytes",
        "arena_total_bytes",
        "arena_with_margin_bytes",
        "macs",
    ):
        if not isinstance(resources.get(resource_key), int) or resources[resource_key] < 0:
            raise ExportInputError(f"selected.resources is missing {resource_key}")
    csv_info = _mapping(_required(manifest, "csv", "run manifest"), "manifest.csv")
    if csv_info.get("schema") not in (None, ["emotion", "pixels", "Usage"]):
        raise ExportInputError("run CSV schema is incompatible")
    if csv_info.get("private_test_access") is not False:
        raise ExportInputError("run CSV provenance does not explicitly deny reserved-partition access")
    dataset = Path(str(_required(csv_info, "path", "manifest.csv"))).expanduser().resolve()
    if not dataset.is_file():
        raise ExportInputError(f"frozen dataset is missing: {dataset}")
    dataset_hash = _required(csv_info, "sha256", "manifest.csv")
    if not isinstance(dataset_hash, str) or sha256_file(dataset) != dataset_hash:
        raise ExportInputError("frozen dataset hash does not match the run manifest")
    _verify_run_artifacts(run, manifest)
    public_index, records = _load_public_index(run, manifest, dataset)
    _verify_index_rows(dataset, records)
    return RunResolution(
        run_id=run_id,
        directory=run,
        manifest=manifest,
        source_manifest_sha256=_sha256_bytes(raw_manifest),
        checkpoint=checkpoint,
        tflite=tflite,
        dataset=dataset,
        public_index=public_index,
        index=records,
        width=float(width),
        seed=seed,
        route=route,
        model_name=model_name,
        metrics_fp32=metrics_fp32,
        metrics_int8=metrics_int8,
        resources=resources,
    )


class _IndexedPublicTest(Dataset[tuple[torch.Tensor, int]]):
    """A dataset with no discovery operation: it only reads indexed offsets."""

    def __init__(self, csv_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        self.csv_path = csv_path
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with self.csv_path.open("rb") as handle:
            handle.seek(int(record["offset"]))
            raw = handle.readline()
        if _sha256_bytes(raw) != record["row_sha256"]:
            raise ExportValidationError(f"indexed PublicTest row {index} changed during export")
        row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
        values = torch.tensor([int(value) for value in row[1].split()], dtype=torch.float32).reshape(1, 48, 48)
        return values.div(127.5).sub(1.0), int(record["label"])


def _state_dict(checkpoint: Path) -> Mapping[str, torch.Tensor]:
    try:
        value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ExportInputError(f"cannot restore frozen checkpoint with weights_only=True: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ExportInputError("frozen checkpoint must contain a state mapping")
    for key in ("model_state_dict", "model_state", "state_dict"):
        if isinstance(value.get(key), Mapping):
            value = value[key]
            break
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            continue
        while key.startswith(("model.", "module.")):
            key = key.split(".", 1)[1]
        key = key.replace(".wrapped.", ".")
        normalized[key] = tensor
    if not normalized:
        raise ExportInputError("frozen checkpoint has no tensor state")
    return normalized


def _restore_model(run: RunResolution) -> nn.Module:
    model = build_candidate(run.model_name, len(LABELS)).to("cpu").eval()
    try:
        model.load_state_dict(_state_dict(run.checkpoint), strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ExportInputError(f"frozen checkpoint does not strictly match {run.model_name}: {exc}") from exc
    return model


def _batches(
    dataset: Dataset[tuple[torch.Tensor, int]], batch_size: int
) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
    values: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        values.append(
            (
                torch.stack([row[0] for row in rows]),
                torch.tensor([row[1] for row in rows], dtype=torch.long),
            )
        )
    return values


def _torch_predictions(
    model: nn.Module, dataset: Dataset[tuple[torch.Tensor, int]], *, batch_size: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    actual: list[int] = []
    logits: list[np.ndarray] = []
    batch_shapes: list[int] = []
    with torch.no_grad():
        for images, labels in _batches(dataset, batch_size):
            batch_shapes.append(int(images.shape[0]))
            output = model(images).detach().cpu().numpy().astype(np.float32)
            logits.append(output)
            actual.extend(labels.tolist())
    scores = np.concatenate(logits, axis=0)
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-15)
    return np.asarray(actual, dtype=np.int64), scores, probabilities, batch_shapes


def _metrics(actual: np.ndarray, scores: np.ndarray, *, dtype: str) -> Metrics:
    return metrics_from_predictions(
        actual,
        scores.argmax(axis=1),
        _softmax(scores),
        dtype=dtype,
        partition="PublicTest",
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-15)


class _OnnxAdapter(Protocol):
    def export(self, model: nn.Module, destination: Path) -> None: ...


class _RealOnnxAdapter:
    def export(self, model: nn.Module, destination: Path) -> None:
        try:
            import onnx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ExportValidationError(f"ONNX adapter unavailable: {exc}") from exc
        dummy = torch.zeros((1, 1, 48, 48), dtype=torch.float32)
        try:
            torch.onnx.export(
                model,
                dummy,
                str(destination),
                opset_version=ONNX_OPSET,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                dynamo=False,
            )
            onnx.checker.check_model(onnx.load(str(destination)))
        except Exception as exc:
            raise ExportValidationError(f"ONNX export/checker failed: {exc}") from exc


_ONNX_ADAPTER: _OnnxAdapter | None = None


def _onnx_adapter() -> _OnnxAdapter:
    return _ONNX_ADAPTER or _RealOnnxAdapter()


def _validate_onnx(path: Path, model: nn.Module, dataset: Dataset[tuple[torch.Tensor, int]]) -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ExportValidationError(f"ONNX Runtime adapter unavailable: {exc}") from exc
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("expected one ONNX input and one output")
        input_meta, output_meta = inputs[0], outputs[0]
        input_shape = list(input_meta.shape)
        output_shape = list(output_meta.shape)
        if (
            input_meta.name != "input"
            or input_meta.type != "tensor(float)"
            or input_shape[0] == 1
            or input_shape[1:] != [1, 48, 48]
        ):
            raise ValueError(f"invalid ONNX input contract: {input_meta.name}/{input_meta.type}/{input_meta.shape}")
        if (
            output_meta.name != "logits"
            or output_meta.type != "tensor(float)"
            or output_shape[0] == 1
            or output_shape[1:] != [4]
        ):
            raise ValueError(f"invalid ONNX output contract: {output_meta.name}/{output_meta.type}/{output_meta.shape}")
        actual, torch_logits, torch_scores, batch_shapes = _torch_predictions(model, dataset, batch_size=3)
        onnx_logits: list[np.ndarray] = []
        onnx_shapes: list[int] = []
        for images, _labels in _batches(dataset, 3):
            value = session.run([output_meta.name], {input_meta.name: images.numpy().astype(np.float32)})[0]
            onnx_shapes.append(int(value.shape[0]))
            onnx_logits.append(np.asarray(value, dtype=np.float32))
        first_image = dataset[0][0].unsqueeze(0).numpy().astype(np.float32)
        dynamic_probe_shapes: list[int] = []
        for probe_batch in (
            first_image,
            np.repeat(first_image, 2, axis=0),
            np.repeat(first_image, 1, axis=0),
        ):
            probe_output = session.run([output_meta.name], {input_meta.name: probe_batch})[0]
            dynamic_probe_shapes.append(int(probe_output.shape[0]))
        onnx_values = np.concatenate(onnx_logits, axis=0)
        absolute = np.abs(torch_logits - onnx_values)
        relative = absolute / np.maximum(np.abs(torch_logits), 1e-12)
        max_absolute = float(absolute.max(initial=0.0))
        max_relative = float(relative.max(initial=0.0))
        predicted_torch = torch_logits.argmax(axis=1)
        predicted_onnx = onnx_values.argmax(axis=1)
        if not np.allclose(torch_logits, onnx_values, atol=ONNX_ATOL, rtol=ONNX_RTOL) or not np.array_equal(
            predicted_torch, predicted_onnx
        ):
            raise ExportValidationError("PyTorch/ONNX PublicTest parity failed")
        return {
            "partition": "PublicTest",
            "atol": ONNX_ATOL,
            "rtol": ONNX_RTOL,
            "max_absolute_error": max_absolute,
            "max_relative_error": max_relative,
            "class_agreement": float(np.mean(predicted_torch == predicted_onnx)),
            "metrics_pytorch": _metrics(actual, torch_logits, dtype="float32").to_dict(),
            "metrics_onnx": _metrics(actual, onnx_values, dtype="float32").to_dict(),
            "dynamic_batch": {
                "pytorch": batch_shapes,
                "onnx": onnx_shapes,
                "probes": dynamic_probe_shapes,
                "tested": sorted(set(onnx_shapes + dynamic_probe_shapes)),
                "unit": 1,
                "multiple": 2,
                "last_partial": dynamic_probe_shapes[-1],
            },
            "input": {
                "name": input_meta.name,
                "dtype": input_meta.type,
                "shape": ["N", 1, 48, 48],
                "layout": "NCHW",
            },
            "output": {
                "name": output_meta.name,
                "dtype": output_meta.type,
                "shape": ["N", 4],
                "kind": "scores/logits",
            },
        }
    except ExportValidationError:
        raise
    except Exception as exc:
        raise ExportValidationError(f"ONNX Runtime validation failed: {exc}") from exc


@dataclass(frozen=True)
class _TensorContract:
    shape: tuple[int, ...]
    dtype: str
    scale: float
    zero_point: int


@dataclass(frozen=True)
class _Operator:
    builtin: str
    version: int
    custom_code: str | None = None


@dataclass(frozen=True)
class _TfliteInspection:
    input: _TensorContract
    output: _TensorContract
    operators: tuple[_Operator, ...]
    activation_dtypes: tuple[str, ...]
    arena: Mapping[str, Any] | None = None


class _TfliteAdapter(Protocol):
    def inspect(self, path: Path) -> _TfliteInspection: ...

    def predict(self, path: Path, sample_nhwc: np.ndarray, inspection: _TfliteInspection) -> np.ndarray: ...


def _dtype_name(value: Any) -> str:
    return str(value).replace("torch.", "").replace("numpy.", "").lower()


def _contract(value: Any, where: str) -> _TensorContract:
    if isinstance(value, _TensorContract):
        return value
    mapping = _mapping(value, where) if isinstance(value, Mapping) else None
    try:
        shape = tuple(int(item) for item in (mapping["shape"] if mapping else value.shape))
        dtype = _dtype_name(mapping["dtype"] if mapping else value.dtype)
        scale = float(mapping["scale"] if mapping else value.scale)
        zero = int(mapping.get("zero_point", mapping.get("zero", 0)) if mapping else value.zero_point)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExportValidationError(f"invalid TFLite {where} contract") from exc
    return _TensorContract(shape, dtype, scale, zero)


def _inspection(value: Any) -> _TfliteInspection:
    if isinstance(value, _TfliteInspection):
        return value
    input_value = value.get("input") if isinstance(value, Mapping) else getattr(value, "input", None)
    output_value = value.get("output") if isinstance(value, Mapping) else getattr(value, "output", None)
    raw_ops = value.get("operators", ()) if isinstance(value, Mapping) else getattr(value, "operators", ())
    operators: list[_Operator] = []
    for raw in raw_ops:
        if isinstance(raw, _Operator):
            operators.append(raw)
        elif isinstance(raw, Mapping):
            operators.append(
                _Operator(
                    str(raw.get("builtin", raw.get("op", ""))),
                    int(raw.get("version", 0)),
                    raw.get("custom_code"),
                )
            )
        else:
            operators.append(
                _Operator(
                    str(raw.builtin),
                    int(raw.version),
                    getattr(raw, "custom_code", None),
                )
            )
    activation = (
        value.get("activation_dtypes", ()) if isinstance(value, Mapping) else getattr(value, "activation_dtypes", ())
    )
    arena = value.get("arena") if isinstance(value, Mapping) else getattr(value, "arena", None)
    return _TfliteInspection(
        _contract(input_value, "input"),
        _contract(output_value, "output"),
        tuple(operators),
        tuple(_dtype_name(item) for item in activation),
        arena,
    )


class _RealTfliteAdapter:
    def __init__(self) -> None:
        try:
            from .preflight import _DefaultToolchain

            self._toolchain = _DefaultToolchain()
        except Exception as exc:
            raise ExportValidationError(f"LiteRT adapter unavailable: {exc}") from exc

    def inspect(self, path: Path) -> _TfliteInspection:
        try:
            return _inspection(self._toolchain.inspect(path))
        except Exception as exc:
            if isinstance(exc, ExportValidationError):
                raise
            raise ExportValidationError(f"LiteRT inspection failed: {exc}") from exc

    def predict(self, path: Path, sample_nhwc: np.ndarray, inspection: _TfliteInspection) -> np.ndarray:
        del inspection
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-not-found]
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ExportValidationError(f"LiteRT runtime unavailable: {exc}") from exc
        try:
            interpreter = Interpreter(model_path=str(path))
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            scale, zero = input_detail["quantization"]
            if not math.isfinite(float(scale)) or float(scale) <= 0:
                raise ValueError("input scale is not finite and positive")
            quantized = np.clip(np.rint(sample_nhwc / float(scale) + int(zero)), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_detail["index"], quantized)
            interpreter.invoke()
            output = interpreter.get_tensor(output_detail["index"])
            out_scale, out_zero = output_detail.get("quantization", (1.0, 0))
            return (output.astype(np.float32) - int(out_zero)) * float(out_scale or 1.0)
        except Exception as exc:
            raise ExportValidationError(f"LiteRT inference failed: {exc}") from exc


_TFLITE_ADAPTER: _TfliteAdapter | None = None


def _tflite_adapter() -> _TfliteAdapter:
    return _TFLITE_ADAPTER or _RealTfliteAdapter()


def _validate_tflite_contract(inspection: _TfliteInspection) -> None:
    if inspection.input.shape != (1, 48, 48, 1) or inspection.input.dtype not in {
        "int8",
        "<class 'numpy.int8'>",
    }:
        raise ExportValidationError(f"invalid TFLite input contract: {inspection.input}")
    if inspection.output.shape != (1, 4) or inspection.output.dtype not in {
        "int8",
        "<class 'numpy.int8'>",
    }:
        raise ExportValidationError(f"invalid TFLite output contract: {inspection.output}")
    for name, contract in (("input", inspection.input), ("output", inspection.output)):
        if not math.isfinite(contract.scale) or contract.scale <= 0:
            raise ExportValidationError(f"TFLite {name} scale must be finite and positive")
    if any("float" in dtype or "fp32" in dtype for dtype in inspection.activation_dtypes):
        raise ExportValidationError("TFLite computational graph contains FP32 activations")
    unsupported = [
        operator
        for operator in inspection.operators
        if operator.custom_code is not None
        or operator.builtin not in ALLOWED_BUILTINS
        or operator.version < 1
        or operator.version > MAX_OPERATOR_VERSIONS.get(operator.builtin, 0)
    ]
    if unsupported:
        raise ExportValidationError(f"unsupported TFLite operators: {unsupported}")


def _tflite_predictions(
    run: RunResolution,
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, int]],
    inspection: _TfliteInspection,
    adapter: _TfliteAdapter,
) -> dict[str, Any]:
    actual: list[int] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, label = dataset[index]
            sample = image.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
            output = np.asarray(adapter.predict(run.tflite, sample, inspection), dtype=np.float32)
            if output.shape != (1, 4):
                raise ExportValidationError(f"TFLite inference returned {output.shape}, expected (1, 4)")
            actual.append(label)
            scores.append(output[0])
    values = np.stack(scores)
    torch_actual, torch_logits, _torch_probabilities, _ = _torch_predictions(model, dataset, batch_size=3)
    if not np.array_equal(np.asarray(actual), torch_actual):
        raise ExportValidationError("TFLite and indexed PublicTest labels differ")
    tflite_pred = values.argmax(axis=1)
    torch_pred = torch_logits.argmax(axis=1)
    if not np.array_equal(tflite_pred, torch_pred):
        raise ExportValidationError("PyTorch/TFLite PublicTest class parity failed")
    pytorch_metrics = _metrics(torch_actual, torch_logits, dtype="float32")
    tflite_metrics = _metrics(torch_actual, values, dtype="int8")
    return {
        "partition": "PublicTest",
        "batch_size": 1,
        "samples": len(actual),
        "class_agreement": float(np.mean(tflite_pred == torch_pred)),
        "metrics_pytorch": pytorch_metrics.to_dict(),
        "metrics_tflite": tflite_metrics.to_dict(),
        "accuracy_delta": pytorch_metrics.accuracy - tflite_metrics.accuracy,
    }


def _operator_payload(inspection: _TfliteInspection) -> dict[str, Any]:
    allowlist = sorted({operator.builtin for operator in inspection.operators})
    return {
        "schema": "tflite-operators/v1",
        "operators": [
            {
                "builtin": op.builtin,
                "version": op.version,
                "custom_code": op.custom_code,
            }
            for op in inspection.operators
        ],
        "resolver_allowlist": allowlist,
        "allowlist": allowlist,
        "max_versions": dict(MAX_OPERATOR_VERSIONS),
    }


def _resource_payload(run: RunResolution, model: nn.Module, inspection: _TfliteInspection) -> dict[str, Any]:
    frozen = run.resources
    arena_total = frozen.get("arena_total_bytes")
    arena_margin = frozen.get("arena_with_margin_bytes")
    arena_section = frozen.get("arena")
    if isinstance(arena_section, Mapping):
        arena_total = arena_section.get("total_bytes", arena_total)
        arena_margin = arena_section.get("with_margin_bytes", arena_margin)
    if not isinstance(arena_total, int) or not isinstance(arena_margin, int):
        raise ExportValidationError("frozen resource evidence lacks a complete TFLM arena estimate")
    model_bytes = run.tflite.stat().st_size
    frozen_bytes = frozen.get("tflite_bytes")
    if isinstance(frozen_bytes, int) and frozen_bytes != model_bytes:
        raise ExportValidationError("selected TFLite size differs from frozen resource evidence")
    frozen_macs = frozen.get("macs")
    actual_macs = _macs(model)
    if isinstance(frozen_macs, int) and frozen_macs != actual_macs:
        raise ExportValidationError("selected MAC count differs from frozen resource evidence")
    frozen_operators = frozen.get("operators")
    actual_operators = _operator_payload(inspection)["operators"]
    if isinstance(frozen_operators, list) and frozen_operators and frozen_operators != actual_operators:
        raise ExportValidationError("selected operator evidence differs from the copied TFLite model")
    expected_margin = math.ceil(arena_total * 1.20)
    if arena_margin != expected_margin:
        raise ExportValidationError("frozen arena margin is not exactly 20 percent")
    return {
        "schema": "tflite-resource-metadata/v1",
        "model_bytes": model_bytes,
        "tflite_bytes": model_bytes,
        "macs": actual_macs,
        "operators": actual_operators,
        "resolver_allowlist": sorted({operator.builtin for operator in inspection.operators}),
        "arena_with_margin_bytes": arena_margin,
        "arena": {
            "head_bytes": int((arena_section or {}).get("head_bytes", frozen.get("arena_head_bytes", 0)))
            if isinstance(arena_section, Mapping)
            else int(frozen.get("arena_head_bytes", 0)),
            "temporary_bytes": int((arena_section or {}).get("temporary_bytes", frozen.get("arena_temporary_bytes", 0)))
            if isinstance(arena_section, Mapping)
            else int(frozen.get("arena_temporary_bytes", 0)),
            "tail_bytes": int((arena_section or {}).get("tail_bytes", frozen.get("arena_tail_bytes", 0)))
            if isinstance(arena_section, Mapping)
            else int(frozen.get("arena_tail_bytes", 0)),
            "total_bytes": arena_total,
            "margin": 0.20,
            "with_margin_bytes": arena_margin,
        },
        "frozen_evidence": dict(frozen),
        "inspection_metadata": {"activation_dtypes": list(inspection.activation_dtypes)},
    }


def _write_cpp(stage: Path, payload: bytes) -> None:
    tokens = [f"0x{byte:02x}" for byte in payload]
    lines = [", ".join(tokens[index : index + 12]) for index in range(0, len(tokens), 12)]
    (stage / "emotion_model_data.h").write_text(
        "#pragma once\n#include <cstddef>\n#include <cstdint>\n\nnamespace sentience_train {\nextern const std::uint8_t emotion_model_data[];\nextern const std::size_t emotion_model_data_len;\n}\n",
        encoding="utf-8",
    )
    (stage / "emotion_model_data.cc").write_text(
        '#include "emotion_model_data.h"\n\nnamespace sentience_train {\nalignas(16) const std::uint8_t emotion_model_data[] = {\n    '
        + ",\n    ".join(lines)
        + "\n};\nconst std::size_t emotion_model_data_len = sizeof(emotion_model_data);\n}\n",
        encoding="utf-8",
    )


def _lock_hash() -> str | None:
    root = Path(__file__).resolve().parents[2]
    lock = root / "uv.lock"
    return sha256_file(lock) if lock.is_file() else None


def _identity(run: RunResolution, format: str) -> tuple[str, dict[str, Any]]:
    lock_hash = _lock_hash()
    components = {
        "source_manifest_sha256": run.source_manifest_sha256,
        "format": format,
        "schema": EXPORT_SCHEMA,
        "exporter_revision": EXPORTER_REVISION,
        "uv_lock_sha256": lock_hash,
    }
    return _sha256_bytes(json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")), components


def _payload_hashes(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = str(path.relative_to(directory)).replace(os.sep, "/")
            result[relative] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return result


def _verify_existing(destination: Path, identity: str) -> None:
    manifest_path = destination / "manifest.json"
    if not destination.is_dir() or not manifest_path.is_file():
        raise ExportCollisionError(f"existing export is incomplete: {destination}")
    try:
        manifest = _json(manifest_path)
    except ExportError as exc:
        raise ExportCollisionError(f"existing export has an invalid manifest: {destination}") from exc
    if manifest.get("identity") != identity:
        raise ExportCollisionError(f"existing export belongs to another identity: {destination}")
    manifest_digest = manifest.get("manifest_sha256")
    if not isinstance(manifest_digest, str):
        raise ExportCollisionError(f"existing export manifest has no self-digest: {destination}")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    canonical = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if _sha256_bytes(canonical) != manifest_digest:
        raise ExportCollisionError(f"existing export manifest is tampered: {destination}")
    payloads = _mapping(manifest.get("payloads"), "export manifest.payloads")
    expected_names = set(payloads)
    actual_names = {
        str(path.relative_to(destination)).replace(os.sep, "/")
        for path in destination.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_names != expected_names:
        raise ExportCollisionError(f"existing export is incomplete or has unexpected files: {destination}")
    for relative, info in payloads.items():
        value = _mapping(info, f"export payload {relative}")
        path = destination / relative
        if sha256_file(path) != value.get("sha256") or path.stat().st_size != value.get("bytes"):
            raise ExportCollisionError(f"existing export payload is tampered: {relative}")


def _common_payloads(
    stage: Path,
    run: RunResolution,
    format: str,
    identity: str,
    components: Mapping[str, Any],
) -> None:
    (stage / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    (stage / "preprocessing.json").write_text(
        json.dumps(
            {
                "face": "already_localized",
                "grayscale": True,
                "height": 48,
                "width": 48,
                "augmentation": False,
                "normalized": "pixel / 127.5 - 1.0",
                "normalization_formula": "pixel / 127.5 - 1.0",
                "range": [-1.0, 1.0],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / "input_contract.json").write_text(
        json.dumps(
            {
                "dtype": "float32" if format == "onnx" else "int8",
                "layout": "NCHW" if format == "onnx" else "NHWC",
                "shape": ["N", 1, 48, 48] if format == "onnx" else [1, 48, 48, 1],
                "batch": "dynamic" if format == "onnx" else "fixed",
                "normalization": "pixel / 127.5 - 1.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / "output_contract.json").write_text(
        json.dumps(
            {
                "dtype": "float32" if format == "onnx" else "int8",
                "shape": ["N", 4] if format == "onnx" else [1, 4],
                "kind": "scores/logits",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(
    stage: Path,
    run: RunResolution,
    format: str,
    identity: str,
    components: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    payloads = _payload_hashes(stage)
    document = {
        "schema": EXPORT_SCHEMA,
        "schema_version": EXPORT_SCHEMA,
        "identity": identity,
        "identity_components": dict(components),
        "run_id": run.run_id,
        "format": format,
        "private_test_access": False,
        "provenance": {
            "evidence": "PublicTest",
            "partition": "PublicTest",
            "public_test_index": str(run.public_index.relative_to(run.directory)).replace(os.sep, "/"),
        },
        "source": {
            "run": run.run_id,
            "manifest_sha256": run.source_manifest_sha256,
            "checkpoint": str(run.checkpoint.relative_to(run.directory)).replace(os.sep, "/"),
            "tflite": str(run.tflite.relative_to(run.directory)).replace(os.sep, "/"),
            "width": run.width,
            "seed": run.seed,
            "route": run.route,
            "dataset_sha256": run.manifest["csv"]["sha256"],
            "dataset_counts": run.manifest["csv"].get("partition_counts", {}),
            "configuration": run.manifest["configuration"],
            "git": run.manifest.get("git"),
            "training_lock_sha256": run.manifest.get("uv_lock_sha256"),
            "export_lock_sha256": components.get("uv_lock_sha256"),
            "devices": {"export": "cpu", "platform": platform.platform()},
            "runtimes": {"python": sys.version, "torch": torch.__version__},
        },
        "validation": validation,
        "payloads": payloads,
    }
    unsigned = json.dumps(document, indent=2, sort_keys=True) + "\n"
    document["manifest_sha256"] = _sha256_bytes(unsigned.encode("utf-8"))
    (stage / "manifest.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publish(stage: Path, destination: Path, identity: str) -> bool:
    if destination.exists():
        _verify_existing(destination, identity)
        shutil.rmtree(stage)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(stage, destination)
    except FileExistsError as exc:
        if destination.exists():
            _verify_existing(destination, identity)
            shutil.rmtree(stage)
            return True
        raise ExportCollisionError(f"could not publish export {destination}") from exc
    return False


def _export_single_format_bundle(
    run_id: str,
    format: str,
    runs_root: Path = Path("runs"),
    exports_root: Path = Path("exports"),
) -> ExportResult:
    """Create or reuse one immutable ONNX or TFLite development bundle."""
    if format not in {"onnx", "tflite"}:
        raise ExportInputError("format must be exactly 'onnx' or 'tflite'")
    run = _resolve_frozen_run(run_id, Path(runs_root))
    identity, components = _identity(run, format)
    destination = Path(exports_root).expanduser().resolve() / run_id / format
    if destination.exists():
        _verify_existing(destination, identity)
        return ExportResult(run_id, format, destination, True, identity)
    model = _restore_model(run)
    dataset = _IndexedPublicTest(run.dataset, run.index)
    if len(dataset) == 0:
        raise ExportInputError("frozen PublicTest index is empty")
    exports_base = Path(exports_root).expanduser().resolve()
    stage = exports_base / ".staging" / f"{run_id}-{format}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        _common_payloads(stage, run, format, identity, components)
        if format == "onnx":
            model_path = stage / "emotion_model_fp32.onnx"
            _onnx_adapter().export(model, model_path)
            validation = _validate_onnx(model_path, model, dataset)
        else:
            model_path = stage / "emotion_model_int8.tflite"
            model_path.write_bytes(run.tflite.read_bytes())
            adapter = _tflite_adapter()
            inspection = _inspection(adapter.inspect(model_path))
            _validate_tflite_contract(inspection)
            (stage / "input_contract.json").write_text(
                json.dumps(
                    {
                        "dtype": inspection.input.dtype,
                        "layout": "NHWC",
                        "shape": list(inspection.input.shape),
                        "batch": "fixed",
                        "scale": inspection.input.scale,
                        "zero_point": inspection.input.zero_point,
                        "normalization": "pixel / 127.5 - 1.0",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (stage / "output_contract.json").write_text(
                json.dumps(
                    {
                        "dtype": inspection.output.dtype,
                        "shape": list(inspection.output.shape),
                        "scale": inspection.output.scale,
                        "zero_point": inspection.output.zero_point,
                        "kind": "scores/logits",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            validation = _tflite_predictions(run, model, dataset, inspection, adapter)
            (stage / "operators.json").write_text(
                json.dumps(_operator_payload(inspection), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (stage / "resource_metadata.json").write_text(
                json.dumps(_resource_payload(run, model, inspection), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_cpp(stage, model_path.read_bytes())
        validation = {
            **validation,
            "public_test_index_sha256": sha256_file(run.public_index),
            "private_test_access": False,
        }
        (stage / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_manifest(stage, run, format, identity, components, validation)
        reused = _publish(stage, destination, identity)
        return ExportResult(run_id, format, destination, reused, identity)
    except ExportError:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(stage, ignore_errors=True)
        raise ExportValidationError(f"{format} export failed: {exc}") from exc


def export_frozen_run(
    run_id: str,
    format: str = "all",
    runs_root: Path = Path("runs"),
    exports_root: Path = Path("exports"),
) -> ExportResult:
    """Export a frozen run through the single public development/release seam."""
    if format in {"onnx", "tflite"}:
        return _export_single_format_bundle(run_id, format, runs_root=runs_root, exports_root=exports_root)
    if format != "all":
        raise ExportInputError("format must be exactly 'all', 'onnx', or 'tflite'")
    from ._release import release_frozen_run

    return release_frozen_run(run_id, runs_root=Path(runs_root), exports_root=Path(exports_root))
