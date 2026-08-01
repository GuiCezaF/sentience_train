"""The irreversible transaction for a complete frozen-run release.

This module is intentionally private.  Development exports may inspect only
the indexed PublicTest records; this transaction is the sole owner of the
ledger which authorizes a read of PrivateTest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .artifacts import sha256_file
from .config import LABELS, QualityGatesConfig
from .data import CANONICAL_FER2013_SHA256, EMOTION_TO_LABEL
from .preflight import MAX_ARENA_BYTES, MAX_MACS, MAX_TFLITE_BYTES
from .metrics import (
    Metrics,
    evaluate_gates,
    evaluate_resource_gates,
    metrics_from_predictions,
    write_confusion_reports,
)


BUNDLE_SCHEMA = "release-bundle/v1"
ACCESS_SCHEMA = "private-test-access/v1"
METRICS_SCHEMA = "release-metrics/v1"
RELEASE_REVISION = "issue-6-release-v1"
QUALITY_LIMITS: Mapping[str, Any] = {
    "accuracy": 0.70,
    "macro_f1": 0.65,
    "recall": 0.55,
    "max_accuracy_delta": 0.02,
    "tflite_bytes": MAX_TFLITE_BYTES,
    "arena_with_margin_bytes": MAX_ARENA_BYTES,
    "macs": MAX_MACS,
}


class ReleaseGateError(RuntimeError):
    """An artifact or resource gate failed before reserved-test access."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid release JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"release JSON must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload_hashes(directory: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != directory / "manifest.json":
            relative = str(path.relative_to(directory)).replace(os.sep, "/")
            payloads[relative] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return payloads


def _canonical_document(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def candidate_identity(run: Any) -> tuple[str, dict[str, Any]]:
    """Return an identity made only from frozen candidate inputs.

    Exporter revisions and lock files are provenance, not identity inputs.  A
    new exporter therefore cannot turn one reserved candidate into a second
    PrivateTest evaluation.
    """
    selected = run.manifest.get("selected", {})
    components = {
        "schema": BUNDLE_SCHEMA,
        "run_id": run.run_id,
        "frozen_manifest_sha256": run.source_manifest_sha256,
        "checkpoint_sha256": sha256_file(run.checkpoint),
        "tflite_sha256": sha256_file(run.tflite),
        "dataset_sha256": sha256_file(run.dataset),
        "configuration_sha256": _sha256_bytes(_canonical_document(run.manifest.get("configuration", {}))),
        "width": run.width,
        "seed": run.seed,
        "route": run.route,
        "model_name": run.model_name,
        "selected": {
            "checkpoint": selected.get("checkpoint"),
            "tflite": selected.get("tflite"),
        },
    }
    identity = _sha256_bytes(json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return identity, components


def _verify_release_bundle(destination: Path, identity: str) -> dict[str, Any]:
    manifest_path = destination / "manifest.json"
    if not destination.is_dir() or not manifest_path.is_file():
        raise RuntimeError(f"existing release is incomplete: {destination}")
    manifest = _json(manifest_path)
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("format") != "all":
        raise RuntimeError(f"existing release has an incompatible schema: {destination}")
    if manifest.get("candidate_identity") != identity:
        raise RuntimeError(f"existing release belongs to another candidate: {destination}")
    if manifest.get("status") not in {"accepted", "failed"}:
        raise RuntimeError(f"existing release has no final status: {destination}")
    digest = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(digest, str) or _sha256_bytes(_canonical_document(unsigned)) != digest:
        raise RuntimeError(f"existing release manifest is tampered: {destination}")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, Mapping):
        raise RuntimeError(f"existing release has no payload hashes: {destination}")
    actual = {
        str(path.relative_to(destination)).replace(os.sep, "/")
        for path in destination.rglob("*")
        if path.is_file() and path != destination / "manifest.json"
    }
    if actual != set(payloads):
        raise RuntimeError(f"existing release payload set is incomplete or adulterated: {destination}")
    for relative, expected in payloads.items():
        path = destination / str(relative)
        if (
            not isinstance(expected, Mapping)
            or not path.is_file()
            or sha256_file(path) != expected.get("sha256")
            or path.stat().st_size != expected.get("bytes")
        ):
            raise RuntimeError(f"existing release payload is adulterated: {relative}")
    for required in ("onnx", "tflite", "validation", "test", "report.md"):
        if not (destination / required).exists():
            raise RuntimeError(f"existing release is missing {required}")
    return manifest


def _publish(stage: Path, destination: Path, identity: str) -> bool:
    if destination.exists():
        manifest = _verify_release_bundle(destination, identity)
        del manifest
        shutil.rmtree(stage, ignore_errors=True)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(stage, destination)
    except FileExistsError as exc:
        if destination.exists():
            _verify_release_bundle(destination, identity)
            shutil.rmtree(stage, ignore_errors=True)
            return True
        raise RuntimeError(f"could not publish release {destination}") from exc
    return False


def _reserve(ledger: Path) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        ledger.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(f"PrivateTest access is already reserved without a reusable final bundle: {ledger}") from exc
    _write_json(
        ledger / "reservation.json",
        {
            "schema": ACCESS_SCHEMA,
            "status": "reserved",
            "candidate_identity": ledger.name,
            "private_test_access": True,
        },
    )


def _finalize_ledger(ledger: Path, status: str, bundle_manifest_sha256: str, test_accessed: bool) -> None:
    _write_json(
        ledger / "finalization.json",
        {
            "schema": ACCESS_SCHEMA,
            "status": status,
            "candidate_identity": ledger.name,
            "private_test_access": test_accessed,
            "bundle_manifest_sha256": bundle_manifest_sha256,
        },
    )


def _validate_release_configuration(run: Any) -> None:
    configuration = run.manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("frozen run has no configuration")
    gates = configuration.get("quality_gates")
    if isinstance(gates, Mapping):
        for key in (
            "accuracy",
            "macro_f1",
            "recall",
            "max_accuracy_delta",
            "tflite_bytes",
            "arena_with_margin_bytes",
            "macs",
        ):
            if key in gates and gates[key] != QUALITY_LIMITS[key]:
                raise ValueError(f"frozen quality gate {key} is not the release limit")


def _expected_private_counts(run: Any) -> dict[str, int]:
    csv_info = run.manifest.get("csv")
    if not isinstance(csv_info, Mapping):
        raise ValueError("frozen run has no CSV provenance")
    selected = csv_info.get("selected_class_counts")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("PrivateTest"), Mapping):
        raise ValueError("frozen run has no selected PrivateTest counts")
    counts: dict[str, int] = {}
    for label in LABELS:
        value = selected["PrivateTest"].get(label)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"frozen PrivateTest count is invalid for {label}")
        counts[label] = value
    total = csv_info.get("partition_counts", {}).get("PrivateTest")
    if not isinstance(total, int) or total < sum(counts.values()):
        raise ValueError("frozen PrivateTest partition count is invalid")
    return counts


def _load_private_samples(run: Any) -> tuple[tuple[torch.Tensor, int], ...]:
    """Read the four target classes only after the ledger has been reserved."""
    expected = _expected_private_counts(run)
    found = {label: 0 for label in LABELS}
    samples: list[tuple[torch.Tensor, int]] = []
    try:
        with run.dataset.open("rb") as handle:
            handle.readline()
            for line_number, raw in enumerate(handle, 2):
                try:
                    row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
                    if len(row) != 3 or row[2] != "PrivateTest":
                        continue
                    raw_emotion = int(row[0])
                    label = EMOTION_TO_LABEL.get(raw_emotion)
                    if label is None:
                        continue
                    pixels = [int(value) for value in row[1].split()]
                except (
                    UnicodeDecodeError,
                    csv.Error,
                    StopIteration,
                    ValueError,
                    IndexError,
                ) as exc:
                    raise ValueError(f"PrivateTest row {line_number} is invalid") from exc
                if len(pixels) != 2304 or any(pixel < 0 or pixel > 255 for pixel in pixels):
                    raise ValueError(f"PrivateTest row {line_number} violates the frozen pixel contract")
                label_name = LABELS[label]
                found[label_name] += 1
                samples.append(
                    (
                        torch.tensor(pixels, dtype=torch.float32).reshape(1, 48, 48).div(127.5).sub(1.0),
                        label,
                    )
                )
    except OSError as exc:
        raise ValueError(f"cannot read PrivateTest: {exc}") from exc
    if found != expected or len(samples) != sum(expected.values()):
        raise ValueError(f"PrivateTest counts differ from the frozen manifest: found={found}, expected={expected}")
    if sha256_file(run.dataset) != run.manifest.get("csv", {}).get("sha256"):
        raise ValueError("frozen dataset hash changed during PrivateTest access")
    return tuple(samples)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-15)


def _metrics(y_true: Sequence[int], scores: np.ndarray, dtype: str) -> Metrics:
    return metrics_from_predictions(
        y_true,
        scores.argmax(axis=1),
        _softmax(scores),
        dtype=dtype,
        partition="PrivateTest",
    )


def _private_predictions(
    model: Any,
    tflite_path: Path,
    inspection: Any,
    adapter: Any,
    samples: Sequence[tuple[torch.Tensor, int]],
) -> tuple[Metrics, Metrics, list[tuple[int, np.ndarray, np.ndarray]]]:
    reference_scores: list[np.ndarray] = []
    tflite_scores: list[np.ndarray] = []
    actual: list[int] = []
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for image, label in samples:
            reference = np.asarray(model(image.unsqueeze(0)).detach().cpu().numpy()[0], dtype=np.float32)
            sample_nhwc = image.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
            converted = np.asarray(adapter.predict(tflite_path, sample_nhwc, inspection), dtype=np.float32)
            if converted.shape != (1, 4):
                raise ValueError(f"TFLite PrivateTest inference returned {converted.shape}, expected (1, 4)")
            actual.append(label)
            reference_scores.append(reference)
            tflite_scores.append(converted[0])
            rows.append((label, reference, converted[0]))
    reference_values = np.stack(reference_scores)
    tflite_values = np.stack(tflite_scores)
    reference_metrics = _metrics(actual, reference_values, "float32")
    tflite_metrics = _metrics(actual, tflite_values, "int8")
    if not np.array_equal(reference_values.argmax(axis=1), tflite_values.argmax(axis=1)):
        # This is recorded as a gate by the caller; metrics remain available for diagnosis.
        pass
    return reference_metrics, tflite_metrics, rows


def _resource_gates(resource: Mapping[str, Any]) -> dict[str, Any]:
    policy = QualityGatesConfig(
        accuracy=QUALITY_LIMITS["accuracy"],
        macro_f1=QUALITY_LIMITS["macro_f1"],
        recall=QUALITY_LIMITS["recall"],
        max_accuracy_delta=QUALITY_LIMITS["max_accuracy_delta"],
        tflite_bytes=QUALITY_LIMITS["tflite_bytes"],
        arena_with_margin_bytes=QUALITY_LIMITS["arena_with_margin_bytes"],
        macs=QUALITY_LIMITS["macs"],
    )
    return evaluate_resource_gates(resource, policy).to_dict()


def _build_handoffs(run: Any, stage: Path, identity: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    from . import export as development

    model = development._restore_model(run)
    public_dataset = development._IndexedPublicTest(run.dataset, run.index)
    if not len(public_dataset):
        raise ValueError("frozen PublicTest index is empty")
    onnx_stage = stage / "onnx"
    tflite_stage = stage / "tflite"
    onnx_stage.mkdir(parents=True)
    tflite_stage.mkdir(parents=True)
    development._common_payloads(onnx_stage, run, "onnx", identity, {"candidate_identity": identity})
    onnx_path = onnx_stage / "emotion_model_fp32.onnx"
    development._onnx_adapter().export(model, onnx_path)
    onnx_validation = development._validate_onnx(onnx_path, model, public_dataset)
    _write_json(onnx_stage / "validation.json", onnx_validation)
    _write_json(
        onnx_stage / "manifest.json",
        {
            "schema": "development-export-bundle/v1",
            "format": "onnx",
            "private_test_access": False,
            "identity": identity,
            "payloads": _payload_hashes(onnx_stage),
        },
    )

    copied_run = replace(run, tflite=tflite_stage / "emotion_model_int8.tflite")
    copied_run.tflite.write_bytes(run.tflite.read_bytes())
    development._common_payloads(tflite_stage, run, "tflite", identity, {"candidate_identity": identity})
    adapter = development._tflite_adapter()
    inspection = development._inspection(adapter.inspect(copied_run.tflite))
    development._validate_tflite_contract(inspection)
    _write_json(
        tflite_stage / "input_contract.json",
        {
            "dtype": inspection.input.dtype,
            "layout": "NHWC",
            "shape": list(inspection.input.shape),
            "batch": "fixed",
            "scale": inspection.input.scale,
            "zero_point": inspection.input.zero_point,
            "normalization": "pixel / 127.5 - 1.0",
        },
    )
    _write_json(
        tflite_stage / "output_contract.json",
        {
            "dtype": inspection.output.dtype,
            "shape": list(inspection.output.shape),
            "scale": inspection.output.scale,
            "zero_point": inspection.output.zero_point,
            "kind": "scores/logits",
        },
    )
    tflite_validation = development._tflite_predictions(copied_run, model, public_dataset, inspection, adapter)
    _write_json(tflite_stage / "validation.json", tflite_validation)
    _write_json(tflite_stage / "operators.json", development._operator_payload(inspection))
    resource = development._resource_payload(copied_run, model, inspection)
    _write_json(tflite_stage / "resource_metadata.json", resource)
    development._write_cpp(tflite_stage, copied_run.tflite.read_bytes())
    _write_json(
        tflite_stage / "manifest.json",
        {
            "schema": "development-export-bundle/v1",
            "format": "tflite",
            "private_test_access": False,
            "identity": identity,
            "payloads": _payload_hashes(tflite_stage),
        },
    )
    artifact_gates: dict[str, Any] = {
        "onnx_public_parity": {
            "passed": True,
            "observed": {
                "class_agreement": onnx_validation.get("class_agreement"),
                "max_absolute_error": onnx_validation.get("max_absolute_error"),
            },
            "limits": {
                "max_absolute_error": development.ONNX_ATOL,
                "max_relative_error": development.ONNX_RTOL,
            },
            "reasons": [],
        },
        "tflite_public_parity": {
            "passed": tflite_validation.get("class_agreement") == 1.0,
            "observed": {
                "class_agreement": tflite_validation.get("class_agreement"),
                "accuracy_delta": tflite_validation.get("accuracy_delta"),
            },
            "limits": {
                "class_agreement": 1.0,
                "max_accuracy_delta": QUALITY_LIMITS["max_accuracy_delta"],
            },
            "reasons": []
            if tflite_validation.get("class_agreement") == 1.0
            else ["PyTorch/TFLite PublicTest class parity failed"],
        },
        "resources": _resource_gates(resource),
    }
    artifact_gates["passed"] = all(
        value.get("passed", False) for key, value in artifact_gates.items() if key != "passed"
    )
    validation = {
        "schema": "public-test-validation/v1",
        "partition": "PublicTest",
        "onnx": onnx_validation,
        "tflite": tflite_validation,
        "resource": resource,
        "gates": artifact_gates,
    }
    _write_json(stage / "validation" / "summary.json", validation)
    _write_json(
        stage / "validation" / "metrics.json",
        {
            "schema": "public-test-metrics/v1",
            "partition": "PublicTest",
            "reference": onnx_validation.get("metrics_pytorch"),
            "tflite": tflite_validation.get("metrics_tflite"),
        },
    )
    return (
        model,
        adapter,
        inspection,
        {
            "validation": validation,
            "resource": resource,
            "gates": artifact_gates,
            "tflite_path": copied_run.tflite,
        },
    )


def _quality_gates(reference: Metrics, tflite: Metrics) -> dict[str, Any]:
    policy = QualityGatesConfig(
        accuracy=QUALITY_LIMITS["accuracy"],
        macro_f1=QUALITY_LIMITS["macro_f1"],
        recall=QUALITY_LIMITS["recall"],
        max_accuracy_delta=QUALITY_LIMITS["max_accuracy_delta"],
        tflite_bytes=QUALITY_LIMITS["tflite_bytes"],
        arena_with_margin_bytes=QUALITY_LIMITS["arena_with_margin_bytes"],
        macs=QUALITY_LIMITS["macs"],
    )
    reference_gate = evaluate_gates(reference, policy)
    tflite_gate = evaluate_gates(tflite, policy, reference=reference)
    delta = reference.accuracy - tflite.accuracy
    parity = {
        "passed": delta <= policy.max_accuracy_delta,
        "observed": {"accuracy_delta": delta, "class_agreement": None},
        "limits": {"max_accuracy_delta": policy.max_accuracy_delta},
        "reasons": []
        if delta <= policy.max_accuracy_delta
        else [f"accuracy_delta {delta:.6f} > {policy.max_accuracy_delta:.6f}"],
    }
    gates: dict[str, Any] = {
        "reference": reference_gate.to_dict(),
        "tflite": tflite_gate.to_dict(),
        "parity_quality": parity,
    }
    gates["passed"] = all(value["passed"] for value in gates.values())
    return gates


def _write_report(
    path: Path,
    run: Any,
    status: str,
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    gates: Mapping[str, Any],
    diagnostics: Sequence[str],
) -> None:
    lines = [
        "# Release report",
        "",
        f"- Run: `{run.run_id}`",
        f"- Status: **{status}**",
        "",
        "## Validation (PublicTest)",
        "",
        "```json",
        json.dumps(validation, indent=2, sort_keys=True),
        "```",
        "",
        "## Final test (PrivateTest)",
        "",
    ]
    if test.get("accessed"):
        lines.extend(
            [
                "PrivateTest was reserved and evaluated in memory. Predictions are not persisted.",
                "",
                "```json",
                json.dumps(test, indent=2, sort_keys=True),
                "```",
            ]
        )
    else:
        lines.append("PrivateTest was not accessed.")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "```json",
            json.dumps(gates, indent=2, sort_keys=True),
            "```",
        ]
    )
    if diagnostics:
        lines.extend(["", "## Diagnostics", "", *[f"- {item}" for item in diagnostics]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_manifest(
    stage: Path,
    run: Any,
    identity: str,
    components: Mapping[str, Any],
    status: str,
    test_accessed: bool,
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    gates: Mapping[str, Any],
    diagnostics: Sequence[str],
) -> dict[str, Any]:
    access_path = stage / "test" / "access.json"
    metrics_path = stage / "test" / "metrics.json"
    payloads = _payload_hashes(stage)
    document: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA,
        "format": "all",
        "status": status,
        "run_id": run.run_id,
        "candidate_identity": identity,
        "identity_components": dict(components),
        "private_test_access": bool(test_accessed),
        "test_accessed": bool(test_accessed),
        "provenance": {
            "run": run.run_id,
            "checkpoint": str(run.checkpoint.relative_to(run.directory)).replace(os.sep, "/"),
            "checkpoint_sha256": sha256_file(run.checkpoint),
            "tflite": str(run.tflite.relative_to(run.directory)).replace(os.sep, "/"),
            "tflite_sha256": sha256_file(run.tflite),
            "width": run.width,
            "seed": run.seed,
            "route": run.route,
            "configuration": run.manifest.get("configuration"),
            "git_training": run.manifest.get("git"),
            "git_export": {"revision": RELEASE_REVISION, "lock_sha256": _lock_hash()},
            "uv_lock_sha256": _lock_hash(),
            "dataset": {
                "sha256": run.manifest.get("csv", {}).get("sha256"),
                "canonical_sha256": CANONICAL_FER2013_SHA256,
                "counts": run.manifest.get("csv", {}).get("partition_counts"),
                "selected_class_counts": run.manifest.get("csv", {}).get("selected_class_counts"),
            },
            "runtimes": {
                "python": sys.version,
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "devices": {"host": "cpu", "esp32_s3_latency_p95": "pending_external"},
        },
        "validation": dict(validation),
        "test": dict(test),
        "gates": dict(gates),
        "diagnostics": list(diagnostics),
        "metrics": {"path": "test/metrics.json", "sha256": sha256_file(metrics_path)},
        "access_marker": {
            "path": "test/access.json",
            "sha256": sha256_file(access_path),
        },
        "payloads": payloads,
    }
    document["manifest_sha256"] = _sha256_bytes(_canonical_document(document))
    _write_json(stage / "manifest.json", document)
    return document


def _lock_hash() -> str | None:
    root = Path(__file__).resolve().parents[2]
    lock = root / "uv.lock"
    return sha256_file(lock) if lock.is_file() else None


def _finish_stage(
    stage: Path,
    run: Any,
    identity: str,
    components: Mapping[str, Any],
    status: str,
    test_accessed: bool,
    validation: Mapping[str, Any],
    reference: Metrics | None,
    tflite: Metrics | None,
    gates: Mapping[str, Any],
    diagnostics: Sequence[str],
) -> dict[str, Any]:
    test = {
        "schema": METRICS_SCHEMA,
        "partition": "PrivateTest",
        "accessed": bool(test_accessed),
        "reference": None if reference is None else reference.to_dict(),
        "tflite": None if tflite is None else tflite.to_dict(),
    }
    _write_json(stage / "test" / "metrics.json", test)
    _write_json(
        stage / "test" / "access.json",
        {
            "schema": ACCESS_SCHEMA,
            "status": status,
            "partition": "PrivateTest",
            "private_test_access": bool(test_accessed),
            "candidate_identity": identity,
            "counts": None if reference is None else dict(zip(LABELS, reference.support_by_class)),
        },
    )
    write_confusion_reports(stage / "test", {"reference": reference, "tflite": tflite})
    _write_report(stage / "report.md", run, status, validation, test, gates, diagnostics)
    return _write_root_manifest(
        stage,
        run,
        identity,
        components,
        status,
        test_accessed,
        validation,
        test,
        gates,
        diagnostics,
    )


def _failed_result(
    run: Any,
    destination: Path,
    identity: str,
    components: Mapping[str, Any],
    stage: Path,
    validation: Mapping[str, Any],
    gates: Mapping[str, Any],
    diagnostics: Sequence[str],
    test_accessed: bool,
    reference: Metrics | None = None,
    tflite: Metrics | None = None,
    ledger: Path | None = None,
) -> Any:
    from .export import ExportResult

    manifest = _finish_stage(
        stage,
        run,
        identity,
        components,
        "failed",
        test_accessed,
        validation,
        reference,
        tflite,
        gates,
        diagnostics,
    )
    reused = _publish(stage, destination, identity)
    if ledger is not None:
        _finalize_ledger(ledger, "failed", manifest["manifest_sha256"], test_accessed)
    return ExportResult(run.run_id, "all", destination, reused, identity, "failed", test_accessed)


def release_frozen_run(run_id: str, runs_root: Path, exports_root: Path) -> Any:
    from .export import (
        ExportCollisionError,
        ExportInputError,
        ExportResult,
        ExportValidationError,
        _resolve_frozen_run,
    )

    run = _resolve_frozen_run(run_id, runs_root)
    csv_info = run.manifest.get("csv", {})
    if csv_info.get("sha256") != CANONICAL_FER2013_SHA256 or not csv_info.get("release_evidence", False):
        raise ExportInputError("complete release requires the canonical FER2013 SHA-256 and release evidence")
    if _lock_hash() is None:
        raise ExportInputError("complete release requires the repository uv.lock")
    try:
        _validate_release_configuration(run)
        _expected_private_counts(run)
    except ValueError as exc:
        raise ExportInputError(str(exc)) from exc
    identity, components = candidate_identity(run)
    exports_base = Path(exports_root).expanduser().resolve()
    destination = exports_base / run_id / "all"
    if destination.exists():
        try:
            manifest = _verify_release_bundle(destination, identity)
        except RuntimeError as exc:
            raise ExportCollisionError(str(exc)) from exc
        return ExportResult(
            run_id,
            "all",
            destination,
            True,
            identity,
            str(manifest["status"]),
            bool(manifest.get("test_accessed", False)),
        )
    ledger = exports_base / run_id / ".private-test-access" / identity
    if ledger.exists():
        raise ExportCollisionError(
            f"PrivateTest access ledger already exists without a reusable final bundle: {ledger}"
        )
    stage = exports_base / ".staging" / f"{run_id}-all-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        try:
            _model, adapter, inspection, built = _build_handoffs(run, stage, identity)
        except ExportInputError:
            raise
        except Exception as exc:
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True, exist_ok=False)
            return _failed_result(
                run,
                destination,
                identity,
                components,
                stage,
                {},
                {"passed": False, "artifact": {"passed": False, "reasons": [str(exc)]}},
                [f"artifact gate failed before PrivateTest: {type(exc).__name__}: {exc}"],
                False,
            )
        validation = built["validation"]
        gates = built["gates"]
        if not gates.get("passed", False):
            return _failed_result(
                run,
                destination,
                identity,
                components,
                stage,
                validation,
                gates,
                ["one or more artifact/resource gates failed before PrivateTest"],
                False,
            )
        try:
            _reserve(ledger)
        except RuntimeError as exc:
            raise ExportCollisionError(str(exc)) from exc
        try:
            samples = _load_private_samples(run)
            reference, tflite, rows = _private_predictions(_model, built["tflite_path"], inspection, adapter, samples)
            quality = _quality_gates(reference, tflite)
            quality["parity_quality"]["observed"]["class_agreement"] = float(
                np.mean([int(np.argmax(ref) == np.argmax(edge)) for _label, ref, edge in rows])
            )
            quality["parity_quality"]["passed"] = (
                quality["parity_quality"]["passed"] and quality["parity_quality"]["observed"]["class_agreement"] == 1.0
            )
            if quality["parity_quality"]["observed"]["class_agreement"] != 1.0:
                quality["parity_quality"]["reasons"].append("PyTorch/TFLite PrivateTest class parity failed")
            gates = {
                **gates,
                "quality": quality,
                "passed": gates.get("passed", False) and quality.get("passed", False),
            }
            status = "accepted" if gates["passed"] else "failed"
            diagnostics = (
                [] if status == "accepted" else ["quality or final parity gates failed after PrivateTest access"]
            )
            manifest = _finish_stage(
                stage,
                run,
                identity,
                components,
                status,
                True,
                validation,
                reference,
                tflite,
                gates,
                diagnostics,
            )
            reused = _publish(stage, destination, identity)
            _finalize_ledger(ledger, status, manifest["manifest_sha256"], True)
            return ExportResult(run_id, "all", destination, reused, identity, status, True)
        except Exception as exc:
            return _failed_result(
                run,
                destination,
                identity,
                components,
                stage,
                validation,
                {
                    **gates,
                    "passed": False,
                    "post_access": {"passed": False, "reasons": [str(exc)]},
                },
                [f"release failed after PrivateTest reservation: {type(exc).__name__}: {exc}"],
                True,
                ledger=ledger,
            )
    except ExportCollisionError:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    except ExportInputError:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(stage, ignore_errors=True)
        raise ExportValidationError(f"release failed: {exc}") from exc
