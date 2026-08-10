"""Transactional release publisher; this is the only private-test owner."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .benchmark import HostBenchmark, BenchmarkReport
from .config import PRIVATE_TEST_ACCESS_SCHEMA, RELEASE_BUNDLE_SCHEMA, TrainingConfig, canonical_hash, canonical_json
from .data import EMOTION_TO_LABEL, load_training_data
from .models import build_model
from .metrics import metrics_from_predictions
from .onnx_artifact import OnnxArtifact, OnnxEvidence


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseResult:
    run_id: str
    directory: Path
    status: str
    reused: bool
    private_test_accessed: bool
    onnx_sha256: str | None = None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"expected JSON object {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reserve_ledger(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ReleaseError("private-test ledger is already reserved") from exc
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _payloads(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory)).replace(os.sep, "/"): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _verify_bundle(directory: Path, identity: str) -> None:
    manifest = _json(directory / "manifest.json")
    if manifest.get("schema") != RELEASE_BUNDLE_SCHEMA or manifest.get("identity") != identity:
        raise ReleaseError("existing release has an incompatible identity")
    digest = manifest.pop("self_digest", None)
    if not isinstance(digest, str) or hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest() != digest:
        raise ReleaseError("existing release self-digest failed")
    expected = manifest.get("payloads")
    if not isinstance(expected, dict) or set(expected) != set(_payloads(directory)):
        raise ReleaseError("existing release has missing or extra payloads")
    onnx_payloads = [name for name in expected if name.endswith(".onnx")]
    if onnx_payloads != ["model.onnx"]:
        raise ReleaseError("release must contain exactly one ONNX payload")
    for relative, info in expected.items():
        path = directory / relative
        if _sha(path) != info.get("sha256") or path.stat().st_size != info.get("bytes"):
            raise ReleaseError(f"release payload was adulterated: {relative}")


def _resolve_run(run_id: str, root: Path) -> tuple[Path, dict[str, Any]]:
    if not run_id or Path(run_id).name != run_id or Path(run_id).is_absolute() or ".." in Path(run_id).parts:
        raise ReleaseError("run-id must resolve to one direct child of the configured runs directory")
    directory = root / run_id
    manifest = _json(directory / "manifest.json")
    if manifest.get("schema") != "training-run/v2":
        raise ReleaseError("old or unsupported training-run schema is rejected")
    if manifest.get("status") != "completed":
        raise ReleaseError("only completed training runs can be exported")
    checkpoint = directory / str(manifest.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise ReleaseError("training checkpoint is missing")
    return directory, manifest


def _identity(run_id: str, manifest: Mapping[str, Any], checkpoint: Path, dataset: Path) -> str:
    return canonical_hash(
        {
            "schema": RELEASE_BUNDLE_SCHEMA,
            "run_id": run_id,
            "checkpoint_sha256": _sha(checkpoint),
            "dataset_sha256": _sha(dataset),
            "recipe_id": manifest.get("recipe_id"),
            "architecture": manifest.get("architecture"),
        }
    )


def _private_samples(dataset: Path) -> tuple[np.ndarray, list[int]]:
    values: list[list[int]] = []
    labels: list[int] = []
    with dataset.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Usage") == "PrivateTest" and int(row["emotion"]) in EMOTION_TO_LABEL:
                values.append([int(item) for item in row["pixels"].split()])
                labels.append(EMOTION_TO_LABEL[int(row["emotion"])])
    if not values:
        raise ReleaseError("PrivateTest contains no selected samples")
    return np.asarray(values, dtype=np.uint8).reshape(-1, 48, 48, 1), labels


def _private_metrics(model: torch.nn.Module, samples: np.ndarray, actual: list[int]) -> dict[str, Any]:
    # Labels are retained only in this function and aggregate metrics are
    # returned; individual private predictions never cross this seam.
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.from_numpy(samples).float().div(127.5).sub(1.0).permute(0, 3, 1, 2)), dim=1
        ).numpy()
    predicted = probabilities.argmax(1)
    return metrics_from_predictions(
        actual, predicted, probabilities, partition="PrivateTest", dtype="float32"
    ).to_dict()


class ReleasePublisher:
    def __init__(
        self,
        *,
        onnx: OnnxArtifact | None = None,
        benchmark: HostBenchmark | None = None,
        model_factory: Any | None = None,
    ) -> None:
        self.onnx = onnx or OnnxArtifact()
        self.benchmark = benchmark or HostBenchmark()
        self.model_factory = model_factory or build_model

    def publish(self, run_id: str, *, training: TrainingConfig, benchmark_cpu: str | None = None) -> ReleaseResult:
        run, manifest = _resolve_run(run_id, training.output.runs_dir)
        dataset = Path(str(manifest["dataset"]["path"]))
        checkpoint = run / str(manifest["checkpoint"])
        identity = _identity(run_id, manifest, checkpoint, dataset)
        destination = training.output.exports_dir / run_id
        if destination.exists():
            _verify_bundle(destination, identity)
            bundle = _json(destination / "manifest.json")
            return ReleaseResult(
                run_id,
                destination,
                str(bundle["status"]),
                True,
                bool(bundle.get("private_test_accessed")),
                bundle.get("onnx_sha256"),
            )
        ledger = training.output.exports_dir / ".ledger" / f"{identity}.json"
        if ledger.exists():
            raise ReleaseError("private-test ledger is already reserved; release cannot select another candidate")
        stage = training.output.exports_dir / ".staging" / f"{run_id}-{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        private_accessed = False
        try:
            model = self.model_factory(
                manifest["architecture"],
                weights_dir=Path(str(manifest.get("weights_dir", training.model.weights_dir))),
                num_classes=4,
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
            onnx_path = stage / "model.onnx"
            public_data = load_training_data(training.data)
            if (
                public_data.provenance.path.resolve() != dataset.resolve()
                or public_data.provenance.sha256 != manifest["dataset"]["sha256"]
            ):
                raise ReleaseError("training configuration dataset differs from the frozen run identity")
            public_samples = np.stack(
                [
                    np.clip(np.rint(((public_data.validation[index][0].numpy() + 1.0) * 127.5)), 0, 255)
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                    for index in range(min(8, len(public_data.validation)))
                ],
                axis=0,
            )
            evidence: OnnxEvidence = self.onnx.build_and_validate(model, onnx_path, samples=public_samples)
            report: BenchmarkReport = self.benchmark.run(onnx_path, gates=training.gates)
            if not report.passed or onnx_path.stat().st_size > training.gates.max_onnx_bytes:
                self._failure(training, run_id, identity, {"onnx": evidence.to_dict(), "benchmark": report.to_dict()})
                shutil.rmtree(stage, ignore_errors=True)
                return ReleaseResult(run_id, destination, "failed", False, False, evidence.sha256)
            # Only the exact normalized host may authorize the protected read.
            if report.cpu_normalized != "AMD Ryzen 7 5700" or (
                benchmark_cpu is not None and benchmark_cpu != report.cpu_normalized
            ):
                self._failure(
                    training, run_id, identity, {"benchmark": report.to_dict(), "reason": "private host identity"}
                )
                shutil.rmtree(stage, ignore_errors=True)
                return ReleaseResult(run_id, destination, "failed", False, False, evidence.sha256)
            _reserve_ledger(
                ledger,
                {
                    "schema": PRIVATE_TEST_ACCESS_SCHEMA,
                    "identity": identity,
                    "status": "reserved",
                    "created_at": str(uuid.uuid4()),
                },
            )
            samples, private_labels = _private_samples(dataset)
            private_accessed = True
            private_metrics = _private_metrics(model, samples, private_labels)
            private_passed = (
                float(private_metrics.get("accuracy", 0.0)) >= training.gates.accuracy
                and float(private_metrics.get("macro_f1", 0.0)) >= training.gates.macro_f1
                and min(float(value) for value in private_metrics.get("recall_by_class", {}).values())
                >= training.gates.min_recall
            )
            final_status = "accepted" if private_passed else "failed"
            payload = {
                "labels.txt": ("\\n".join(["angry", "happy", "neutral", "sad"]) + "\\n").encode(),
                "onnx-contract.json": (json.dumps(evidence.contract, sort_keys=True, indent=2) + "\\n").encode(),
                "metrics-public.json": (
                    json.dumps(manifest.get("public_metrics", {}), sort_keys=True, indent=2) + "\\n"
                ).encode(),
                "metrics-private.json": (json.dumps(private_metrics, sort_keys=True, indent=2) + "\\n").encode(),
                "benchmark-raw.json": (json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\\n").encode(),
                "benchmark-summary.json": (
                    json.dumps(
                        {
                            "schema": report.schema,
                            "cpu_normalized": report.cpu_normalized,
                            "p50_ms": report.p50_ms,
                            "p95_ms": report.p95_ms,
                            "rss_increment_bytes": report.rss_increment_bytes,
                            "passed": report.passed,
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\\n"
                ).encode(),
                "provenance.json": (
                    json.dumps(
                        {"run_id": run_id, "identity": identity, "onnx_sha256": evidence.sha256},
                        sort_keys=True,
                        indent=2,
                    )
                    + "\\n"
                ).encode(),
                "report.json": (
                    json.dumps({"status": final_status, "private_test_accessed": True}, sort_keys=True, indent=2)
                    + "\\n"
                ).encode(),
            }
            (stage / "model.onnx").write_bytes(onnx_path.read_bytes())
            for relative, content in payload.items():
                (stage / relative).write_bytes(content)
            document = {
                "schema": RELEASE_BUNDLE_SCHEMA,
                "status": final_status,
                "run_id": run_id,
                "identity": identity,
                "private_test_accessed": True,
                "onnx_sha256": evidence.sha256,
                "payloads": _payloads(stage),
            }
            document["self_digest"] = hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
            _write(stage / "manifest.json", document)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
            _write(
                ledger,
                {
                    "schema": PRIVATE_TEST_ACCESS_SCHEMA,
                    "identity": identity,
                    "status": "published",
                    "bundle": str(destination),
                    "manifest_sha256": _sha(destination / "manifest.json"),
                },
            )
            return ReleaseResult(run_id, destination, final_status, False, True, evidence.sha256)
        except Exception as exc:
            if private_accessed and ledger.exists():
                (stage / "report.json").write_text(
                    json.dumps(
                        {"status": "failed", "private_test_accessed": True, "error": f"{type(exc).__name__}: {exc}"},
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                failed = {
                    "schema": RELEASE_BUNDLE_SCHEMA,
                    "status": "failed",
                    "run_id": run_id,
                    "identity": identity,
                    "private_test_accessed": True,
                    "onnx_sha256": _sha(stage / "model.onnx"),
                    "payloads": _payloads(stage),
                }
                failed["self_digest"] = hashlib.sha256(canonical_json(failed).encode("utf-8")).hexdigest()
                _write(stage / "manifest.json", failed)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage, destination)
                _write(
                    ledger,
                    {
                        "schema": PRIVATE_TEST_ACCESS_SCHEMA,
                        "identity": identity,
                        "status": "published",
                        "bundle": str(destination),
                        "manifest_sha256": _sha(destination / "manifest.json"),
                    },
                )
                return ReleaseResult(run_id, destination, "failed", False, True, failed["onnx_sha256"])
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @staticmethod
    def _failure(training: TrainingConfig, run_id: str, identity: str, diagnostic: Mapping[str, Any]) -> None:
        path = training.output.exports_dir / ".failures" / f"{run_id}-{identity}.json"
        if path.exists():
            return
        _write(
            path,
            {
                "schema": RELEASE_BUNDLE_SCHEMA,
                "status": "failed",
                "run_id": run_id,
                "identity": identity,
                "diagnostic": dict(diagnostic),
            },
        )
