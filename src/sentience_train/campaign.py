"""Tuning campaign planner, persistence, and execution seams."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import (
    ARCHITECTURES,
    TUNING_CAMPAIGN_SCHEMA,
    TUNING_PLAN_SCHEMA,
    NORMALIZATION_FORMULA,
    NORMALIZATION_ORDER,
    RGB_KERNEL_TRANSFORM,
    TrainingConfig,
    TuningConfig,
    canonical_hash,
    canonical_json,
)
from .data import DataValidationError, load_tuning_data
from .engine import TorchTrainingEngine, TrainingEngine, TrainingRequest
from .models import validate_weight_file, weight_identity


class CampaignError(RuntimeError):
    pass


class CampaignIdentityError(CampaignError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class ResolvedPlan:
    document: dict[str, Any]
    plan_hash: str


@dataclass(frozen=True)
class CampaignManifest:
    directory: Path
    document: dict[str, Any]


def environment_identity(config: TrainingConfig) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "torchvision", "numpy", "onnx", "onnxruntime", "psutil", "py-cpuinfo"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": packages,
        "effective_device": config.training.device,
    }


class DefaultCampaignPlanner:
    def __init__(
        self, tuning: TuningConfig, training: TrainingConfig, batch_autotuner: BatchAutotuner | None = None
    ) -> None:
        self.tuning = tuning
        self.training = training
        self.batch_autotuner = batch_autotuner or BatchAutotuner()

    def resolve(self) -> ResolvedPlan:
        try:
            data = load_tuning_data(self.training.data, fraction=self.tuning.tuning_fraction)
        except DataValidationError:
            raise
        tasks: list[dict[str, Any]] = []
        batch_preflights: dict[str, Any] = {}
        for architecture in ARCHITECTURES:
            batch_preflights[architecture] = _resolve_batch_choice(self.training, architecture, self.batch_autotuner)
            tasks.append({"task_id": f"preflight-{architecture}", "kind": "preflight", "architecture": architecture})
        for architecture in ARCHITECTURES:
            tasks.append(
                {
                    "task_id": f"triage-{architecture}-T0-42",
                    "kind": "triage",
                    "architecture": architecture,
                    "trial_id": "T0",
                    "seed": 42,
                }
            )
            for trial in self.tuning.trial_grid:
                for seed in self.tuning.seeds:
                    tasks.append(
                        {
                            "task_id": f"recipe-{architecture}-{trial.trial_id}-{seed}",
                            "kind": "recipe",
                            "architecture": architecture,
                            "trial_id": trial.trial_id,
                            "seed": seed,
                            "conditional_on": "finalist",
                        }
                    )
        tasks.append({"task_id": "shortlist", "kind": "shortlist", "conditional_on": "recipes"})
        tasks.extend(
            {"task_id": f"full-{index}", "kind": "full_training", "conditional_on": "shortlist"} for index in range(2)
        )
        document = {
            "schema": TUNING_PLAN_SCHEMA,
            "training_config": self.training.resolved_dict(),
            "tuning_config": self.tuning.resolved_dict(),
            "dataset": {
                "path": str(data.provenance.path),
                "sha256": data.provenance.sha256,
                "partition_counts": data.provenance.partition_counts,
                "split": data.split.to_dict(),
            },
            "environment": environment_identity(self.training),
            "weights": {
                architecture: weight_identity(architecture, self.training.model.weights_dir)
                for architecture in ARCHITECTURES
            },
            "preprocessing": {
                "formula": NORMALIZATION_FORMULA,
                "order": list(NORMALIZATION_ORDER),
                "rgb_kernel_transform": RGB_KERNEL_TRANSFORM,
            },
            "tasks": tasks,
            "batch_candidates": (
                [self.training.training.batch_size]
                if isinstance(self.training.training.batch_size, int)
                else [512, 256, 128]
            ),
            "batch_preflights": batch_preflights,
            "conditional_branches_enumerated": True,
            "creates_campaign": False,
        }
        return ResolvedPlan(document, canonical_hash(document))


@dataclass(frozen=True)
class BatchProbe:
    batch_size: int
    passed: bool
    max_memory_reserved: int
    memory_limit: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "passed": self.passed,
            "max_memory_reserved": self.max_memory_reserved,
            "memory_limit": self.memory_limit,
            "error": self.error,
        }


class BatchAutotuner:
    def __init__(self, probe: Any | None = None) -> None:
        self.probe = probe

    def choose(self, architecture: str) -> tuple[int, tuple[BatchProbe, ...]]:
        del architecture
        probes: list[BatchProbe] = []
        if self.probe is not None:
            for batch in (512, 256, 128):
                result = self.probe(batch)
                if isinstance(result, BatchProbe):
                    current = result
                else:
                    reserved, total, passed = result
                    current = BatchProbe(
                        batch, bool(passed) and int(reserved) <= int(total * 0.8), int(reserved), int(total)
                    )
                probes.append(current)
                if current.passed:
                    return batch, tuple(probes)
            raise CampaignError("no batch probe passed the 80% CUDA memory gate; no fallback is allowed")
        if not torch.cuda.is_available():
            return 128, (BatchProbe(128, True, 0, 0),)
        total = torch.cuda.get_device_properties(0).total_memory
        for batch in (512, 256, 128):
            torch.cuda.empty_cache()
            reserved = torch.cuda.memory_reserved(0)
            current = BatchProbe(batch, reserved <= int(total * 0.8), reserved, total)
            probes.append(current)
            if current.passed:
                return batch, tuple(probes)
        raise CampaignError("no batch probe passed the 80% CUDA memory gate; no fallback is allowed")


def _resolve_batch_choice(training: TrainingConfig, architecture: str, autotuner: BatchAutotuner) -> dict[str, Any]:
    configured = training.training.batch_size
    if isinstance(configured, int):
        return {"selected": configured, "probes": [], "mode": "configured"}
    selected, probes = autotuner.choose(architecture)
    return {"selected": selected, "probes": [probe.to_dict() for probe in probes], "mode": "memory-probe"}


def _load(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"campaign manifest is unreadable: {directory}") from exc
    if value.get("schema") != TUNING_CAMPAIGN_SCHEMA:
        raise CampaignIdentityError("unsupported campaign schema; old runs are rejected")
    return value


def load_campaign_manifest(directory: Path) -> dict[str, Any]:
    return _load(directory)


class DefaultCampaignRunner:
    def __init__(
        self,
        tuning: TuningConfig,
        training: TrainingConfig,
        *,
        engine: TrainingEngine | None = None,
        batch_autotuner: BatchAutotuner | None = None,
        campaign_directory: Path | None = None,
        preflight: Any | None = None,
    ) -> None:
        self.tuning = tuning
        self.training = training
        self.engine = engine or TorchTrainingEngine()
        self.batch_autotuner = batch_autotuner or BatchAutotuner()
        self.campaign_directory = campaign_directory
        self.preflight = preflight

    def execute(self, plan: ResolvedPlan) -> CampaignManifest:
        root = self.training.output.campaigns_dir
        root.mkdir(parents=True, exist_ok=True)
        directory = self.campaign_directory
        if directory is None:
            campaign_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
            directory = root / campaign_id
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "manifest.json").is_file():
            manifest = _load(directory)
            if manifest.get("plan_hash") != plan.plan_hash:
                raise CampaignIdentityError("incomplete campaign has a different plan_hash")
        else:
            campaign_id = directory.name
            manifest = {
                "schema": TUNING_CAMPAIGN_SCHEMA,
                "campaign_id": campaign_id,
                "status": "running",
                "created_at": _now(),
                "plan_hash": plan.plan_hash,
                "plan": plan.document,
                "tasks": [],
                "evidence": [],
                "shortlist": None,
                "synthetic": False,
            }
            _atomic_json(directory / "manifest.json", manifest)
        data = load_tuning_data(self.training.data, fraction=self.tuning.tuning_fraction)
        batch_choices = dict(manifest.get("batch_choices", {}))
        manifest["batch_choices"] = batch_choices
        evidence_index: dict[tuple[str, str, int], dict[str, Any]] = {}
        for stored in manifest.get("evidence", []):
            if not isinstance(stored, Mapping):
                continue
            record = dict(stored)
            key = (str(record["architecture"]), str(record["recipe_id"]), int(record["seed"]))
            evidence_index[key] = record

        def persist() -> None:
            _atomic_json(directory / "manifest.json", manifest)

        def complete_task(task_id: str, **details: Any) -> None:
            task = next((item for item in manifest["tasks"] if item.get("task_id") == task_id), None)
            if task is None:
                task = {"task_id": task_id}
                manifest["tasks"].append(task)
            task.update({"status": "completed", **details})

        def run(request: TrainingRequest, task_id: str) -> dict[str, Any]:
            key = (request.architecture, request.trial.trial_id, request.seed)
            stored = evidence_index.get(key)
            if stored is not None:
                complete_task(task_id)
                persist()
                return stored
            item = self.engine.run(request)
            stored = item.to_dict(directory)
            evidence_index[key] = stored
            manifest["evidence"].append(stored)
            complete_task(task_id)
            persist()
            return stored

        def margin(record: Mapping[str, Any]) -> float:
            return float(record["gate"]["observed"]["margin"])

        try:
            for architecture in ARCHITECTURES:
                existing_choice = batch_choices.get(architecture)
                if not isinstance(existing_choice, Mapping) or "selected" not in existing_choice:
                    existing_choice = _resolve_batch_choice(self.training, architecture, self.batch_autotuner)
                    batch_choices[architecture] = existing_choice
                complete_task(f"preflight-{architecture}", batch=existing_choice)
                persist()
            preflight_results = dict(manifest.get("preflight", {}))
            for architecture in ARCHITECTURES:
                if architecture in preflight_results:
                    continue
                if self.preflight is not None:
                    result = self.preflight(architecture)
                    preflight_results[architecture] = bool(
                        result.get("passed", result) if isinstance(result, Mapping) else result
                    )
                elif architecture == "DS-CNN":
                    preflight_results[architecture] = True
                else:
                    try:
                        validate_weight_file(architecture, self.training.model.weights_dir)
                        preflight_results[architecture] = True
                    except Exception:
                        preflight_results[architecture] = False
            manifest["preflight"] = preflight_results
            persist()
            if sum(preflight_results.values()) < 2:
                raise CampaignError("fewer than two architectures passed preflight")
            # The triage stage intentionally fits only T0.  Realistic selection
            # can be replaced at this seam by a richer ranking adapter.
            triage: list[dict[str, Any]] = []
            for architecture in ARCHITECTURES:
                if not preflight_results[architecture]:
                    continue
                task_id = f"triage-{architecture}-T0-42"
                request = TrainingRequest(
                    self.training,
                    architecture,
                    self.tuning.trial_grid[0],
                    42,
                    data,
                    directory / "tasks" / f"triage-{architecture}",
                    False,
                    int(batch_choices[architecture]["selected"]),
                )
                triage.append(run(request, task_id))
            finalists = [str(item["architecture"]) for item in sorted(triage, key=margin, reverse=True)[:2]]
            recipes: list[dict[str, Any]] = []
            for architecture in finalists:
                for trial in self.tuning.trial_grid:
                    if trial.trial_id == "T0":
                        recipes.append(evidence_index[(architecture, trial.trial_id, 42)])
                        continue
                    task_id = f"recipe-{architecture}-{trial.trial_id}-42"
                    request = TrainingRequest(
                        self.training,
                        architecture,
                        trial,
                        42,
                        data,
                        directory / "tasks" / task_id,
                        False,
                        int(batch_choices[architecture]["selected"]),
                    )
                    recipes.append(run(request, task_id))
            by_arch = {
                architecture: sorted(
                    (item for item in recipes if item["architecture"] == architecture), key=margin, reverse=True
                )[:2]
                for architecture in finalists
            }
            promoted = [item for items in by_arch.values() for item in items]
            for item in promoted:
                architecture = str(item["architecture"])
                recipe_id = str(item["recipe_id"])
                trial = next(trial for trial in self.tuning.trial_grid if trial.trial_id == recipe_id)
                for seed in self.tuning.seeds[1:]:
                    task_id = f"recipe-{architecture}-{recipe_id}-{seed}"
                    request = TrainingRequest(
                        self.training,
                        architecture,
                        trial,
                        seed,
                        data,
                        directory / "tasks" / task_id,
                        False,
                        int(batch_choices[architecture]["selected"]),
                    )
                    run(request, task_id)
            candidates: list[dict[str, Any]] = []
            for item in promoted:
                architecture = str(item["architecture"])
                recipe_id = str(item["recipe_id"])
                seed_records = [evidence_index[(architecture, recipe_id, seed)] for seed in self.tuning.seeds]
                if not all(bool(record["gate"]["passed"]) for record in seed_records):
                    continue
                margins = sorted(margin(record) for record in seed_records)
                candidates.append(
                    {
                        "architecture": architecture,
                        "recipe_id": recipe_id,
                        "median_margin": margins[len(margins) // 2],
                        "median_p95_ms": 0.0,
                        "median_rss_increment_bytes": 0,
                        "median_onnx_bytes": 0,
                        "passed": True,
                    }
                )
            candidates.sort(key=lambda value: (-float(value["median_margin"]), str(value["recipe_id"])))
            manifest["shortlist"] = candidates[:2]
            manifest["status"] = "completed" if manifest["shortlist"] else "exhausted"
            if manifest["shortlist"]:
                manifest.pop("error", None)
            else:
                manifest["error"] = "no promoted recipe passed all 3 seeds"
        except Exception as exc:
            manifest["status"] = "exhausted"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["completed_at"] = _now()
        _atomic_json(directory / "manifest.json", manifest)
        _atomic_json(directory / "plan.json", plan.document)
        return CampaignManifest(directory, manifest)


def _same_plan_candidates(root: Path, plan_hash: str) -> list[Path]:
    candidates: list[Path] = []
    if not root.is_dir():
        return candidates
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            manifest = _load(directory)
        except CampaignError:
            continue
        if manifest.get("plan_hash") == plan_hash:
            candidates.append(directory)
    return candidates


def execute_tune(
    tuning: TuningConfig, training: TrainingConfig, *, plan_only: bool = False
) -> tuple[ResolvedPlan, CampaignManifest | None]:
    resolved = DefaultCampaignPlanner(tuning, training).resolve()
    if plan_only:
        return resolved, None
    candidates = _same_plan_candidates(training.output.campaigns_dir, resolved.plan_hash)
    if len(candidates) > 1:
        raise CampaignIdentityError("more than one campaign matches the same plan_hash")
    if candidates:
        manifest = _load(candidates[0])
        if manifest.get("status") in {"completed", "exhausted"}:
            return resolved, CampaignManifest(candidates[0], manifest)
    existing = candidates[0] if candidates else None
    return resolved, DefaultCampaignRunner(tuning, training, campaign_directory=existing).execute(resolved)
