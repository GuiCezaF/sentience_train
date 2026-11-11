"""Deep training module: campaign selection, execution, and atomic run publication."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..engine.artifacts import RunStaging
from .campaign import CampaignError, load_campaign_manifest
from ..core.config import (
    NORMALIZATION_FORMULA,
    NORMALIZATION_ORDER,
    RGB_KERNEL_TRANSFORM,
    SEEDS,
    TRAINING_RUN_SCHEMA,
    ConfigError,
    TrainingConfig,
)
from ..core.data import TrainingData, load_training_data
from ..engine.trainer import (
    SeedEvidence,
    TorchTrainingEngine,
    TrainingEngine,
    TrainingRequest,
)
from ..core.metrics import order_candidates
from ..core.recipes import default_recipe_grid


class TrainingPipelineError(RuntimeError):
    """Raised when a valid training workflow cannot produce a publishable run."""


class CampaignExhaustedError(TrainingPipelineError):
    """Raised when tuning ended normally without a recipe that passed every seed."""


def _select_campaign(config: TrainingConfig) -> tuple[Path, dict[str, Any]]:
    completed: list[tuple[Path, dict[str, Any]]] = []
    exhausted: list[tuple[Path, dict[str, Any]]] = []
    directories = sorted(config.output.campaigns_dir.iterdir()) if config.output.campaigns_dir.is_dir() else ()
    for directory in directories:
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            manifest = load_campaign_manifest(directory)
        except CampaignError:
            continue
        if manifest.get("status") == "completed" and manifest.get("shortlist"):
            completed.append((directory, manifest))
        elif manifest.get("status") == "exhausted":
            exhausted.append((directory, manifest))
    if len(completed) == 1:
        return completed[0]
    if not completed and len(exhausted) == 1:
        directory, manifest = exhausted[0]
        reason = manifest.get("error") or "no recipe met all predictive gates"
        raise CampaignExhaustedError(f"tuning campaign {directory.name} is exhausted: {reason}")
    raise ConfigError("train requires exactly one completed tuning campaign with a shortlist")


def execute_training(
    config: TrainingConfig,
    *,
    engine: TrainingEngine | None = None,
    data: TrainingData | None = None,
) -> Path:
    """Train the frozen shortlist and atomically publish one immutable run.

    The interface accepts resolved configuration plus optional local adapters.
    Production uses the PyTorch and FER2013 adapters; tests use in-memory data
    and a deterministic engine through the same seam.
    """
    campaign_directory, campaign = _select_campaign(config)
    training_data = data if data is not None else load_training_data(config.data)
    training_engine = engine if engine is not None else TorchTrainingEngine()
    recipes = {recipe.trial_id: recipe for recipe in default_recipe_grid()}
    batch_choices = campaign.get("batch_choices", {})
    transaction = RunStaging.create(config.output.runs_dir)
    candidate_records: list[SeedEvidence] = []
    try:
        transaction.write_resolved_config(config.resolved_dict())
        for candidate in campaign["shortlist"]:
            architecture = str(candidate["architecture"])
            recipe_id = str(candidate["recipe_id"])
            try:
                recipe = recipes[recipe_id]
            except KeyError as exc:
                raise ConfigError(f"campaign references unknown recipe {recipe_id!r}") from exc
            configured_batch = config.training.batch_size if isinstance(config.training.batch_size, int) else 128
            batch_size = int(batch_choices.get(architecture, {}).get("selected", configured_batch))
            for seed in SEEDS:
                candidate_records.append(
                    training_engine.run(
                        TrainingRequest(
                            config,
                            architecture,
                            recipe,
                            seed,
                            training_data,
                            transaction.staging / "candidates" / f"{architecture}-{recipe_id}-{seed}",
                            True,
                            batch_size,
                        )
                    )
                )
        if not candidate_records:
            raise ConfigError("campaign shortlist did not contain trainable recipes")
        if any(item.synthetic for item in candidate_records):
            raise TrainingPipelineError("synthetic evidence cannot produce a training run")

        by_recipe: dict[tuple[str, str], list[SeedEvidence]] = {}
        for evidence in candidate_records:
            by_recipe.setdefault((evidence.architecture, evidence.recipe_id), []).append(evidence)
        summaries: list[dict[str, Any]] = []
        for (architecture, recipe_id), items in by_recipe.items():
            margins = sorted(item.metrics.predictive_margin(config.gates) for item in items)
            summaries.append(
                {
                    "architecture": architecture,
                    "recipe_id": recipe_id,
                    "median_margin": margins[len(margins) // 2],
                    "median_p95_ms": 0.0,
                    "median_rss_increment_bytes": 0,
                    "median_onnx_bytes": 0,
                    "passed": all(item.gate.passed for item in items),
                    "seeds": [item.to_dict(transaction.staging) for item in items],
                }
            )
        ordered = order_candidates(summaries, limit=1)
        if not ordered or not ordered[0]["passed"]:
            raise TrainingPipelineError("all shortlisted recipes failed full-training gates")
        winner = ordered[0]
        winner_items = sorted(
            by_recipe[(str(winner["architecture"]), str(winner["recipe_id"]))],
            key=lambda item: item.metrics.predictive_margin(config.gates),
        )
        representative = winner_items[len(winner_items) // 2]
        transaction.stage_checkpoint(representative.checkpoint)
        manifest = {
            "schema": TRAINING_RUN_SCHEMA,
            "status": "completed",
            "run_id": transaction.run_id,
            "architecture": representative.architecture,
            "recipe_id": representative.recipe_id,
            "seed": representative.seed,
            "checkpoint": "checkpoint.pt",
            "dataset": {
                "path": str(training_data.provenance.path),
                "sha256": training_data.provenance.sha256,
                "partition_counts": training_data.provenance.partition_counts,
            },
            "public_metrics": representative.metrics.to_dict(),
            "preprocessing": {
                "formula": NORMALIZATION_FORMULA,
                "order": list(NORMALIZATION_ORDER),
                "rgb_kernel_transform": RGB_KERNEL_TRANSFORM,
            },
            "candidates": summaries,
            "campaign": {
                "directory": str(campaign_directory),
                "manifest_sha256": hashlib.sha256((campaign_directory / "manifest.json").read_bytes()).hexdigest(),
            },
            "synthetic": False,
        }
        transaction.write_manifest(manifest)
        return transaction.finalize()
    except BaseException:
        transaction.abort()
        raise


__all__ = [
    "CampaignExhaustedError",
    "TrainingPipelineError",
    "execute_training",
]
