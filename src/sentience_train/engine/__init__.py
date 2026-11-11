"""Training engine seam, deterministic PyTorch loop, and transactional run staging."""

from __future__ import annotations

from .artifacts import RunError, RunStaging
from .trainer import (
    SeedEvidence,
    TorchTrainingEngine,
    TrainingEngine,
    TrainingError,
    TrainingRequest,
    _checkpoint_better,
    _evaluate,
    _freeze_for_classifier_warmup,
    _loader,
    _loss,
    resolve_batch_size,
    resolve_worker_count,
    seed_everything,
)

__all__ = [
    "RunError",
    "RunStaging",
    "SeedEvidence",
    "TorchTrainingEngine",
    "TrainingEngine",
    "TrainingError",
    "TrainingRequest",
    "_checkpoint_better",
    "_evaluate",
    "_freeze_for_classifier_warmup",
    "_loader",
    "_loss",
    "resolve_batch_size",
    "resolve_worker_count",
    "seed_everything",
]
