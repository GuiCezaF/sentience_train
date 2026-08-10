"""Training engine seam and deterministic PyTorch adapter."""

from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ARCHITECTURES, QualityGatesConfig, TrainingConfig
from .data import TuningData, TrainingData
from .metrics import GateResult, Metrics, evaluate_predictive_gate, metrics_from_predictions
from .models import build_model
from .recipes import RecipeSpec, learning_rate_for, scheduled_learning_rate


class TrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedEvidence:
    architecture: str
    recipe_id: str
    seed: int
    checkpoint: Path
    metrics: Metrics
    gate: GateResult
    best_epoch: int
    environment: dict[str, Any]
    synthetic: bool = False

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        checkpoint = (
            str(self.checkpoint.relative_to(root))
            if root and self.checkpoint.is_relative_to(root)
            else str(self.checkpoint)
        )
        return {
            "architecture": self.architecture,
            "recipe_id": self.recipe_id,
            "seed": self.seed,
            "checkpoint": checkpoint,
            "metrics": self.metrics.to_dict(),
            "gate": self.gate.to_dict(),
            "best_epoch": self.best_epoch,
            "environment": self.environment,
            "synthetic": self.synthetic,
        }


class TrainingEngine(Protocol):
    def run(self, request: "TrainingRequest") -> SeedEvidence: ...


@dataclass(frozen=True)
class TrainingRequest:
    config: TrainingConfig
    architecture: str
    trial: RecipeSpec
    seed: int
    data: TrainingData | TuningData
    output: Path
    full_training: bool = False
    batch_size: int | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_worker_count(requested: int | str, logical_cpus: int | None = None) -> int:
    if requested != "auto":
        return int(requested)
    count = logical_cpus or (os.cpu_count() or 1)
    cap = 4 if platform.system() == "Windows" else 8
    return min(cap, count // 2)


def resolve_batch_size(config: TrainingConfig, architecture: str, probe: Any | None = None) -> int:
    del architecture
    if isinstance(config.training.batch_size, int):
        return config.training.batch_size
    if probe is None:
        return 128
    return int(probe)


def _loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    batch: int,
    workers: int,
    seed: int,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": config.training.device.startswith("cuda"),
        "generator": generator,
    }
    if workers:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **kwargs)


def _loss(trial: RecipeSpec, labels: tuple[int, ...]) -> nn.Module:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=4).float()
    if trial.loss == "balanced_softmax":

        class Balanced(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("log_counts", counts.clamp_min(1.0).log())

            def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return nn.functional.cross_entropy(logits + self.get_buffer("log_counts"), target)

        return Balanced()
    exponent = 0.5 if trial.class_weight_exponent is None else trial.class_weight_exponent
    weights = counts.clamp_min(1.0) ** -float(exponent)
    weights /= weights.mean()
    if trial.loss == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=trial.label_smoothing)
    if trial.loss == "focal":

        class Focal(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cross_entropy = nn.CrossEntropyLoss(weight=weights, reduction="none")
                self.gamma = float(trial.focal_gamma or 2.0)

            def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                value = self.cross_entropy(logits, target)
                return ((1.0 - torch.exp(-value)) ** self.gamma * value).mean()

        return Focal()
    raise TrainingError(f"unknown loss {trial.loss!r}")


def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    loss: nn.Module,
    *,
    partition: str,
) -> Metrics:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            losses.append(float(loss(logits.float(), targets).item()))
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
            predicted.extend(logits.argmax(1).cpu().tolist())
            actual.extend(targets.cpu().tolist())
    return metrics_from_predictions(
        actual,
        predicted,
        np.asarray(probabilities),
        loss=float(np.mean(losses or [0.0])),
        partition=partition,
    )


def _checkpoint_better(candidate: Metrics, incumbent: Metrics | None, gates: QualityGatesConfig) -> bool:
    if incumbent is None:
        return True
    return (
        int(evaluate_predictive_gate(candidate, gates).passed),
        candidate.predictive_margin(gates),
        candidate.macro_f1,
        -candidate.loss,
    ) > (
        int(evaluate_predictive_gate(incumbent, gates).passed),
        incumbent.predictive_margin(gates),
        incumbent.macro_f1,
        -incumbent.loss,
    )


def _freeze_for_classifier_warmup(model: nn.Module, architecture: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    head: nn.Module | None
    if architecture == "DS-CNN":
        candidate = getattr(model, "head", None)
        head = candidate if isinstance(candidate, nn.Module) else None
    elif architecture == "MobileNetV3-Small":
        classifier = getattr(model, "classifier", None)
        if isinstance(classifier, nn.Sequential):
            parameterized = tuple(module for module in classifier.children() if tuple(module.parameters()))
            head = parameterized[-1] if parameterized else None
        else:
            head = classifier if isinstance(classifier, nn.Module) else None
    elif architecture == "ResNet-18":
        candidate = getattr(model, "fc", None)
        head = candidate if isinstance(candidate, nn.Module) else None
    else:
        head = None

    if head is None:
        raise TrainingError(f"could not locate classifier head for {architecture}")
    head_parameters = tuple(head.parameters())
    if not head_parameters:
        raise TrainingError(f"classifier head for {architecture} has no parameters")
    for parameter in head_parameters:
        parameter.requires_grad = True


class TorchTrainingEngine:
    def __init__(self, *, model_factory: Any | None = None) -> None:
        self.model_factory = model_factory or build_model

    def run(self, request: TrainingRequest) -> SeedEvidence:
        if request.architecture not in ARCHITECTURES:
            raise TrainingError(f"unsupported architecture {request.architecture}")
        seed_everything(request.seed)
        config = request.config
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = config.training.tf32
            torch.backends.cudnn.allow_tf32 = config.training.tf32
            torch.backends.cudnn.benchmark = config.training.cudnn_benchmark
        device = torch.device(
            config.training.device if config.training.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        if config.training.device.startswith("cuda") and device.type != "cuda":
            raise TrainingError("CUDA requested but unavailable")
        if isinstance(request.data, TuningData):
            train_dataset, validation = request.data.optimization, request.data.tuning
            labels = request.data.optimization_labels
            validation_partition = "Tuning"
        else:
            train_dataset, validation = request.data.train, request.data.validation
            labels = request.data.train_labels
            validation_partition = "PublicTest"
        train_dataset = train_dataset.with_augmentation(request.trial.augmentation)
        workers = resolve_worker_count(config.training.workers)
        batch = request.batch_size or resolve_batch_size(config, request.architecture)
        train_loader = _loader(train_dataset, batch, workers, request.seed, config, True)
        validation_loader = _loader(validation, batch, workers, request.seed, config, False)
        model = self.model_factory(request.architecture, weights_dir=config.model.weights_dir, num_classes=4)
        model = model.to(device)
        if config.training.channels_last:
            model = getattr(model, "to")(memory_format=torch.channels_last)
        criterion = _loss(request.trial, labels).to(device)
        base_lr = self._lr(request.architecture, request.trial)
        if request.trial.policy == "classifier-only":
            _freeze_for_classifier_warmup(model, request.architecture)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=base_lr,
            weight_decay=config.training.weight_decay,
        )
        epochs = config.training.full_epochs if request.full_training else config.training.triage_epochs
        patience = config.training.full_patience if request.full_training else config.training.triage_patience
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.training.amp)
        best: Metrics | None = None
        best_epoch = 0
        stale = 0
        request.output.mkdir(parents=True, exist_ok=True)
        checkpoint = request.output / "checkpoint.pt"
        history = request.output / "history.jsonl"
        for epoch in range(1, epochs + 1):
            if request.trial.policy == "classifier-only" and epoch == 4:
                for parameter in model.parameters():
                    parameter.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=config.training.weight_decay)
            if request.trial.policy != "classifier-only" or epoch > 3:
                schedule_epoch = epoch - 3 if request.trial.policy == "classifier-only" else epoch
                schedule_epochs = epochs - 3 if request.trial.policy == "classifier-only" else epochs
                warmup_epochs = min(request.trial.warmup_epochs, max(schedule_epochs - 1, 0))
                current_lr = scheduled_learning_rate(
                    base_lr,
                    schedule_epoch,
                    schedule_epochs,
                    warmup_epochs,
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            model.train()
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and config.training.amp
                ):
                    output = model(images)
                    value = criterion(output, targets)
                scaler.scale(value).backward()
                if request.trial.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), request.trial.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            current = _evaluate(model, validation_loader, device, criterion, partition=validation_partition)
            with history.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "metrics": current.to_dict(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if _checkpoint_better(current, best, config.gates):
                best, best_epoch, stale = current, epoch, 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "architecture": request.architecture,
                        "recipe_id": request.trial.trial_id,
                        "seed": request.seed,
                    },
                    checkpoint,
                )
            else:
                stale += 1
                if stale >= patience:
                    break
        if best is None:
            raise TrainingError("training produced no validation evidence")
        return SeedEvidence(
            request.architecture,
            request.trial.trial_id,
            request.seed,
            checkpoint,
            best,
            evaluate_predictive_gate(best, config.gates),
            best_epoch,
            {
                "seed": request.seed,
                "device": str(device),
                "workers": workers,
                "batch_size": batch,
                "amp": config.training.amp,
                "tf32": config.training.tf32,
                "cudnn_benchmark": config.training.cudnn_benchmark,
                "channels_last": config.training.channels_last,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "driver": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None,
                "pytorch": torch.__version__,
                "platform": platform.platform(),
            },
            False,
        )

    @staticmethod
    def _lr(architecture: str, trial: RecipeSpec) -> float:
        return learning_rate_for(architecture, trial)
