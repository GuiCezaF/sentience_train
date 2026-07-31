"""Deterministic Training/PublicTest optimization engine."""

from __future__ import annotations

import json
import platform
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import RunStaging, append_jsonl, git_state, sha256_file, utc_now
from .config import AppConfig, LABELS, ModelConfig
from .data import TrainingData, load_training_data
from .models import build_model


@dataclass(frozen=True)
class CompletedRun:
    run_id: str
    directory: Path
    best_epoch: int
    best_macro_f1: float
    best_loss: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device is cuda but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loaders(config: AppConfig, data: TrainingData) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "worker_init_fn": _worker_init,
        "persistent_workers": config.training.num_workers > 0,
    }
    return (
        DataLoader(data.train, shuffle=True, generator=generator, **common),
        DataLoader(data.validation, shuffle=False, **common),
    )


def _class_weights(labels: tuple[int, ...]) -> torch.Tensor:
    counts = Counter(labels)
    values = torch.tensor([1.0 / np.sqrt(counts[index]) for index in range(len(LABELS))], dtype=torch.float32)
    return values / values.mean()


def _validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, list[int]]:
    model.eval()
    total_loss = 0.0
    total = 0
    actual: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            total += labels.size(0)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / total, float(f1_score(actual, predicted, labels=list(range(4)), average="macro", zero_division=0)), predicted


def _manifest(config: AppConfig, data: TrainingData, weights: torch.Tensor, staging: RunStaging, status: str, device: torch.device, artifacts: dict[str, str]) -> dict:
    lock = config.source_path.parent.parent / "uv.lock"
    return {
        "status": status,
        "run_id": staging.run_id,
        "created_at": utc_now(),
        "configuration": config.resolved_dict(),
        "seed": config.training.seed,
        "labels": list(LABELS),
        "label_mapping": {"0": "angry", "3": "happy", "6": "neutral", "4": "sad"},
        "csv": {"path": str(data.provenance.path), "sha256": data.provenance.sha256, "schema": list(data.provenance.header), "release_evidence": data.provenance.release_evidence, "total_rows": data.provenance.total_rows, "partition_counts": data.provenance.partition_counts, "raw_class_counts": data.provenance.raw_class_counts, "selected_class_counts": data.provenance.selected_class_counts},
        "loss_weights": weights.tolist(),
        "git": git_state(config.source_path.parent.parent),
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "device": str(device),
        "artifacts": artifacts,
    }


def run_training(config: AppConfig, model_factory: Callable[[ModelConfig], nn.Module] = build_model) -> CompletedRun:
    """Run a fully deterministic FP32 train/PublicTest-only optimization."""
    seed_everything(config.training.seed)
    device = _device(config.training.device)
    data = load_training_data(config.data, config.augmentation)
    staging = RunStaging.create(config.run.directory)
    staging.write_resolved_config(config.resolved_dict())
    log_path = staging.staging / "logs" / "train.jsonl"
    append_jsonl(log_path, {"event": "run_started", "run_id": staging.run_id})
    try:
        model = model_factory(config.model).to(device)
        weights = _class_weights(data.train_labels)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
        train_loader, validation_loader = _loaders(config, data)
        history_path = staging.staging / "metrics" / "history.jsonl"
        best_f1, best_loss, best_epoch = -1.0, float("inf"), 0
        patience_f1, patience_loss, stale = -1.0, float("inf"), 0
        checkpoint_path = staging.staging / "checkpoints" / "best.pt"
        for epoch in range(1, config.training.epochs + 1):
            model.train()
            train_loss_sum, train_examples = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * labels.size(0)
                train_examples += labels.size(0)
            validation_loss, macro_f1, _predictions = _validate(model, validation_loader, criterion, device)
            scheduler.step()
            row = {"epoch": epoch, "train_loss": train_loss_sum / train_examples, "validation_loss": validation_loss, "macro_f1": macro_f1, "learning_rate": optimizer.param_groups[0]["lr"]}
            append_jsonl(history_path, row)
            append_jsonl(log_path, {"event": "epoch_completed", **row})
            checkpoint_improved = macro_f1 > best_f1 or (macro_f1 == best_f1 and validation_loss < best_loss)
            if checkpoint_improved:
                best_f1, best_loss, best_epoch = macro_f1, validation_loss, epoch
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "macro_f1": macro_f1, "validation_loss": validation_loss, "config": config.resolved_dict()}, checkpoint_path)
            patience_improved = macro_f1 >= patience_f1 + config.early_stopping.min_delta or (macro_f1 == patience_f1 and validation_loss < patience_loss)
            if patience_improved:
                patience_f1, patience_loss, stale = macro_f1, validation_loss, 0
            else:
                stale += 1
                if stale >= config.early_stopping.patience:
                    append_jsonl(log_path, {"event": "early_stopped", "epoch": epoch})
                    break
        selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(selected["model_state_dict"])
        torch.save(selected, checkpoint_path)
        validation = {"best_epoch": best_epoch, "macro_f1": best_f1, "loss": best_loss, "selection": "highest_macro_f1_then_lowest_loss", "restored_checkpoint": "checkpoints/best.pt"}
        validation_path = staging.staging / "metrics" / "validation.json"
        validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        append_jsonl(log_path, {"event": "run_completed", "best_epoch": best_epoch, "macro_f1": best_f1, "validation_loss": best_loss})
        artifacts = {str(path.relative_to(staging.staging)): sha256_file(path) for path in sorted(staging.staging.rglob("*")) if path.is_file()}
        staging.write_manifest(_manifest(config, data, weights, staging, "completed", device, artifacts))
        directory = staging.finalize()
        return CompletedRun(staging.run_id, directory, best_epoch, best_f1, best_loss)
    except Exception as exc:
        append_jsonl(log_path, {"event": "run_failed", "error": type(exc).__name__, "message": str(exc)})
        staging.write_manifest(_manifest(config, data, torch.ones(4), staging, "failed", device, {}))
        raise
