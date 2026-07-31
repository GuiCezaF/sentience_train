"""Strict, immutable configuration loading for the training foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


LABELS = ("angry", "happy", "neutral", "sad")


class ConfigError(ValueError):
    """Raised when a configuration document is not part of the public schema."""


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    classes: tuple[str, ...] = LABELS


@dataclass(frozen=True)
class ModelConfig:
    name: str = "minivgg"
    num_classes: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 100
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    scheduler: str = "cosine"


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int = 12
    min_delta: float = 0.002


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 10.0
    translate_fraction: float = 0.08
    scale_min: float = 0.95
    scale_max: float = 1.05
    brightness: float = 0.15
    contrast: float = 0.15


@dataclass(frozen=True)
class RunConfig:
    directory: Path = Path("runs")


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    early_stopping: EarlyStoppingConfig
    augmentation: AugmentationConfig
    run: RunConfig
    source_path: Path

    def resolved_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_path")
        return _paths_to_strings(value)


_SCHEMA: dict[str, set[str]] = {
    "root": {"data", "model", "training", "optimizer", "early_stopping", "augmentation", "run"},
    "data": {"csv_path", "classes"},
    "model": {"name", "num_classes"},
    "training": {"batch_size", "epochs", "seed", "device", "num_workers", "scheduler"},
    "optimizer": {"name", "lr", "weight_decay"},
    "early_stopping": {"patience", "min_delta"},
    "augmentation": {"horizontal_flip_probability", "rotation_degrees", "translate_fraction", "scale_min", "scale_max", "brightness", "contrast"},
    "run": {"directory"},
}


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _paths_to_strings(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, received {type(value).__name__}")
    return value


def _strict_keys(value: Mapping[str, Any], section: str) -> None:
    unknown = set(value) - _SCHEMA[section]
    missing = _SCHEMA[section] - set(value)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"{section}: missing key(s): {', '.join(sorted(missing))}")


def _int(value: Any, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer, received {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, received {value!r}")
    return value


def _float(value: Any, where: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, received {value!r}")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, received {value!r}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{where} must be <= {maximum}, received {value!r}")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string, received {value!r}")
    return value


def _apply_overrides(document: dict[str, Any], overrides: Mapping[str, Any] | None) -> None:
    for dotted_key, value in (overrides or {}).items():
        pieces = dotted_key.split(".")
        target: dict[str, Any] = document
        for piece in pieces[:-1]:
            child = target.get(piece)
            if not isinstance(child, dict):
                raise ConfigError(f"override {dotted_key!r} does not name a configuration value")
            target = child
        if pieces[-1] not in target:
            raise ConfigError(f"override {dotted_key!r} does not name a configuration value")
        target[pieces[-1]] = value


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load and validate one YAML document, resolving paths from its directory."""
    source_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source_path}: {exc}") from exc
    document = _mapping(loaded, "root")
    _apply_overrides(document, overrides)
    _strict_keys(document, "root")
    for section in _SCHEMA["root"]:
        document[section] = _mapping(document[section], section)
        _strict_keys(document[section], section)

    data = document["data"]
    classes = data["classes"]
    if not isinstance(classes, list) or any(not isinstance(item, str) for item in classes) or tuple(classes) != LABELS:
        raise ConfigError(f"data.classes must be exactly {list(LABELS)!r}, received {classes!r}")
    csv_path = Path(_str(data["csv_path"], "data.csv_path"))
    if not csv_path.is_absolute():
        csv_path = source_path.parent / csv_path

    model = document["model"]
    name = _str(model["name"], "model.name")
    if name != "minivgg":
        raise ConfigError(f"model.name must be 'minivgg', received {name!r}")
    num_classes = _int(model["num_classes"], "model.num_classes", 1)
    if num_classes != len(LABELS):
        raise ConfigError(f"model.num_classes must be {len(LABELS)}, received {num_classes}")

    training = document["training"]
    device = _str(training["device"], "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError(f"training.device must be auto, cpu, or cuda, received {device!r}")
    scheduler = _str(training["scheduler"], "training.scheduler")
    if scheduler != "cosine":
        raise ConfigError(f"training.scheduler must be 'cosine', received {scheduler!r}")

    optimizer = document["optimizer"]
    optimizer_name = _str(optimizer["name"], "optimizer.name")
    if optimizer_name != "adamw":
        raise ConfigError(f"optimizer.name must be 'adamw', received {optimizer_name!r}")

    early = document["early_stopping"]
    aug = document["augmentation"]
    scale_min = _float(aug["scale_min"], "augmentation.scale_min", 0.01)
    scale_max = _float(aug["scale_max"], "augmentation.scale_max", scale_min)

    run = document["run"]
    run_dir = Path(_str(run["directory"], "run.directory"))
    if not run_dir.is_absolute():
        run_dir = source_path.parent / run_dir

    return AppConfig(
        data=DataConfig(csv_path=csv_path.resolve(), classes=tuple(classes)),
        model=ModelConfig(name=name, num_classes=num_classes),
        training=TrainingConfig(
            batch_size=_int(training["batch_size"], "training.batch_size", 1),
            epochs=_int(training["epochs"], "training.epochs", 1),
            seed=_int(training["seed"], "training.seed", 0),
            device=device,
            num_workers=_int(training["num_workers"], "training.num_workers", 0),
            scheduler=scheduler,
        ),
        optimizer=OptimizerConfig(
            name=optimizer_name,
            lr=_float(optimizer["lr"], "optimizer.lr", minimum=0.0),
            weight_decay=_float(optimizer["weight_decay"], "optimizer.weight_decay", minimum=0.0),
        ),
        early_stopping=EarlyStoppingConfig(
            patience=_int(early["patience"], "early_stopping.patience", 1),
            min_delta=_float(early["min_delta"], "early_stopping.min_delta", 0.0),
        ),
        augmentation=AugmentationConfig(
            horizontal_flip_probability=_float(aug["horizontal_flip_probability"], "augmentation.horizontal_flip_probability", 0.0, 1.0),
            rotation_degrees=_float(aug["rotation_degrees"], "augmentation.rotation_degrees", 0.0),
            translate_fraction=_float(aug["translate_fraction"], "augmentation.translate_fraction", 0.0, 1.0),
            scale_min=scale_min,
            scale_max=scale_max,
            brightness=_float(aug["brightness"], "augmentation.brightness", 0.0, 1.0),
            contrast=_float(aug["contrast"], "augmentation.contrast", 0.0, 1.0),
        ),
        run=RunConfig(directory=run_dir.resolve()),
        source_path=source_path,
    )
