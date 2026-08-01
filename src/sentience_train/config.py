"""Strict, immutable configuration loading for reproducible selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


LABELS = ("angry", "happy", "neutral", "sad")
WIDTHS = (0.25, 0.35, 0.50)
SEEDS = (42, 43, 44)


class ConfigError(ValueError):
    """Raised when a configuration document is not part of the public schema."""


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    classes: tuple[str, ...] = LABELS


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 100
    device: str = "auto"
    num_workers: int = 0
    scheduler: str = "cosine"


@dataclass(frozen=True)
class SelectionConfig:
    widths: tuple[float, ...] = WIDTHS
    seeds: tuple[int, ...] = SEEDS
    representative: str = "median_int8_macro_f1_then_lowest_seed"


@dataclass(frozen=True)
class QualityGatesConfig:
    accuracy: float = 0.70
    macro_f1: float = 0.65
    recall: float = 0.55
    max_accuracy_delta: float = 0.02
    tflite_bytes: int = 1_048_576
    arena_with_margin_bytes: int = 1_572_864
    macs: int = 15_000_000


@dataclass(frozen=True)
class QATConfig:
    enabled: bool = True
    epochs: int = 20
    lr: float = 1e-5


@dataclass(frozen=True)
class QuantizationConfig:
    calibration_seed: int = 42
    calibration_examples: int = 512
    qat: QATConfig = QATConfig()


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
    selection: SelectionConfig
    quality_gates: QualityGatesConfig
    quantization: QuantizationConfig
    optimizer: OptimizerConfig
    early_stopping: EarlyStoppingConfig
    augmentation: AugmentationConfig
    run: RunConfig
    source_path: Path

    def resolved_dict(self) -> dict[str, Any]:
        """Serialize the effective public policy."""
        value = asdict(self)
        value.pop("source_path", None)
        return _paths_to_strings(value)


_SCHEMA: dict[str, set[str]] = {
    "root": {
        "data",
        "model",
        "training",
        "selection",
        "quality_gates",
        "quantization",
        "optimizer",
        "early_stopping",
        "augmentation",
        "run",
    },
    "data": {"csv_path", "classes"},
    "model": {"num_classes"},
    "training": {"batch_size", "epochs", "device", "num_workers", "scheduler"},
    "selection": {"widths", "seeds", "representative"},
    "quality_gates": {
        "accuracy",
        "macro_f1",
        "recall",
        "max_accuracy_delta",
        "tflite_bytes",
        "arena_with_margin_bytes",
        "macs",
    },
    "quantization": {"calibration_seed", "calibration_examples", "qat"},
    "qat": {"enabled", "epochs", "lr"},
    "optimizer": {"name", "lr", "weight_decay"},
    "early_stopping": {"patience", "min_delta"},
    "augmentation": {
        "horizontal_flip_probability",
        "rotation_degrees",
        "translate_fraction",
        "scale_min",
        "scale_max",
        "brightness",
        "contrast",
    },
    "run": {"directory"},
}


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_paths_to_strings(item) for item in value]
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, received {type(value).__name__}")
    return value


def _strict_keys(value: Mapping[str, Any], section: str, *, required: set[str] | None = None) -> None:
    unknown = set(value) - _SCHEMA[section]
    missing = (required or set()) - set(value)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"{section}: missing key(s): {', '.join(sorted(missing))}")


def _int(value: Any, where: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer, received {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, received {value!r}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{where} must be <= {maximum}, received {value!r}")
    return value


def _float(value: Any, where: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, received {value!r}")
    converted = float(value)
    if minimum is not None and converted < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, received {value!r}")
    if maximum is not None and converted > maximum:
        raise ConfigError(f"{where} must be <= {maximum}, received {value!r}")
    return converted


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string, received {value!r}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be boolean, received {value!r}")
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


def _defaults(document: dict[str, Any]) -> None:
    defaults: dict[str, dict[str, Any]] = {
        "selection": {
            "widths": list(WIDTHS),
            "seeds": list(SEEDS),
            "representative": "median_int8_macro_f1_then_lowest_seed",
        },
        "quality_gates": asdict(QualityGatesConfig()),
        "quantization": {
            "calibration_seed": 42,
            "calibration_examples": 512,
            "qat": asdict(QATConfig()),
        },
    }
    for section, values in defaults.items():
        current = document.setdefault(section, {})
        if not isinstance(current, dict):
            continue
        for key, value in values.items():
            current.setdefault(key, value)
    document["quantization"].setdefault("qat", asdict(QATConfig()))


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
    _defaults(document)
    _apply_overrides(document, overrides)
    _strict_keys(document, "root")
    required = {
        "data": {"csv_path", "classes"},
        "model": {"num_classes"},
        "training": {"batch_size", "epochs", "device", "num_workers", "scheduler"},
        "optimizer": {"name", "lr", "weight_decay"},
        "early_stopping": {"patience", "min_delta"},
        "augmentation": {
            "horizontal_flip_probability",
            "rotation_degrees",
            "translate_fraction",
            "scale_min",
            "scale_max",
            "brightness",
            "contrast",
        },
        "run": {"directory"},
        "selection": {"widths", "seeds", "representative"},
        "quality_gates": set(_SCHEMA["quality_gates"]),
        "quantization": {"calibration_seed", "calibration_examples", "qat"},
    }
    for section in _SCHEMA["root"]:
        document[section] = _mapping(document[section], section)
        _strict_keys(document[section], section, required=required.get(section))
    document["quantization"]["qat"] = _mapping(document["quantization"]["qat"], "qat")
    _strict_keys(document["quantization"]["qat"], "qat", required=set(_SCHEMA["qat"]))

    data = document["data"]
    classes = data["classes"]
    if not isinstance(classes, list) or any(not isinstance(item, str) for item in classes) or tuple(classes) != LABELS:
        raise ConfigError(f"data.classes must be exactly {list(LABELS)!r}, received {classes!r}")
    csv_path = Path(_str(data["csv_path"], "data.csv_path"))
    if not csv_path.is_absolute():
        csv_path = source_path.parent / csv_path

    model = document["model"]
    num_classes = _int(model["num_classes"], "model.num_classes", 1)
    if num_classes != len(LABELS):
        raise ConfigError(f"model.num_classes must be {len(LABELS)}, received {num_classes}")

    training = document["training"]
    device = _str(training["device"], "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError(f"training.device must be auto, cpu, or cuda, received {device!r}")
    scheduler = _str(training["scheduler"], "training.scheduler")
    if scheduler != "cosine":
        raise ConfigError("training.scheduler must be 'cosine'")
    _validate_selection(document["selection"])
    gates = _parse_gates(document["quality_gates"])
    quantization = _parse_quantization(document["quantization"])

    optimizer = document["optimizer"]
    optimizer_name = _str(optimizer["name"], "optimizer.name")
    if optimizer_name != "adamw":
        raise ConfigError("optimizer.name must be 'adamw'")
    early = document["early_stopping"]
    aug = document["augmentation"]
    scale_min = _float(aug["scale_min"], "augmentation.scale_min", 0.01)
    scale_max = _float(aug["scale_max"], "augmentation.scale_max", scale_min)
    if scale_max < scale_min:
        raise ConfigError("augmentation.scale_max must be >= augmentation.scale_min")
    run_dir = Path(_str(document["run"]["directory"], "run.directory"))
    if not run_dir.is_absolute():
        run_dir = source_path.parent / run_dir

    return AppConfig(
        data=DataConfig(csv_path=csv_path.resolve(), classes=tuple(classes)),
        model=ModelConfig(num_classes=num_classes),
        training=TrainingConfig(
            batch_size=_int(training["batch_size"], "training.batch_size", 1),
            epochs=_int(training["epochs"], "training.epochs", 1, 100),
            device=device,
            num_workers=_int(training["num_workers"], "training.num_workers", 0),
            scheduler=scheduler,
        ),
        selection=SelectionConfig(**_selection_values(document["selection"])),
        quality_gates=gates,
        quantization=quantization,
        optimizer=OptimizerConfig(
            name=optimizer_name,
            lr=_float(optimizer["lr"], "optimizer.lr", 0.0),
            weight_decay=_float(optimizer["weight_decay"], "optimizer.weight_decay", 0.0),
        ),
        early_stopping=EarlyStoppingConfig(
            patience=_int(early["patience"], "early_stopping.patience", 1),
            min_delta=_float(early["min_delta"], "early_stopping.min_delta", 0.0),
        ),
        augmentation=AugmentationConfig(
            horizontal_flip_probability=_float(
                aug["horizontal_flip_probability"],
                "augmentation.horizontal_flip_probability",
                0.0,
                1.0,
            ),
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


def _validate_selection(value: Mapping[str, Any]) -> None:
    widths = value["widths"]
    seeds = value["seeds"]
    if not isinstance(widths, list) or tuple(float(item) for item in widths) != WIDTHS:
        raise ConfigError(f"selection.widths must be exactly {list(WIDTHS)!r}")
    if not isinstance(seeds, list) or tuple(seeds) != SEEDS:
        raise ConfigError(f"selection.seeds must be exactly {list(SEEDS)!r}")
    if value["representative"] != "median_int8_macro_f1_then_lowest_seed":
        raise ConfigError("selection.representative is fixed to median_int8_macro_f1_then_lowest_seed")


def _selection_values(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "widths": tuple(float(item) for item in value["widths"]),
        "seeds": tuple(int(item) for item in value["seeds"]),
        "representative": value["representative"],
    }


def _parse_gates(value: Mapping[str, Any]) -> QualityGatesConfig:
    gates = QualityGatesConfig(
        accuracy=_float(value["accuracy"], "quality_gates.accuracy", 0.0, 1.0),
        macro_f1=_float(value["macro_f1"], "quality_gates.macro_f1", 0.0, 1.0),
        recall=_float(value["recall"], "quality_gates.recall", 0.0, 1.0),
        max_accuracy_delta=_float(value["max_accuracy_delta"], "quality_gates.max_accuracy_delta", 0.0),
        tflite_bytes=_int(value["tflite_bytes"], "quality_gates.tflite_bytes", 1),
        arena_with_margin_bytes=_int(value["arena_with_margin_bytes"], "quality_gates.arena_with_margin_bytes", 1),
        macs=_int(value["macs"], "quality_gates.macs", 1),
    )
    return gates


def _parse_quantization(value: Mapping[str, Any]) -> QuantizationConfig:
    if _int(value["calibration_seed"], "quantization.calibration_seed", 0) != 42:
        raise ConfigError("quantization.calibration_seed is fixed to 42")
    if _int(value["calibration_examples"], "quantization.calibration_examples", 1) != 512:
        raise ConfigError("quantization.calibration_examples is fixed to 512")
    qat = value["qat"]
    return QuantizationConfig(
        calibration_seed=42,
        calibration_examples=512,
        qat=QATConfig(
            enabled=_bool(qat["enabled"], "quantization.qat.enabled"),
            epochs=_int(qat["epochs"], "quantization.qat.epochs", 1, 20),
            lr=_float(qat["lr"], "quantization.qat.lr", 0.0),
        ),
    )
