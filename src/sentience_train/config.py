"""Public immutable configuration contracts for the ONNX-only pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .recipes import AugmentationConfig, RecipeSpec, default_recipe_grid

LABELS = ("angry", "happy", "neutral", "sad")
SEEDS = (42, 43, 44)
ARCHITECTURES = ("DS-CNN", "MobileNetV3-Small", "ResNet-18")
NORMALIZATION_FORMULA = "float32(pixel) / 127.5 - 1.0"
NORMALIZATION_ORDER = ("cast_float32", "divide_127.5", "subtract_1.0")
RGB_KERNEL_TRANSFORM = "rgb-kernel-sum/v1"

TRAINING_CONFIG_SCHEMA = "training-config/v2"
TUNING_CONFIG_SCHEMA = "tuning-config/v2"
TUNING_PLAN_SCHEMA = "tuning-plan/v2"
TUNING_CAMPAIGN_SCHEMA = "tuning-campaign/v2"
TRAINING_RUN_SCHEMA = "training-run/v2"
RELEASE_BUNDLE_SCHEMA = "release-bundle/v2"
RELEASE_METRICS_SCHEMA = "release-metrics/v2"
PRIVATE_TEST_ACCESS_SCHEMA = "private-test-access/v2"
ONNX_CONTRACT_SCHEMA = "onnx-contract/v1"
HOST_BENCHMARK_SCHEMA = "host-benchmark/v1"
TUNING_SPLIT_SCHEMA = "tuning-split/v1"


class ConfigError(ValueError):
    """Raised for invalid, unsupported, or identity-incompatible input."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _paths(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    classes: tuple[str, ...] = LABELS


@dataclass(frozen=True)
class ModelConfig:
    architectures: tuple[str, ...] = ARCHITECTURES
    weights_dir: Path = Path("models/pretrained")
    num_classes: int = 4


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int | str = "auto"
    workers: int | str = "auto"
    device: str = "cuda:0"
    amp: bool = True
    tf32: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = True
    optimizer: str = "AdamW"
    weight_decay: float = 1e-4
    triage_epochs: int = 60
    triage_patience: int = 10
    full_epochs: int = 100
    full_patience: int = 15


@dataclass(frozen=True)
class QualityGatesConfig:
    accuracy: float = 0.70
    macro_f1: float = 0.65
    min_recall: float = 0.55
    max_onnx_bytes: int = 52_428_800
    max_p95_ms: float = 10.0
    max_rss_increment_bytes: int = 134_217_728
    parity_atol: float = 1e-5
    parity_rtol: float = 1e-4


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: Path = Path("artifacts/runs")
    campaigns_dir: Path = Path("artifacts/campaigns")
    exports_dir: Path = Path("artifacts/exports")


@dataclass(frozen=True)
class TrainingConfig:
    schema: str
    data: DataConfig
    model: ModelConfig
    training: TrainingSettings
    gates: QualityGatesConfig
    output: OutputConfig
    source_path: Path

    def resolved_dict(self) -> dict[str, Any]:
        return _paths(
            {
                "schema": self.schema,
                "data": asdict(self.data),
                "model": asdict(self.model),
                "training": asdict(self.training),
                "gates": asdict(self.gates),
                "output": asdict(self.output),
            }
        )


@dataclass(frozen=True)
class TuningConfig:
    schema: str
    training_config: Path
    architectures: tuple[str, ...]
    seeds: tuple[int, ...]
    trial_grid: tuple[RecipeSpec, ...]
    tuning_fraction: float
    source_path: Path

    def resolved_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "training_config": str(self.training_config),
            "architectures": list(self.architectures),
            "seeds": list(self.seeds),
            "trial_grid": [item.to_dict() for item in self.trial_grid],
            "tuning_fraction": self.tuning_fraction,
        }


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be an object")
    return value


def _strict(value: Mapping[str, Any], allowed: set[str], where: str, required: set[str] | None = None) -> None:
    unknown = set(value) - allowed
    missing = (required or set()) - set(value)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"{where}: missing key(s): {', '.join(sorted(missing))}")


def _int(value: Any, where: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{where} must be <= {maximum}")
    return value


def _number(value: Any, where: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < minimum:
        raise ConfigError(f"{where} must be a number >= {minimum}")
    return float(value)


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be boolean")
    return value


def _path(value: Any, where: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _read(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read configuration {source}: {exc}") from exc
    return source, _mapping(loaded, "root")


def load_training_config(path: str | Path) -> TrainingConfig:
    source, document = _read(path)
    _strict(
        document,
        {"schema", "data", "model", "training", "gates", "output"},
        "root",
        {"schema", "data", "model", "training", "gates", "output"},
    )
    if document["schema"] != TRAINING_CONFIG_SCHEMA:
        raise ConfigError(f"expected {TRAINING_CONFIG_SCHEMA}, received {document['schema']!r}")

    data = _mapping(document["data"], "data")
    _strict(data, {"csv_path", "classes"}, "data", {"csv_path", "classes"})
    classes = tuple(data["classes"]) if isinstance(data["classes"], list) else ()
    if classes != LABELS:
        raise ConfigError(f"data.classes must be exactly {list(LABELS)!r}")

    model = _mapping(document["model"], "model")
    _strict(model, {"architectures", "weights_dir", "num_classes"}, "model")
    architectures = tuple(model.get("architectures", ARCHITECTURES))
    if (
        not architectures
        or any(item not in ARCHITECTURES for item in architectures)
        or len(set(architectures)) != len(architectures)
    ):
        raise ConfigError(f"model.architectures must be a unique subset of {list(ARCHITECTURES)!r}")
    num_classes = _int(model.get("num_classes", 4), "model.num_classes", 4, 4)

    training = _mapping(document["training"], "training")
    _strict(
        training,
        {
            "batch_size",
            "workers",
            "device",
            "amp",
            "tf32",
            "cudnn_benchmark",
            "channels_last",
            "optimizer",
            "weight_decay",
            "triage_epochs",
            "triage_patience",
            "full_epochs",
            "full_patience",
        },
        "training",
    )
    batch = training.get("batch_size", "auto")
    if batch != "auto":
        batch = _int(batch, "training.batch_size", 1)
    workers = training.get("workers", "auto")
    if workers != "auto":
        workers = _int(workers, "training.workers", 0)
    device = training.get("device", "cuda:0")
    if device not in {"cuda:0", "cpu"}:
        raise ConfigError("training.device must be 'cuda:0' or 'cpu'")
    optimizer = training.get("optimizer", "AdamW")
    if optimizer != "AdamW":
        raise ConfigError("training.optimizer must be AdamW")
    settings = TrainingSettings(
        batch,
        workers,
        device,
        _flag(training.get("amp", True), "training.amp"),
        _flag(training.get("tf32", True), "training.tf32"),
        _flag(training.get("cudnn_benchmark", True), "training.cudnn_benchmark"),
        _flag(training.get("channels_last", True), "training.channels_last"),
        optimizer,
        _number(training.get("weight_decay", 1e-4), "training.weight_decay"),
        _int(training.get("triage_epochs", 60), "training.triage_epochs", 1, 60),
        _int(training.get("triage_patience", 10), "training.triage_patience", 1, 10),
        _int(training.get("full_epochs", 100), "training.full_epochs", 1, 100),
        _int(training.get("full_patience", 15), "training.full_patience", 1, 15),
    )

    gates = _mapping(document["gates"], "gates")
    _strict(
        gates,
        {
            "accuracy",
            "macro_f1",
            "min_recall",
            "max_onnx_bytes",
            "max_p95_ms",
            "max_rss_increment_bytes",
            "parity_atol",
            "parity_rtol",
        },
        "gates",
    )
    gate_config = QualityGatesConfig(
        _number(gates.get("accuracy", 0.70), "gates.accuracy"),
        _number(gates.get("macro_f1", 0.65), "gates.macro_f1"),
        _number(gates.get("min_recall", 0.55), "gates.min_recall"),
        _int(gates.get("max_onnx_bytes", 52_428_800), "gates.max_onnx_bytes", 1),
        _number(gates.get("max_p95_ms", 10.0), "gates.max_p95_ms"),
        _int(gates.get("max_rss_increment_bytes", 134_217_728), "gates.max_rss_increment_bytes", 1),
        _number(gates.get("parity_atol", 1e-5), "gates.parity_atol"),
        _number(gates.get("parity_rtol", 1e-4), "gates.parity_rtol"),
    )

    output = _mapping(document["output"], "output")
    _strict(
        output, {"runs_dir", "campaigns_dir", "exports_dir"}, "output", {"runs_dir", "campaigns_dir", "exports_dir"}
    )
    return TrainingConfig(
        TRAINING_CONFIG_SCHEMA,
        DataConfig(_path(data["csv_path"], "data.csv_path", source.parent), classes),
        ModelConfig(
            architectures,
            _path(model.get("weights_dir", "models/pretrained"), "model.weights_dir", source.parent),
            num_classes,
        ),
        settings,
        gate_config,
        OutputConfig(
            _path(output["runs_dir"], "output.runs_dir", source.parent),
            _path(output["campaigns_dir"], "output.campaigns_dir", source.parent),
            _path(output["exports_dir"], "output.exports_dir", source.parent),
        ),
        source,
    )


def _parse_trial(value: Mapping[str, Any], index: int) -> RecipeSpec:
    _strict(
        value,
        {
            "trial_id",
            "augmentation",
            "loss",
            "class_weight_exponent",
            "focal_gamma",
            "learning_rate",
            "policy",
            "label_smoothing",
            "warmup_epochs",
            "gradient_clip_norm",
        },
        "trial",
        {"trial_id", "augmentation", "loss", "class_weight_exponent", "focal_gamma", "learning_rate", "policy"},
    )
    augmentation = _mapping(value["augmentation"], "trial.augmentation")
    _strict(
        augmentation,
        {"flip", "rotation", "translate", "scale_min", "scale_max", "brightness", "contrast"},
        "trial.augmentation",
        {"flip", "rotation", "translate", "scale_min", "scale_max", "brightness", "contrast"},
    )
    trial_id = value["trial_id"]
    if trial_id != f"T{index}":
        raise ConfigError("trial_grid must use canonical ids T0 through T8")
    label_smoothing = _number(value.get("label_smoothing", 0.0), "trial.label_smoothing")
    if label_smoothing >= 1.0:
        raise ConfigError("trial.label_smoothing must be < 1")
    gradient_clip = value.get("gradient_clip_norm")
    if gradient_clip is not None:
        gradient_clip = _number(gradient_clip, "trial.gradient_clip_norm")
        if gradient_clip == 0:
            raise ConfigError("trial.gradient_clip_norm must be > 0")
    return RecipeSpec(
        trial_id,
        AugmentationConfig(
            *(
                float(augmentation[key])
                for key in ("flip", "rotation", "translate", "scale_min", "scale_max", "brightness", "contrast")
            )
        ),
        str(value["loss"]),
        value["class_weight_exponent"],
        value["focal_gamma"],
        str(value["learning_rate"]),
        str(value["policy"]),
        label_smoothing,
        _int(value.get("warmup_epochs", 0), "trial.warmup_epochs", 0, 20),
        gradient_clip,
    )


def load_tuning_config(path: str | Path) -> TuningConfig:
    source, document = _read(path)
    _strict(
        document,
        {"schema", "training_config", "architectures", "seeds", "trial_grid", "tuning_fraction"},
        "root",
        {"schema", "training_config", "architectures", "seeds", "trial_grid", "tuning_fraction"},
    )
    if document["schema"] != TUNING_CONFIG_SCHEMA:
        raise ConfigError(f"expected {TUNING_CONFIG_SCHEMA}, received {document['schema']!r}")
    architectures = tuple(document["architectures"])
    if architectures != ARCHITECTURES:
        raise ConfigError("tuning.architectures must enumerate the three fixed architectures in order")
    seeds = tuple(document["seeds"])
    if seeds != SEEDS:
        raise ConfigError("tuning.seeds must be exactly [42, 43, 44]")
    if float(document["tuning_fraction"]) != 0.20:
        raise ConfigError("tuning_fraction is fixed at 0.20")
    grid = tuple(_parse_trial(_mapping(item, f"trial_grid[{i}]"), i) for i, item in enumerate(document["trial_grid"]))
    canonical_grid = default_recipe_grid()
    if [item.to_dict() for item in grid] != [item.to_dict() for item in canonical_grid]:
        raise ConfigError("trial_grid must match the fixed T0..T8 policy exactly")
    return TuningConfig(
        TUNING_CONFIG_SCHEMA,
        _path(document["training_config"], "training_config", source.parent),
        architectures,
        seeds,
        grid,
        0.20,
        source,
    )
