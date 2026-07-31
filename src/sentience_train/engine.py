"""Training engine and the reproducible release-candidate orchestrator."""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import math
import platform
import random
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .artifacts import RunStaging, append_jsonl, git_state, sha256_file, utc_now
from .config import AppConfig, LABELS, ModelConfig, QATConfig, SEEDS, WIDTHS
from .data import FER2013Dataset, TrainingData, load_training_data
from .models import NHWCModel, build_candidate, build_teacher
from .preflight import (
    Arena,
    Inspection,
    PreflightError,
    PreflightFailed,
    _DefaultToolchain,
    _macs,
    _validate_inspection,
)
from .selection import (
    EarlyStopping,
    FrozenRun,
    GateResult,
    Metrics,
    ResourceMetrics,
    SeedResult,
    SelectionFailed,
    WidthAggregate,
    aggregate,
    aggregate_metrics,
    evaluate_gates,
    metrics_from_predictions,
    representative_seed,
    should_run_qat,
    width_order,
)


@dataclass(frozen=True)
class CompletedRun:
    """Compatibility result for issue-#3 fixture configurations."""

    run_id: str
    directory: Path
    best_epoch: int
    best_macro_f1: float
    best_loss: float


@dataclass
class _TrainingResult:
    model: nn.Module
    metrics: Metrics
    checkpoint: Path
    history: Path
    best_epoch: int


@dataclass
class _QuantizedResult:
    metrics: Metrics
    resources: ResourceMetrics
    artifact: Path
    route: str


class SelectionQuantizer(Protocol):
    """Substitutable direct LiteRT/AI Edge quantization surface."""

    def export_quantize(self, model: nn.Module, calibration: torch.Tensor, destination: Path) -> None: ...

    def inspect(self, flatbuffer: Path) -> Inspection: ...

    def infer_litert(self, flatbuffer: Path, sample: torch.Tensor) -> Any: ...

    def probe_tflm(self, flatbuffer: Path, operators: Sequence[Any], working_directory: Path) -> Arena: ...


class _FakeQuantWrapper(nn.Module):
    """Apply W8A8 fake quantization to a Conv2d/Linear module."""

    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped

    @staticmethod
    def _fake(value: torch.Tensor) -> torch.Tensor:
        detached = value.detach()
        minimum = float(detached.min().item())
        maximum = float(detached.max().item())
        scale = max((maximum - minimum) / 255.0, 1e-8)
        zero_point = int(max(-128, min(127, round(-minimum / scale - 128))))
        return torch.fake_quantize_per_tensor_affine(value, scale, zero_point, -128, 127)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self._fake(value)
        module = self.wrapped
        if isinstance(module, nn.Conv2d):
            weight = self._fake(module.weight)
            bias = None if module.bias is None else self._fake(module.bias)
            return F.conv2d(value, weight, bias, module.stride, module.padding, module.dilation, module.groups)
        if isinstance(module, nn.Linear):
            weight = self._fake(module.weight)
            bias = None if module.bias is None else self._fake(module.bias)
            return F.linear(value, weight, bias)
        return self._fake(module(value))


class _FakeQuantizedModel(nn.Module):
    """A training-only W8A8 graph; it is never sent to the converter."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._wrap(self.model)

    def _wrap(self, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(module, name, _FakeQuantWrapper(child))
            else:
                self._wrap(child)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _FakeQuantWrapper._fake(self.model(value))


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


def _loaders(config: AppConfig, data: TrainingData, seed: int) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
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


def _validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, *, dtype: str = "float32") -> Metrics:
    model.eval()
    total_loss = 0.0
    total = 0
    actual: list[int] = []
    predicted: list[int] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            scores.append(torch.softmax(logits, dim=1).cpu().numpy())
    return metrics_from_predictions(actual, predicted, np.concatenate(scores), loss=total_loss / max(total, 1), dtype=dtype)


def _distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    criterion: nn.Module,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
) -> torch.Tensor:
    ce = criterion(student_logits, labels)
    student = F.log_softmax(student_logits / temperature, dim=1)
    teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(student, teacher, reduction="batchmean") * (temperature ** 2)
    return ce_weight * ce + kl_weight * kl


def _build_model(config: AppConfig, width: float, model_factory: Callable[..., nn.Module] | None) -> nn.Module:
    name = f"ds_cnn_w{int(round(width * 100)):03d}"
    if model_factory is None:
        return build_candidate(name, config.model.num_classes)
    candidate = ModelConfig(num_classes=config.model.num_classes, name=name)
    try:
        signature = inspect.signature(model_factory)
        positional = [parameter for parameter in signature.parameters.values() if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        positional = [None]
    if len(positional) >= 2:
        return model_factory(width, config.model.num_classes)
    return model_factory(candidate)


def _train_epochs(
    config: AppConfig,
    data: TrainingData,
    model: nn.Module,
    seed: int,
    checkpoint: Path,
    history_path: Path,
    device: torch.device,
    *,
    max_epochs: int,
    teacher: nn.Module | None = None,
    learning_rate: float | None = None,
) -> _TrainingResult:
    weights = _class_weights(data.train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer_config = config.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.lr if learning_rate is None else learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    train_loader, validation_loader = _loaders(config, data, seed)
    stopper = EarlyStopping(config.early_stopping.patience, config.early_stopping.min_delta)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for images, labels in train_loader:
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if teacher is None:
                loss = criterion(logits, labels)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                distill = config.distillation
                loss = _distillation_loss(logits, labels, teacher_logits, criterion, distill.temperature, distill.ce_weight, distill.kl_weight)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * labels.size(0)
            train_examples += labels.size(0)
        validation = _validate(model, validation_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_examples, 1),
            "validation_loss": validation.loss,
            "macro_f1": validation.macro_f1,
            "accuracy": validation.accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(history_path, row)
        if stopper.update(epoch, validation.macro_f1, validation.loss):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": validation.macro_f1,
                "validation_loss": validation.loss,
                "seed": seed,
            }, checkpoint)
        if stopper.should_stop:
            break
    selected = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    # Save the restored state, making the checkpoint self-describing.
    selected["restored_best"] = True
    torch.save(selected, checkpoint)
    final_metrics = _validate(model, validation_loader, criterion, device)
    return _TrainingResult(model, final_metrics, checkpoint, history_path, int(selected["epoch"]))


def _write_metrics(path: Path, metrics: Metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_confusion(directory: Path, prefix: str, metrics: Metrics) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{prefix}.json"
    csv_path = directory / f"{prefix}.csv"
    json_path.write_text(json.dumps({"labels": list(LABELS), "matrix": [list(row) for row in metrics.confusion_matrix]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *LABELS])
        for label, row in zip(LABELS, metrics.confusion_matrix):
            writer.writerow([label, *row])
    return json_path, csv_path


def _calibration(data: TrainingData, seed: int, examples: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if examples % len(LABELS) != 0:
        raise PreflightError("calibration", "calibration_examples must be divisible by class count")
    per_label = examples // len(LABELS)
    rng = random.Random(seed)
    selected = []
    for label, name in enumerate(LABELS):
        candidates = [record for record in data.train.records if record.label == label]
        if len(candidates) < per_label:
            raise PreflightError("calibration", f"Training has too few {name} samples", observed=len(candidates), limit=per_label)
        selected.extend(rng.sample(candidates, per_label))
    dataset = FER2013Dataset(data.provenance.path, tuple(selected), augmentation=None)
    values = torch.stack([dataset[index][0] for index in range(len(dataset))])
    identifiers = [str(record.line_number) for record in selected]
    return values.permute(0, 2, 3, 1).contiguous(), {
        "seed": seed,
        "per_label": per_label,
        "total": len(selected),
        "counts": {name: per_label for name in LABELS},
        "identifier_sha256": hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest(),
        "source_partition": "Training",
    }


def _as_contract(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {key: getattr(value, key) for key in value.__dataclass_fields__}
    return dict(value)


def _resource_from_mapping(value: Mapping[str, Any]) -> ResourceMetrics:
    return ResourceMetrics(
        tflite_bytes=int(value["tflite_bytes"]),
        arena_with_margin_bytes=int(value["arena_with_margin_bytes"]),
        arena_total_bytes=None if value.get("arena_total_bytes") is None else int(value["arena_total_bytes"]),
        macs=int(value["macs"]),
        input_contract=value.get("input_contract", {}),
        output_contract=value.get("output_contract", {}),
        operators=tuple(value.get("operators", ())),
        sha256=value.get("sha256"),
    )


def _metrics_from_value(value: Any, *, dtype: str = "int8") -> Metrics:
    if isinstance(value, Metrics):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("metrics"), Metrics):
        return value["metrics"]
    if isinstance(value, Mapping) and {"y_true", "y_pred"}.issubset(value):
        return metrics_from_predictions(value["y_true"], value["y_pred"], value.get("y_score"), loss=value.get("loss"), dtype=dtype)
    raise PreflightError("evaluation", "quantization toolchain did not return PublicTest metrics")


def _quantize(
    config: AppConfig,
    data: TrainingData,
    model: nn.Module,
    calibration: torch.Tensor,
    destination: Path,
    seed: int,
    width: float,
    route: str,
    toolchain: SelectionQuantizer,
    quantization_runner: Callable[..., Any] | None,
) -> _QuantizedResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    runner_value: Any = None
    if quantization_runner is not None:
        try:
            signature = inspect.signature(quantization_runner)
            positional = [parameter for parameter in signature.parameters.values() if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
            if len(positional) >= 6:
                runner_value = quantization_runner(model, calibration, destination, route, width, seed)
            elif len(positional) >= 4:
                runner_value = quantization_runner(model, calibration, destination, route)
            else:
                runner_value = quantization_runner(model, calibration, destination)
        except PreflightError:
            raise
        except Exception as exc:
            raise PreflightError("conversion", str(exc)) from exc
    else:
        try:
            toolchain.export_quantize(NHWCModel(model).eval(), calibration, destination)
        except PreflightError:
            raise
        except Exception as exc:
            raise PreflightError("conversion", str(exc)) from exc
    if not destination.is_file():
        raise PreflightError("conversion", f"quantizer did not create {destination}")

    embedded_metrics: Metrics | None = None
    embedded_resources: ResourceMetrics | None = None
    if isinstance(runner_value, Mapping):
        if runner_value.get("metrics") is not None:
            embedded_metrics = _metrics_from_value(runner_value, dtype="int8")
        if runner_value.get("resources") is not None:
            resource_value = runner_value["resources"]
            embedded_resources = resource_value if isinstance(resource_value, ResourceMetrics) else _resource_from_mapping(resource_value)
    try:
        inspection = toolchain.inspect(destination)
        _validate_inspection(inspection)
        inference = toolchain.infer_litert(destination, calibration[:1])
        if getattr(inference, "shape", None) != (1, len(LABELS)) or getattr(inference, "dtype", None) != "int8":
            raise PreflightError("litert_inference", "LiteRT inference contract failed", observed=_as_contract(inference))
        arena = toolchain.probe_tflm(destination, inspection.operators, destination.parent / ".probe")
        arena_total = int(arena.total_bytes)
        resources = ResourceMetrics(
            tflite_bytes=destination.stat().st_size,
            arena_with_margin_bytes=math.ceil(arena_total * 1.20),
            arena_total_bytes=arena_total,
            macs=_macs(model),
            input_contract=_as_contract(inspection.input),
            output_contract=_as_contract(inspection.output),
            operators=tuple(_as_contract(operator) for operator in inspection.operators),
            sha256=sha256_file(destination),
        )
    except PreflightError:
        if embedded_resources is None:
            raise
        resources = embedded_resources
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        if embedded_resources is None:
            raise PreflightError("inspection", str(exc)) from exc
        resources = embedded_resources

    if embedded_resources is not None:
        # The substitute runner may provide a complete inspection result.  The
        # file hash is still taken here so the manifest remains authoritative.
        resources = replace(embedded_resources, tflite_bytes=destination.stat().st_size, sha256=sha256_file(destination))

    metric_value = embedded_metrics
    if metric_value is None:
        evaluator = getattr(toolchain, "evaluate_litert", None) or getattr(toolchain, "evaluate", None)
        predictor = getattr(toolchain, "predict_litert", None) or getattr(toolchain, "predict", None)
        validation_loader = DataLoader(data.validation, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)
        if evaluator is not None:
            value = evaluator(destination, validation_loader)
            metric_value = _metrics_from_value(value)
        elif predictor is not None:
            value = predictor(destination, validation_loader)
            metric_value = _metrics_from_value(value)
        else:
            raise PreflightError("evaluation", "quantization toolchain has no PublicTest evaluator")
    return _QuantizedResult(metric_value, resources, destination, route)


def _unwrap_qat_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model."):]
        key = key.replace(".wrapped.", ".")
        result[key] = value
    return result


def _qat_train(
    config: AppConfig,
    data: TrainingData,
    fp32: _TrainingResult,
    seed: int,
    checkpoint: Path,
    history: Path,
    device: torch.device,
) -> _TrainingResult:
    base = copy.deepcopy(fp32.model).to(device)
    qat = _FakeQuantizedModel(base).to(device)
    result = _train_epochs(config, data, qat, seed, checkpoint, history, device, max_epochs=config.quantization.qat.epochs, learning_rate=config.quantization.qat.lr)
    normal = copy.deepcopy(fp32.model).to(device)
    normal.load_state_dict(_unwrap_qat_state(qat.state_dict()))
    normal_result = _TrainingResult(normal, result.metrics, result.checkpoint, result.history, result.best_epoch)
    return normal_result


def _seed_result(
    config: AppConfig,
    data: TrainingData,
    width: float,
    seed: int,
    candidate_directory: Path,
    device: torch.device,
    model_factory: Callable[..., nn.Module] | None,
    toolchain: SelectionQuantizer,
    quantization_runner: Callable[..., Any] | None,
    *,
    teacher: nn.Module | None = None,
) -> SeedResult:
    name = f"ds_cnn_w{int(round(width * 100)):03d}"
    seed_directory = candidate_directory / f"seed-{seed}"
    fp32_metrics_path = seed_directory / "metrics" / "fp32.json"
    fp32_checkpoint = seed_directory / "checkpoints" / "fp32.pt"
    history = seed_directory / "metrics" / "history.jsonl"
    seed_everything(seed)
    model = _build_model(config, width, model_factory).to(device)
    fp32 = _train_epochs(config, data, model, seed, fp32_checkpoint, history, device, max_epochs=config.training.epochs, teacher=teacher)
    _write_metrics(fp32_metrics_path, fp32.metrics)
    fp32_gate = evaluate_gates(fp32.metrics, config.quality_gates)
    fp32_cm_json, fp32_cm_csv = _write_confusion(seed_directory / "metrics", "fp32-confusion", fp32.metrics)
    common = {
        "width": width,
        "seed": seed,
        "fp32_metrics": fp32.metrics,
        "fp32_gate": fp32_gate,
        "checkpoint_fp32": str(fp32_checkpoint.relative_to(candidate_directory.parent.parent)),
        "history": str(history.relative_to(candidate_directory.parent.parent)),
        "confusion_matrix_json": str(fp32_cm_json.relative_to(candidate_directory.parent.parent)),
        "confusion_matrix_csv": str(fp32_cm_csv.relative_to(candidate_directory.parent.parent)),
        "best_epoch": fp32.best_epoch,
    }
    if not fp32_gate.passed:
        return SeedResult(**common, status="failed", transition_reasons=("fp32_predictive_gates_failed", *fp32_gate.reasons))

    calibration, calibration_report = _calibration(data, config.quantization.calibration_seed, config.quantization.calibration_examples)
    (seed_directory / "metrics" / "calibration.json").write_text(json.dumps(calibration_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_artifact = seed_directory / "selection.tflite"
    ptq_error: PreflightError | None = None
    ptq: _QuantizedResult | None = None
    try:
        ptq = _quantize(config, data, fp32.model, calibration, selection_artifact, seed, width, "ptq", toolchain, quantization_runner)
    except PreflightError as exc:
        ptq_error = exc

    if ptq is not None:
        delta = fp32.metrics.accuracy - ptq.metrics.accuracy
        int8_gate = evaluate_gates(ptq.metrics, config.quality_gates, reference=fp32.metrics, resources=ptq.resources)
        _write_metrics(seed_directory / "metrics" / "int8.json", ptq.metrics)
        int8_cm_json, int8_cm_csv = _write_confusion(seed_directory / "metrics", "int8-confusion", ptq.metrics)
        if not should_run_qat(ptq_conversion_failed=False, accuracy_delta=delta, max_accuracy_delta=config.quality_gates.max_accuracy_delta):
            status = "passed" if int8_gate.passed else "failed"
            reasons = () if int8_gate.passed else ("ptq_gates_failed_without_qat_trigger", *int8_gate.reasons)
            return SeedResult(**{**common, "confusion_matrix_json": str(int8_cm_json.relative_to(candidate_directory.parent.parent)), "confusion_matrix_csv": str(int8_cm_csv.relative_to(candidate_directory.parent.parent))}, int8_metrics=ptq.metrics, int8_gate=int8_gate, resources=ptq.resources, accuracy_delta=delta, route="ptq", status=status, transition_reasons=reasons, tflite=str(selection_artifact.relative_to(candidate_directory.parent.parent)))
    qat_trigger = should_run_qat(
        ptq_conversion_failed=ptq_error is not None and ptq_error.stage == "conversion",
        accuracy_delta=None if ptq is None else fp32.metrics.accuracy - ptq.metrics.accuracy,
        max_accuracy_delta=config.quality_gates.max_accuracy_delta,
    )
    if not qat_trigger:
        reasons = ("ptq_failed_without_qat_trigger",) if ptq_error else ("ptq_gates_failed_without_qat_trigger",)
        if ptq_error:
            reasons += (f"ptq:{ptq_error.stage}", str(ptq_error))
        else:
            reasons += tuple(int8_gate.reasons if ptq else ())
        return SeedResult(**common, status="failed", transition_reasons=reasons)
    if not config.quantization.qat.enabled:
        return SeedResult(**common, status="failed", transition_reasons=("qat_triggered_but_disabled",))

    qat_checkpoint = seed_directory / "checkpoints" / "qat.pt"
    qat_history = seed_directory / "metrics" / "qat-history.jsonl"
    try:
        qat = _qat_train(config, data, fp32, seed, qat_checkpoint, qat_history, device)
        qat_result = _quantize(config, data, qat.model, calibration, selection_artifact, seed, width, "qat", toolchain, quantization_runner)
    except PreflightError as exc:
        return SeedResult(**common, status="failed", transition_reasons=("qat_failed", f"{exc.stage}:{exc}"))
    except Exception as exc:
        return SeedResult(**common, status="failed", transition_reasons=("qat_failed", f"unexpected:{type(exc).__name__}:{exc}"))
    delta = fp32.metrics.accuracy - qat_result.metrics.accuracy
    int8_gate = evaluate_gates(qat_result.metrics, config.quality_gates, reference=fp32.metrics, resources=qat_result.resources)
    _write_metrics(seed_directory / "metrics" / "int8.json", qat_result.metrics)
    int8_cm_json, int8_cm_csv = _write_confusion(seed_directory / "metrics", "int8-confusion", qat_result.metrics)
    reasons = () if int8_gate.passed else ("qat_gates_failed", *int8_gate.reasons)
    return SeedResult(**{**common, "confusion_matrix_json": str(int8_cm_json.relative_to(candidate_directory.parent.parent)), "confusion_matrix_csv": str(int8_cm_csv.relative_to(candidate_directory.parent.parent)), "best_epoch": qat.best_epoch, "qat_history": str(qat_history.relative_to(candidate_directory.parent.parent))}, int8_metrics=qat_result.metrics, int8_gate=int8_gate, resources=qat_result.resources, accuracy_delta=delta, route="qat", status="passed" if int8_gate.passed else "failed", transition_reasons=reasons, checkpoint_qat=str(qat_checkpoint.relative_to(candidate_directory.parent.parent)), tflite=str(selection_artifact.relative_to(candidate_directory.parent.parent)))


def _mean_metrics(results: Sequence[SeedResult], dtype: str = "int8") -> Metrics:
    usable = [result for result in results if result.int8_metrics is not None]
    if not usable:
        raise ValueError("no INT8 metrics available")
    def avg(values: Iterable[float]) -> float:
        return float(np.mean(list(values)))
    return Metrics(
        accuracy=avg(result.int8_metrics.accuracy for result in usable),
        macro_f1=avg(result.int8_metrics.macro_f1 for result in usable),
        loss=avg(result.int8_metrics.loss for result in usable),
        recall_by_class=tuple(avg(result.int8_metrics.recall_by_class[index] for result in usable) for index in range(len(LABELS))),
        precision_by_class=tuple(avg(result.int8_metrics.precision_by_class[index] for result in usable) for index in range(len(LABELS))),
        f1_by_class=tuple(avg(result.int8_metrics.f1_by_class[index] for result in usable) for index in range(len(LABELS))),
        support_by_class=tuple(int(round(avg(result.int8_metrics.support_by_class[index] for result in usable))) for index in range(len(LABELS))),
        confusion_matrix=usable[0].int8_metrics.confusion_matrix,
        dtype=dtype,
    )


def _aggregate_width(width: float, results: Sequence[SeedResult], config: AppConfig) -> WidthAggregate:
    if len(results) != len(SEEDS) or any(result.int8_metrics is None for result in results) or any(result.resources is None for result in results):
        reasons = ["not_all_three_seeds_have_int8_metrics"]
        if any(result.resources is None for result in results):
            reasons.append("not_all_three_seeds_have_resource_contracts")
        reasons.extend(reason for result in results for reason in result.transition_reasons)
        failed = GateResult(False, {"seed_count": len(results)}, {"seed_count": len(SEEDS)}, tuple(reasons))
        return WidthAggregate(width, tuple(results), {}, None, None, failed, "failed", "; ".join(reasons))
    aggregates = aggregate_metrics(results)
    mean_int8 = _mean_metrics(results)
    fp32_mean = Metrics(
        accuracy=aggregate(result.fp32_metrics.accuracy for result in results).mean,
        macro_f1=aggregate(result.fp32_metrics.macro_f1 for result in results).mean,
        loss=aggregate(result.fp32_metrics.loss for result in results).mean,
        recall_by_class=tuple(aggregate(result.fp32_metrics.recall_by_class[index] for result in results).mean for index in range(len(LABELS))),
        precision_by_class=tuple(aggregate(result.fp32_metrics.precision_by_class[index] for result in results).mean for index in range(len(LABELS))),
        f1_by_class=tuple(aggregate(result.fp32_metrics.f1_by_class[index] for result in results).mean for index in range(len(LABELS))),
        support_by_class=results[0].fp32_metrics.support_by_class,
        confusion_matrix=results[0].fp32_metrics.confusion_matrix,
    )
    mean_gate = evaluate_gates(mean_int8, config.quality_gates, reference=fp32_mean)
    representative = next(result for result in results if result.seed == representative_seed(results))
    reasons = list(mean_gate.reasons)
    if not representative.fp32_gate.passed:
        reasons.extend(("representative_fp32_gate_failed", *representative.fp32_gate.reasons))
    if representative.int8_gate is None or not representative.int8_gate.passed:
        reasons.extend(("representative_int8_gate_failed", *(representative.int8_gate.reasons if representative.int8_gate else ())))
    passed = mean_gate.passed and representative.fp32_gate.passed and representative.int8_gate is not None and representative.int8_gate.passed
    return WidthAggregate(width, tuple(results), aggregates, representative.seed, representative, mean_gate, "passed" if passed else "failed", "passed" if passed else "; ".join(reasons))


def _comparison_payload(widths: Sequence[WidthAggregate], *, status: str, selected: WidthAggregate | None, teacher: Mapping[str, Any] | None, config: AppConfig) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": status,
        "canonical_order": list(WIDTHS),
        "selection_policy": {"stop_at_first_pass": True, "representative": config.selection.representative},
        "quality_gates": {
            "accuracy": config.quality_gates.accuracy,
            "macro_f1": config.quality_gates.macro_f1,
            "recall": config.quality_gates.recall,
            "max_accuracy_delta": config.quality_gates.max_accuracy_delta,
            "tflite_bytes": config.quality_gates.tflite_bytes,
            "arena_with_margin_bytes": config.quality_gates.arena_with_margin_bytes,
            "macs": config.quality_gates.macs,
        },
        "selected": None if selected is None else {"width": selected.width, "seed": selected.representative_seed, "route": selected.representative.route if selected.representative else None},
        "teacher": teacher,
        "widths": [width.to_dict() for width in widths],
    }


def _write_comparison(staging: Path, payload: Mapping[str, Any]) -> None:
    (staging / "candidate-comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Candidate comparison", "", f"Status: **{payload['status']}**", "", "| Width | Status | Representative | Mean INT8 accuracy | Mean INT8 macro-F1 | Reason |", "|---:|---|---:|---:|---:|---|"]
    for item in payload["widths"]:
        aggregates = item.get("aggregates", {})
        accuracy = aggregates.get("accuracy", {}).get("mean", "—")
        macro = aggregates.get("macro_f1", {}).get("mean", "—")
        lines.append(f"| {item['width']:.2f} | {item['status']} | {item.get('representative_seed') or '—'} | {accuracy} | {macro} | {item['reason']} |")
    lines.extend(["", "## Gates", "", "```json", json.dumps(payload["quality_gates"], indent=2, sort_keys=True), "```", ""])
    (staging / "candidate-comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256_file(path) for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"}


def _selection_manifest(config: AppConfig, data: TrainingData, staging: RunStaging, status: str, device: torch.device, widths: Sequence[WidthAggregate], selected: WidthAggregate | None, teacher: Mapping[str, Any] | None) -> dict[str, Any]:
    lock = config.source_path.parent.parent / "uv.lock"
    selected_rep = selected.representative if selected else None
    document: dict[str, Any] = {
        "status": status,
        "run_id": staging.run_id,
        "created_at": utc_now(),
        "configuration": config.resolved_dict(),
        "selection": {"width_order": list(WIDTHS), "seed_order": list(SEEDS), "stop_at_first_pass": True, "representative_policy": config.selection.representative},
        "labels": list(LABELS),
        "label_mapping": {"0": "angry", "3": "happy", "6": "neutral", "4": "sad"},
        "csv": {"path": str(data.provenance.path), "sha256": data.provenance.sha256, "schema": list(data.provenance.header), "release_evidence": data.provenance.release_evidence, "total_rows": data.provenance.total_rows, "partition_counts": data.provenance.partition_counts, "raw_class_counts": data.provenance.raw_class_counts, "selected_class_counts": data.provenance.selected_class_counts, "records_used": ["Training", "PublicTest"], "private_test_access": False},
        "git": git_state(config.source_path.parent.parent),
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "device": str(device),
        "candidate_comparison": "candidate-comparison.json",
        "candidates": [width.to_dict() for width in widths],
        "teacher": teacher,
        "artifacts": _artifact_hashes(staging.staging),
    }
    if selected_rep is not None:
        document["selected"] = {
            "width": selected.width,
            "seed": selected_rep.seed,
            "checkpoint_fp32": selected_rep.checkpoint_fp32,
            "checkpoint_qat": selected_rep.checkpoint_qat,
            "int8_artifact": selected_rep.tflite,
            "labels": list(LABELS),
            "route": selected_rep.route,
            "best_epoch": selected_rep.best_epoch,
            "configuration": config.resolved_dict(),
            "metrics": selected_rep.int8_metrics.to_dict() if selected_rep.int8_metrics else None,
        }
    return document


def _train_teacher(config: AppConfig, data: TrainingData, directory: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    seed = config.distillation.teacher_seed
    seed_everything(seed)
    model = build_teacher(config.model.num_classes).to(device)
    result = _train_epochs(config, data, model, seed, directory / "checkpoints" / "teacher.pt", directory / "metrics" / "teacher-history.jsonl", device, max_epochs=config.training.epochs)
    _write_metrics(directory / "metrics" / "teacher-fp32.json", result.metrics)
    provenance = {"architecture": "MiniVGG", "checkpoint": str(result.checkpoint.relative_to(directory.parent)), "seed": seed, "partition": "Training/PublicTest", "loss": "0.5 * CE + 0.5 * T^2 * KL", "temperature": config.distillation.temperature, "exportable": False, "metrics": result.metrics.to_dict()}
    model.eval()
    return model, provenance


def _run_selection(
    config: AppConfig,
    *,
    model_factory: Callable[..., nn.Module] | None = None,
    toolchain: SelectionQuantizer | None = None,
    quantization_runner: Callable[..., Any] | None = None,
    preflight_runner: Callable[[AppConfig, TrainingData, Path], dict] | None = None,
) -> FrozenRun:
    device = _device(config.training.device)
    data = load_training_data(config.data, config.augmentation)
    staging = RunStaging.create(config.run.directory)
    staging.write_resolved_config(config.resolved_dict())
    log_path = staging.staging / "logs" / "selection.jsonl"
    append_jsonl(log_path, {"event": "selection_started", "run_id": staging.run_id, "private_test_access": False})
    quantizer = toolchain or _DefaultToolchain()
    widths: list[WidthAggregate] = []
    selected: WidthAggregate | None = None
    teacher_provenance: dict[str, Any] | None = None
    try:
        if preflight_runner is not None:
            preflight = preflight_runner(config, data, staging.staging / "preflight")
            append_jsonl(log_path, {"event": "preflight_passed", "report": "preflight/report.json", "candidates": len(preflight.get("candidates", []))})
        for width in width_order(config.selection.widths):
            result_42 = _seed_result(config, data, width, 42, staging.staging / "candidates" / f"ds_cnn_w{int(round(width * 100)):03d}", device, model_factory, quantizer, quantization_runner)
            results = [result_42]
            if result_42.fp32_gate.passed:
                for seed in (43, 44):
                    results.append(_seed_result(config, data, width, seed, staging.staging / "candidates" / f"ds_cnn_w{int(round(width * 100)):03d}", device, model_factory, quantizer, quantization_runner))
            aggregate_result = _aggregate_width(width, results, config)
            widths.append(aggregate_result)
            append_jsonl(log_path, {"event": "width_completed", "width": width, "status": aggregate_result.status, "reason": aggregate_result.reason})
            if aggregate_result.status == "passed":
                selected = aggregate_result
                break
        if selected is None and config.distillation.enabled:
            teacher, teacher_provenance = _train_teacher(config, data, staging.staging / "teacher", device)
            append_jsonl(log_path, {"event": "distillation_started", "teacher": teacher_provenance})
            for width in width_order(config.selection.widths):
                result_42 = _seed_result(config, data, width, 42, staging.staging / "distilled" / f"ds_cnn_w{int(round(width * 100)):03d}", device, model_factory, quantizer, quantization_runner, teacher=teacher)
                results = [result_42]
                if result_42.fp32_gate.passed:
                    for seed in (43, 44):
                        results.append(_seed_result(config, data, width, seed, staging.staging / "distilled" / f"ds_cnn_w{int(round(width * 100)):03d}", device, model_factory, quantizer, quantization_runner, teacher=teacher))
                aggregate_result = _aggregate_width(width, results, config)
                widths.append(aggregate_result)
                if aggregate_result.status == "passed":
                    selected = aggregate_result
                    break
        if selected is not None:
            widths = [replace(item, status="advanced") if item.status == "failed" else item for item in widths]
            attempted = {item.width for item in widths}
            widths.extend(WidthAggregate(width, (), {}, None, None, GateResult(False, {}, {}, ("not_attempted_after_first_pass",)), "not_attempted", "not_attempted_after_first_pass") for width in WIDTHS if width not in attempted)
            payload = _comparison_payload(widths, status="frozen", selected=selected, teacher=teacher_provenance, config=config)
            _write_comparison(staging.staging, payload)
            staging.write_manifest(_selection_manifest(config, data, staging, "frozen", device, widths, selected, teacher_provenance))
            final = staging.finalize()
            representative = selected.representative
            assert representative is not None and representative.tflite is not None and representative.checkpoint_fp32 is not None
            checkpoint_relative = representative.checkpoint_qat or representative.checkpoint_fp32
            checkpoint = final / checkpoint_relative
            artifact = final / representative.tflite
            return FrozenRun(staging.run_id, final, selected.width, representative.seed, checkpoint, artifact, representative.route or "ptq", representative.int8_metrics or representative.fp32_metrics, best_epoch=representative.best_epoch)
        attempted = {item.width for item in widths}
        failed_indexes = [index for index, item in enumerate(widths) if item.status == "failed"]
        if failed_indexes:
            last_failed = failed_indexes[-1]
            widths = [replace(item, status="advanced") if item.status == "failed" and index != last_failed else item for index, item in enumerate(widths)]
        widths.extend(WidthAggregate(width, (), {}, None, None, GateResult(False, {}, {}, ("not_attempted",)), "not_attempted", "not_attempted") for width in WIDTHS if width not in attempted)
        payload = _comparison_payload(widths, status="selection_failed", selected=None, teacher=teacher_provenance, config=config)
        _write_comparison(staging.staging, payload)
        staging.write_manifest(_selection_manifest(config, data, staging, "selection_failed", device, widths, None, teacher_provenance))
        final = staging.finalize()
        raise SelectionFailed(final)
    except SelectionFailed:
        raise
    except Exception as exc:
        append_jsonl(log_path, {"event": "selection_failed_unexpected", "error": type(exc).__name__, "message": str(exc)})
        if staging.staging.exists():
            payload = _comparison_payload(widths, status="selection_failed", selected=None, teacher=teacher_provenance, config=config)
            _write_comparison(staging.staging, payload)
            staging.write_manifest(_selection_manifest(config, data, staging, "selection_failed", device, widths, None, teacher_provenance))
            final = staging.finalize()
            raise SelectionFailed(final, f"unexpected selection error: {exc}") from exc
        raise


def train_one_seed(
    config: AppConfig,
    data: TrainingData,
    width: float,
    seed: int,
    directory: Path,
    *,
    model_factory: Callable[..., nn.Module] | None = None,
    toolchain: SelectionQuantizer | None = None,
    quantization_runner: Callable[..., Any] | None = None,
    teacher: nn.Module | None = None,
) -> SeedResult:
    """Train/evaluate exactly one width and seed.

    This is the seam used by integration tests and by the orchestration loop;
    it never accesses a partition other than the datasets already supplied in
    ``TrainingData``.
    """
    return _seed_result(
        config,
        data,
        width,
        seed,
        directory,
        _device(config.training.device),
        model_factory,
        toolchain or _DefaultToolchain(),
        quantization_runner,
        teacher=teacher,
    )


def _legacy_manifest(config: AppConfig, data: TrainingData, staging: RunStaging, status: str, device: torch.device, artifacts: dict[str, str], preflight_failure: dict[str, Any] | None = None) -> dict[str, Any]:
    document = {
        "status": status,
        "run_id": staging.run_id,
        "created_at": utc_now(),
        "configuration": config.resolved_dict(),
        "seed": config.training.seed,
        "labels": list(LABELS),
        "csv": {"path": str(data.provenance.path), "sha256": data.provenance.sha256, "schema": list(data.provenance.header), "release_evidence": data.provenance.release_evidence, "total_rows": data.provenance.total_rows, "partition_counts": data.provenance.partition_counts, "raw_class_counts": data.provenance.raw_class_counts, "selected_class_counts": data.provenance.selected_class_counts},
        "device": str(device),
        "artifacts": artifacts,
    }
    if preflight_failure is not None:
        document["preflight_failure"] = preflight_failure
    return document


def _legacy_run_training(config: AppConfig, model_factory: Callable[[ModelConfig], nn.Module], preflight_runner: Callable[[AppConfig, TrainingData, Path], dict]) -> CompletedRun:
    """Read-only compatibility route for pre-selection issue-#3 fixtures."""
    seed = config.training.seed if config.training.seed is not None else 42
    seed_everything(seed)
    device = _device(config.training.device)
    data = load_training_data(config.data, config.augmentation)
    staging = RunStaging.create(config.run.directory)
    staging.write_resolved_config(config.resolved_dict())
    log_path = staging.staging / "logs" / "train.jsonl"
    try:
        preflight = preflight_runner(config, data, staging.staging / "preflight")
        append_jsonl(log_path, {"event": "preflight_passed", "candidates": len(preflight["candidates"])})
        model = model_factory(config.model).to(device)
        weights = _class_weights(data.train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        train_loader, validation_loader = _loaders(config, data, seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
        checkpoint = staging.staging / "checkpoints" / "best.pt"
        history = staging.staging / "metrics" / "history.jsonl"
        stopper = EarlyStopping(config.early_stopping.patience, config.early_stopping.min_delta)
        for epoch in range(1, config.training.epochs + 1):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
            validation = _validate(model, validation_loader, criterion, device)
            scheduler.step()
            append_jsonl(history, {"epoch": epoch, "validation_loss": validation.loss, "macro_f1": validation.macro_f1})
            if stopper.update(epoch, validation.macro_f1, validation.loss):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "macro_f1": validation.macro_f1, "validation_loss": validation.loss}, checkpoint)
            if stopper.should_stop:
                break
        (staging.staging / "metrics" / "validation.json").write_text(json.dumps({
            "best_epoch": stopper.best_epoch,
            "macro_f1": stopper.best_score,
            "loss": stopper.best_loss,
            "selection": "highest_macro_f1_then_lowest_loss",
            "restored_checkpoint": "checkpoints/best.pt",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts = _artifact_hashes(staging.staging)
        staging.write_manifest(_legacy_manifest(config, data, staging, "completed", device, artifacts))
        directory = staging.finalize()
        return CompletedRun(staging.run_id, directory, stopper.best_epoch, stopper.best_score, stopper.best_loss)
    except PreflightFailed as exc:
        report = json.loads((exc.report_directory / "report.json").read_text(encoding="utf-8"))
        failed = next((candidate for candidate in report.get("candidates", []) if candidate.get("status") == "failed"), report.get("calibration", {}))
        failure = {key: failed[key] for key in ("name", "failed_stage", "observed", "limit", "error") if key in failed}
        staging.write_manifest(_legacy_manifest(config, data, staging, "preflight_failed", device, _artifact_hashes(staging.staging), failure))
        directory = staging.finalize()
        exc.relocate(directory / "preflight")
        raise


def run_training(
    config: AppConfig,
    model_factory: Callable[..., nn.Module] | None = None,
    preflight_runner: Callable[[AppConfig, TrainingData, Path], dict] | None = None,
    *,
    toolchain: SelectionQuantizer | None = None,
    quantization_runner: Callable[..., Any] | None = None,
) -> FrozenRun | CompletedRun:
    """Train and freeze the first eligible candidate, or raise SelectionFailed.

    New configurations always use the fixed DS-CNN selection policy.  The
    compatibility branch is limited to documents containing the removed
    issue-#3 ``model.name``/``training.seed`` fields and exists solely so old
    deterministic foundation fixtures remain executable.
    """
    if config.legacy_mode:
        if preflight_runner is None:
            raise ValueError("legacy configuration requires preflight_runner")
        return _legacy_run_training(config, model_factory or (lambda model_config: build_candidate(model_config.name or "ds_cnn_w025", model_config.num_classes)), preflight_runner)
    return _run_selection(config, model_factory=model_factory, toolchain=toolchain, quantization_runner=quantization_runner, preflight_runner=preflight_runner)
