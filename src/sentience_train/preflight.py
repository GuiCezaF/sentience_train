"""The deployment evidence gate which runs before optimization.

The public entry point is intentionally small: :func:`run_preflight` receives
the immutable configuration, the already validated ``TrainingData``, and an
empty directory belonging to the run.  Conversion, FlatBuffer inspection,
runtime probing, and reporting are private details behind ``PreflightToolchain``
so fast tests can substitute only those impure operations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
import numpy as np
from torch import nn

from .config import AppConfig, LABELS
from .data import TrainingData
from .models import DSCNN_BASELINES, NHWCModel, build_candidate


SCHEMA_VERSION = 1
CALIBRATION_SEED = 42
CALIBRATION_PER_LABEL = 128
MAX_TFLITE_BYTES = 1_048_576
MAX_MACS = 15_000_000
MAX_ARENA_BYTES = 1_572_864
ALLOWED_BUILTINS = frozenset({
    "CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "RELU6",
})
# The interpreter exposes these versions; newer versions require an explicit
# review of the micro resolver before accepting them.
MAX_OPERATOR_VERSIONS = {
    "CONV_2D": 3, "DEPTHWISE_CONV_2D": 3, "AVERAGE_POOL_2D": 2,
    "RESHAPE": 1, "FULLY_CONNECTED": 4, "RELU6": 1,
}
TFLM_COMMIT = "330b1747c9d51c0e394f51a2a34ff42deb9b95f5"
TFLM_PROBE_IMAGE = "ghcr.io/tensorflow/tflite-micro-build:2.16.1@sha256:9b11b1846b9c6d733fb5e3ea580bde410d9f0b5fc51aed738ec9a480fa0b1be5"


class PreflightError(RuntimeError):
    """One candidate did not meet the deployment contract."""

    def __init__(self, stage: str, message: str, *, observed: Any = None, limit: Any = None) -> None:
        self.stage, self.observed, self.limit = stage, observed, limit
        super().__init__(message)


class PreflightFailed(RuntimeError):
    """All candidates ran, and at least one of them failed the preflight."""

    def __init__(self, report_directory: Path) -> None:
        self.report_directory = report_directory
        super().__init__(f"preflight failed; report directory: {report_directory}")

    def relocate(self, report_directory: Path) -> None:
        """Point the raised error at the immutable location after finalisation."""
        self.report_directory = report_directory
        self.args = (f"preflight failed; report directory: {report_directory}",)


@dataclass(frozen=True)
class TensorContract:
    shape: tuple[int, ...]
    dtype: str
    scale: float
    zero_point: int


@dataclass(frozen=True)
class Operator:
    builtin: str
    version: int
    custom_code: str | None = None


@dataclass(frozen=True)
class Inspection:
    input: TensorContract
    output: TensorContract
    operators: tuple[Operator, ...]
    activation_dtypes: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Inference:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class Arena:
    head_bytes: int
    temporary_bytes: int
    tail_bytes: int
    total_bytes: int


class PreflightToolchain(Protocol):
    """Impure converter/inspector/runtime surface; all methods are internal."""

    def versions(self) -> dict[str, str]: ...

    def export_quantize(self, model: nn.Module, calibration: torch.Tensor, destination: Path) -> None: ...

    def inspect(self, flatbuffer: Path) -> Inspection: ...

    def infer_litert(self, flatbuffer: Path, sample: torch.Tensor) -> Inference: ...

    def probe_tflm(self, flatbuffer: Path, operators: Sequence[Operator], working_directory: Path) -> Arena: ...


def _name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "").replace("numpy.", "").lower()


class _DefaultToolchain:
    """Production route: torch.export -> LiteRT Torch -> AI Edge Quantizer.

    Imports are deferred on purpose.  A normal source checkout can run unit
    tests without the Linux-only deployment toolchain installed, while ``train``
    fails explicitly and immutably when that prerequisite is absent.
    """

    def versions(self) -> dict[str, str]:
        names = ("litert-torch", "ai-edge-quantizer-nightly", "ai-edge-litert")
        result: dict[str, str] = {}
        for name in names:
            try:
                result[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result[name] = "not-installed"
        result["tflm_commit"] = TFLM_COMMIT
        result["tflm_probe_image"] = TFLM_PROBE_IMAGE
        return result

    def export_quantize(self, model: nn.Module, calibration: torch.Tensor, destination: Path) -> None:
        try:
            import ai_edge_quantizer  # type: ignore[import-not-found]
            from ai_edge_quantizer import recipe  # type: ignore[import-not-found]
            from ai_edge_quantizer.utils import tfl_interpreter_utils  # type: ignore[import-not-found]
            from ai_edge_litert import interpreter as litert_interpreter  # type: ignore[import-not-found]
            import litert_torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PreflightError("conversion", f"deployment toolchain unavailable: {exc}") from exc
        fp32_path = destination.with_suffix(".fp32.tflite")
        try:
            model.eval()
            example = calibration[:1]
            # Keep torch.export explicit in the evidence chain.  LiteRT Torch's
            # converter accepts the exported module rather than tracing again.
            exported = torch.export.export(model, (example,))
            edge_model = litert_torch.convert(exported.module(), (example,))
            edge_model.export(str(fp32_path))
            # ai-edge-quantizer's public static INT8 recipe owns both input and
            # output quantization; no TensorFlow converter is involved.
            quantizer = ai_edge_quantizer.Quantizer(str(fp32_path))
            quantizer.load_quantization_recipe(recipe.static_wi8_ai8())
            interpreter = litert_interpreter.Interpreter(model_path=str(fp32_path))
            interpreter.allocate_tensors()
            input_name = interpreter.get_input_details()[0]["name"]
            calibration_data = {
                tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: [
                    {input_name: sample.detach().cpu().numpy()} for sample in calibration
                ]
            }
            calibration_result = quantizer.calibrate(calibration_data)
            result = quantizer.quantize(calibration_result)
            destination.write_bytes(bytes(result.quantized_model))
        except PreflightError:
            raise
        except Exception as exc:
            raise PreflightError("conversion", str(exc)) from exc
        finally:
            fp32_path.unlink(missing_ok=True)

    def inspect(self, flatbuffer: Path) -> Inspection:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-not-found]
            from ai_edge_litert import schema_py_generated as schema  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PreflightError("inspection", f"LiteRT runtime unavailable: {exc}") from exc
        try:
            interpreter = Interpreter(model_path=str(flatbuffer))
            interpreter.allocate_tensors()
            input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            tensors = interpreter.get_tensor_details()
            root = schema.Model.GetRootAs(flatbuffer.read_bytes(), 0)
        except Exception as exc:
            raise PreflightError("inspection", str(exc)) from exc
        def contract(detail: dict[str, Any]) -> TensorContract:
            scale, point = detail.get("quantization", (0.0, 0))
            return TensorContract(tuple(int(x) for x in detail["shape"]), _name(detail["dtype"]), float(scale), int(point))
        names = {
            value: name for name, value in vars(schema.BuiltinOperator).items()
            if name.isupper() and isinstance(value, int)
        }
        codes = [root.OperatorCodes(index) for index in range(root.OperatorCodesLength())]
        operators: list[Operator] = []
        for graph_index in range(root.SubgraphsLength()):
            graph = root.Subgraphs(graph_index)
            for operation_index in range(graph.OperatorsLength()):
                code = codes[graph.Operators(operation_index).OpcodeIndex()]
                raw_custom = code.CustomCode()
                custom = raw_custom.decode("utf-8") if isinstance(raw_custom, bytes) else raw_custom
                builtin = names.get(code.BuiltinCode(), f"BUILTIN_{code.BuiltinCode()}")
                operators.append(Operator(builtin, int(code.Version()), custom))
        return Inspection(
            input=contract(input_detail), output=contract(output_detail),
            operators=tuple(operators),
            activation_dtypes=tuple(sorted({_name(row["dtype"]) for row in tensors})),
            metadata={"tensor_count": len(tensors)},
        )

    def infer_litert(self, flatbuffer: Path, sample: torch.Tensor) -> Inference:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-not-found]
            interpreter = Interpreter(model_path=str(flatbuffer))
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            scale, zero_point = input_detail["quantization"]
            if not scale:
                raise ValueError("input tensor is not quantized")
            values = sample.detach().cpu().numpy()
            values = np.clip(np.rint(values / scale + zero_point), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_detail["index"], values)
            interpreter.invoke()
            output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
            return Inference(tuple(int(value) for value in output.shape), _name(output.dtype))
        except PreflightError:
            raise
        except Exception as exc:
            raise PreflightError("litert_inference", str(exc)) from exc

    def probe_tflm(self, flatbuffer: Path, operators: Sequence[Operator], working_directory: Path) -> Arena:
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            message = getattr(exc, "stderr", None) or str(exc)
            raise PreflightError("tflm_probe", f"Docker prerequisite failed: {message}") from exc
        # The pinned image builds the probe against the exact TFLM commit.  The
        # generated resolver is intentionally limited to inspected operations.
        resolver = ",".join(sorted({operator.builtin for operator in operators}))
        command = ["docker", "run", "--rm", "-v", f"{flatbuffer.parent.resolve()}:/work:ro", TFLM_PROBE_IMAGE,
                   "tflm-probe", "--model", f"/work/{flatbuffer.name}", "--commit", TFLM_COMMIT,
                   "--ops", resolver, "--interpreter", "RecordingMicroInterpreter"]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            value = json.loads(completed.stdout)
            return Arena(**{key: int(value[key]) for key in ("head_bytes", "temporary_bytes", "tail_bytes", "total_bytes")})
        except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PreflightError("tflm_probe", f"TFLM probe failed: {exc}") from exc


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _macs(model: nn.Module) -> int:
    total = 0
    handles = []
    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += int(output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3] * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1])
        elif isinstance(module, nn.Linear):
            total += int(output.shape[0] * module.in_features * module.out_features)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(torch.zeros((1, 1, 48, 48), dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
    return total


def _calibration(data: TrainingData) -> tuple[torch.Tensor, dict[str, Any]]:
    records = data.train.records
    selected = []
    random_source = random.Random(CALIBRATION_SEED)
    for label, label_name in enumerate(LABELS):
        candidates = [record for record in records if record.label == label]
        if len(candidates) < CALIBRATION_PER_LABEL:
            raise PreflightError("calibration", f"Training has {len(candidates)} {label_name} samples; requires {CALIBRATION_PER_LABEL}", observed=len(candidates), limit=CALIBRATION_PER_LABEL)
        selected.extend(random_source.sample(candidates, CALIBRATION_PER_LABEL))
    # The dataset passed by the data layer has augmentation disabled here.
    from .data import FER2013Dataset
    dataset = FER2013Dataset(data.provenance.path, tuple(selected), augmentation=None)
    values = torch.stack([dataset[index][0] for index in range(len(dataset))])
    identifiers = [str(record.line_number) for record in selected]
    return values.permute(0, 2, 3, 1).contiguous(), {
        "seed": CALIBRATION_SEED, "per_label": CALIBRATION_PER_LABEL, "total": len(selected),
        "counts": {label: CALIBRATION_PER_LABEL for label in LABELS},
        "identifier_sha256": hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest(),
    }


def _validate_inspection(inspection: Inspection) -> None:
    for name, actual, expected in (("input", inspection.input, ((1, 48, 48, 1), "int8")), ("output", inspection.output, ((1, 4), "int8"))):
        if actual.shape != expected[0] or actual.dtype != expected[1]:
            raise PreflightError("contract", f"{name} contract is {actual.shape}/{actual.dtype}; expected {expected[0]}/{expected[1]}", observed=asdict(actual), limit={"shape": expected[0], "dtype": expected[1]})
        if not math.isfinite(actual.scale) or actual.scale <= 0:
            raise PreflightError("contract", f"{name} quantization scale must be finite and positive", observed=actual.scale)
    if any(dtype in {"float", "float32", "fp32"} for dtype in inspection.activation_dtypes):
        raise PreflightError("contract", "FP32 activation found in computational path", observed=list(inspection.activation_dtypes))
    unsupported = [asdict(operator) for operator in inspection.operators if operator.custom_code is not None or operator.builtin not in ALLOWED_BUILTINS or operator.version > MAX_OPERATOR_VERSIONS.get(operator.builtin, 0)]
    if unsupported:
        raise PreflightError("operators", "unsupported, custom, or incompatible operator version", observed=unsupported, limit=sorted(ALLOWED_BUILTINS))


def _candidate_report(name: str, baseline_parameters: int, baseline_macs: int) -> dict[str, Any]:
    return {"name": name, "status": "pending", "baseline": {"parameters": baseline_parameters, "macs": baseline_macs}}


def run_preflight(config: AppConfig, data: TrainingData, directory: Path, toolchain: PreflightToolchain | None = None) -> dict[str, Any]:
    """Produce deployment evidence for every DS-CNN width, or raise after all fail.

    ``directory`` must not exist.  This prevents a partial preflight from ever
    changing an already finalized run artifact.
    """
    if directory.exists():
        raise ValueError(f"preflight directory already exists: {directory}")
    directory.mkdir(parents=True)
    toolchain = toolchain or _DefaultToolchain()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "status": "running", "labels": list(LABELS),
        "toolchain": toolchain.versions(),
        "tflm": {"commit": TFLM_COMMIT, "image": TFLM_PROBE_IMAGE},
        "budgets": {"tflite_bytes": MAX_TFLITE_BYTES, "macs": MAX_MACS, "arena_with_margin_bytes": MAX_ARENA_BYTES, "arena_margin": 1.20},
        "candidates": [],
    }
    try:
        calibration, calibration_report = _calibration(data)
        report["calibration"] = calibration_report
    except PreflightError as exc:
        report.update({"status": "failed", "failure_count": 1, "calibration": {"failed_stage": exc.stage, "error": str(exc), "observed": exc.observed, "limit": exc.limit}})
        (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise PreflightFailed(directory) from exc
    failures = []
    for baseline in DSCNN_BASELINES:
        item = _candidate_report(baseline.name, baseline.parameters, baseline.macs)
        report["candidates"].append(item)
        try:
            model = build_candidate(baseline.name, config.model.num_classes).eval()
            parameters, macs = _parameter_count(model), _macs(model)
            item.update({"parameters": parameters, "macs": macs})
            if parameters != baseline.parameters or macs != baseline.macs:
                raise PreflightError("structure", "DS-CNN baseline mismatch", observed={"parameters": parameters, "macs": macs}, limit={"parameters": baseline.parameters, "macs": baseline.macs})
            if macs > MAX_MACS:
                raise PreflightError("budget", "MAC budget exceeded", observed=macs, limit=MAX_MACS)
            artifact = directory / f"{baseline.name}.int8.tflite"
            toolchain.export_quantize(NHWCModel(model).eval(), calibration, artifact)
            inspection = toolchain.inspect(artifact)
            item["inspection"] = {"input": asdict(inspection.input), "output": asdict(inspection.output), "operators": [asdict(operator) for operator in inspection.operators], "activation_dtypes": list(inspection.activation_dtypes), "metadata": inspection.metadata}
            _validate_inspection(inspection)
            size = artifact.stat().st_size
            item["tflite_bytes"] = size
            if size > MAX_TFLITE_BYTES:
                raise PreflightError("budget", "FlatBuffer budget exceeded", observed=size, limit=MAX_TFLITE_BYTES)
            litert = toolchain.infer_litert(artifact, calibration[:1])
            item["litert_inference"] = asdict(litert)
            if litert.shape != (1, 4) or litert.dtype != "int8":
                raise PreflightError("litert_inference", "LiteRT inference contract failed", observed=asdict(litert))
            arena = toolchain.probe_tflm(artifact, inspection.operators, directory / f".{baseline.name}.probe")
            margin = math.ceil(arena.total_bytes * 1.20)
            item["tflm"] = {**asdict(arena), "arena_with_margin_bytes": margin}
            if margin > MAX_ARENA_BYTES:
                raise PreflightError("budget", "TFLM arena budget exceeded", observed=margin, limit=MAX_ARENA_BYTES)
            item["status"] = "passed"
            item["artifact"] = artifact.name
            item["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except PreflightError as exc:
            item.update({"status": "failed", "failed_stage": exc.stage, "error": str(exc), "observed": exc.observed, "limit": exc.limit})
            failures.append(item)
        except Exception as exc:
            item.update({"status": "failed", "failed_stage": "unexpected", "error": f"{type(exc).__name__}: {exc}"})
            failures.append(item)
    report["status"] = "failed" if failures else "passed"
    report["failure_count"] = len(failures)
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise PreflightFailed(directory)
    return report
