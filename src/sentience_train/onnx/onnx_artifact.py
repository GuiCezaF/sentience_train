"""ONNX FP32 artifact construction and public-contract validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..core.config import NORMALIZATION_FORMULA, NORMALIZATION_ORDER, ONNX_CONTRACT_SCHEMA
from ..core.models import PublicOnnxWrapper

ONNX_OPSET = 18


class OnnxError(RuntimeError):
    pass


@dataclass(frozen=True)
class OnnxEvidence:
    path: Path
    sha256: str
    size: int
    contract: dict[str, Any]
    batches: tuple[int, ...]
    class_agreement: float
    max_absolute_error: float
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.size,
            "contract": self.contract,
            "batches": list(self.batches),
            "class_agreement": self.class_agreement,
            "max_absolute_error": self.max_absolute_error,
            "synthetic": self.synthetic,
        }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def onnx_contract() -> dict[str, Any]:
    return {
        "schema": ONNX_CONTRACT_SCHEMA,
        "input": {"name": "images", "dtype": "uint8", "shape": ["N", 48, 48, 1], "layout": "NHWC"},
        "output": {"name": "probabilities", "dtype": "float32", "shape": ["N", 4], "kind": "probabilities"},
        "normalization": {
            "formula": NORMALIZATION_FORMULA,
            "order": list(NORMALIZATION_ORDER),
            "divisor": 127.5,
            "offset": 1.0,
        },
        "opset": ONNX_OPSET,
        "external_data": False,
    }


class OnnxArtifact:
    def __init__(self, *, exporter: Any | None = None, runtime: Any | None = None) -> None:
        self.exporter = exporter
        self.runtime = runtime

    def build_and_validate(
        self, model: nn.Module, destination: Path, *, samples: np.ndarray | None = None
    ) -> OnnxEvidence:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise OnnxError("ONNX staging path already exists; the artifact is built exactly once")
        wrapper = PublicOnnxWrapper(model).eval()
        example = torch.zeros((1, 48, 48, 1), dtype=torch.uint8)
        try:
            if self.exporter is not None:
                self.exporter(wrapper, example, destination)
            else:
                torch.onnx.export(
                    wrapper,
                    example,
                    destination,
                    opset_version=ONNX_OPSET,
                    input_names=["images"],
                    output_names=["probabilities"],
                    dynamic_axes={"images": {0: "N"}, "probabilities": {0: "N"}},
                    do_constant_folding=True,
                    external_data=False,
                    dynamo=False,
                )
        except Exception as exc:
            raise OnnxError(f"ONNX export failed: {exc}") from exc
        self._validate_graph(destination)
        agreement, error = self._validate_runtime(wrapper, destination, samples)
        return OnnxEvidence(
            destination, _sha(destination), destination.stat().st_size, onnx_contract(), (1, 8), agreement, error
        )

    @staticmethod
    def _validate_graph(path: Path) -> None:
        try:
            import onnx

            model = onnx.load(path, load_external_data=False)
            onnx.checker.check_model(model)
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as exc:
            raise OnnxError(f"ONNX checker or shape inference failed: {exc}") from exc
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise OnnxError("public ONNX graph must have exactly one input and one output")
        input_value = model.graph.input[0]
        output_value = model.graph.output[0]
        if input_value.name != "images" or output_value.name != "probabilities":
            raise OnnxError("public ONNX names must be images and probabilities")
        tensor_type = input_value.type.tensor_type
        if tensor_type.elem_type != 2:
            raise OnnxError("public ONNX input must be uint8")
        dims = tensor_type.shape.dim
        if (
            len(dims) != 4
            or dims[1].dim_value != 48
            or dims[2].dim_value != 48
            or dims[3].dim_value != 1
            or not dims[0].dim_param
        ):
            raise OnnxError("input shape must be dynamic N,48,48,1")
        out_type = output_value.type.tensor_type
        if out_type.elem_type != 1:
            raise OnnxError("public ONNX output must be FP32")
        out_dims = out_type.shape.dim
        if len(out_dims) != 2 or not out_dims[0].dim_param or out_dims[1].dim_value != 4:
            raise OnnxError("output shape must be dynamic N,4")
        ops = {node.op_type for node in model.graph.node}
        if not {"Cast", "Div", "Sub", "Transpose", "Softmax"}.issubset(ops):
            raise OnnxError("normalization, NHWC/NCHW layout, and softmax must be in the graph")

    def _validate_runtime(self, wrapper: nn.Module, path: Path, samples: np.ndarray | None) -> tuple[float, float]:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise OnnxError(f"onnxruntime is unavailable: {exc}") from exc
        session = self.runtime or ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
            sess_options=self._session_options(ort),
        )
        values = (
            samples
            if samples is not None
            else np.random.default_rng(42).integers(0, 256, (8, 48, 48, 1), dtype=np.uint8)
        )
        agreements: list[bool] = []
        errors: list[float] = []
        for batch in (1, 8):
            current = values[:batch]
            result = np.asarray(session.run(["probabilities"], {"images": current})[0])
            if result.shape != (batch, 4) or result.dtype != np.float32:
                raise OnnxError(f"runtime output must be FP32 [{batch},4]")
            if (
                not np.isfinite(result).all()
                or (result < -1e-7).any()
                or not np.allclose(result.sum(1), 1.0, atol=1e-5)
            ):
                raise OnnxError("probabilities must be finite, non-negative, and sum to one")
            with torch.no_grad():
                reference = wrapper(torch.from_numpy(current)).numpy()
            errors.append(float(np.max(np.abs(reference - result))))
            agreements.extend(np.argmax(reference, axis=1) == np.argmax(result, axis=1))
        return float(np.mean(agreements)), max(errors)

    @staticmethod
    def _session_options(ort: Any) -> Any:
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options
