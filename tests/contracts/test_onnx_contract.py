from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sentience_train.benchmark import SyntheticBenchmarkAdapter
from sentience_train.models import DSCNN
from sentience_train.onnx_artifact import OnnxArtifact, onnx_contract


@pytest.mark.parametrize("batch", [1, 8])
def test_contract_declares_dynamic_n_and_uint8_nhwc(batch: int) -> None:
    contract = onnx_contract()
    assert contract["input"]["shape"] == ["N", 48, 48, 1]
    assert contract["output"]["shape"] == ["N", 4]
    assert contract["input"]["dtype"] == "uint8"
    assert contract["normalization"]["formula"] == "float32(pixel) / 127.5 - 1.0"
    assert batch in (1, 8)


def test_real_onnx_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    evidence = OnnxArtifact().build_and_validate(
        DSCNN(), tmp_path / "model.onnx", samples=np.zeros((8, 48, 48, 1), dtype=np.uint8)
    )
    assert evidence.batches == (1, 8)
    assert evidence.class_agreement == 1.0
    assert evidence.max_absolute_error <= 1e-5
    assert evidence.size < 52_428_800


def test_synthetic_benchmark_is_marked() -> None:
    report = SyntheticBenchmarkAdapter().run(Path("synthetic.onnx"))
    assert report.synthetic is True
    assert report.passed is True
