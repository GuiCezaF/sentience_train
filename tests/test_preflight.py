from __future__ import annotations

import pytest
import torch

from sentience_train.models import DSCNN_BASELINES, build_candidate
from sentience_train.preflight import (
    Inspection,
    Operator,
    PreflightError,
    TensorContract,
    _macs,
    _validate_inspection,
)


def _inspection(**changes):  # type: ignore[no-untyped-def]
    value = Inspection(
        input=TensorContract((1, 48, 48, 1), "int8", 0.01, 0),
        output=TensorContract((1, 4), "int8", 0.02, 0),
        operators=(Operator("CONV_2D", 3), Operator("RELU6", 1)),
        activation_dtypes=("int8", "int32"),
        metadata={},
    )
    return value.__class__(**{**value.__dict__, **changes})


def test_ds_cnn_structural_baselines_are_fixed() -> None:
    for baseline in DSCNN_BASELINES:
        model = build_candidate(baseline.name)
        assert sum(parameter.numel() for parameter in model.parameters()) == baseline.parameters
        assert _macs(model) == baseline.macs
        assert model(torch.zeros(1, 1, 48, 48)).shape == (1, 4)


@pytest.mark.parametrize(
    "inspection",
    [
        _inspection(input=TensorContract((1, 48, 48, 1), "float32", 0.01, 0)),
        _inspection(output=TensorContract((1, 4), "int8", 0.0, 0)),
        _inspection(operators=(Operator("MAX_POOL_2D", 1),)),
        _inspection(operators=(Operator("CONV_2D", 99),)),
        _inspection(activation_dtypes=("int8", "float32")),
    ],
)
def test_inspection_rejects_a_broken_int8_contract(inspection: Inspection) -> None:
    with pytest.raises(PreflightError):
        _validate_inspection(inspection)
