"""Deterministic FER2013 classification with an ONNX-only release contract."""

from __future__ import annotations

import sys

from . import engine as engine
from .core import config as config
from .core import data as data
from .core import metrics as metrics
from .core import models as models
from .core import recipes as recipes
from .engine import artifacts as artifacts
from .onnx import benchmark as benchmark
from .onnx import onnx_artifact as onnx_artifact
from .workflows import campaign as campaign
from .workflows import export as export
from .workflows import release as release
from .workflows import training as training

_LEGACY_MODULES = {
    "config": config,
    "data": data,
    "metrics": metrics,
    "models": models,
    "recipes": recipes,
    "artifacts": artifacts,
    "benchmark": benchmark,
    "onnx_artifact": onnx_artifact,
    "campaign": campaign,
    "export": export,
    "release": release,
    "training": training,
}

for _name, _module in _LEGACY_MODULES.items():
    sys.modules[f"{__name__}.{_name}"] = _module
