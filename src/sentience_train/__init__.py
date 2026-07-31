"""Deterministic, self-contained FER2013 training package.

Importing this package is intentionally side-effect free: no directories are
created and CUDA is never probed until a training run is requested.
"""

from .config import (
    AppConfig,
    AugmentationConfig,
    ConfigError,
    DataConfig,
    DistillationConfig,
    EarlyStoppingConfig,
    ModelConfig,
    OptimizerConfig,
    QATConfig,
    QualityGatesConfig,
    QuantizationConfig,
    RunConfig,
    SelectionConfig,
    TrainingConfig,
    load_config,
)

__all__ = [
    "AppConfig", "AugmentationConfig", "ConfigError", "DataConfig",
    "DistillationConfig", "EarlyStoppingConfig", "ModelConfig", "OptimizerConfig",
    "QATConfig", "QualityGatesConfig", "QuantizationConfig", "RunConfig",
    "SelectionConfig", "TrainingConfig", "load_config",
]
