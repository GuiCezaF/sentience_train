"""Reproducible optimization recipes used by tuning and full training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AugmentationConfig:
    flip: float = 0.5
    rotation: float = 5.0
    translate: float = 0.04
    scale_min: float = 0.98
    scale_max: float = 1.02
    brightness: float = 0.10
    contrast: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def horizontal_flip_probability(self) -> float:
        return self.flip

    @property
    def rotation_degrees(self) -> float:
        return self.rotation

    @property
    def translate_fraction(self) -> float:
        return self.translate


@dataclass(frozen=True)
class RecipeSpec:
    trial_id: str
    augmentation: AugmentationConfig
    loss: str
    class_weight_exponent: float | None
    focal_gamma: float | None
    learning_rate: str
    policy: str
    label_smoothing: float = 0.0
    warmup_epochs: int = 0
    gradient_clip_norm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "augmentation": self.augmentation.to_dict(),
            "loss": self.loss,
            "class_weight_exponent": self.class_weight_exponent,
            "focal_gamma": self.focal_gamma,
            "learning_rate": self.learning_rate,
            "policy": self.policy,
            "label_smoothing": self.label_smoothing,
            "warmup_epochs": self.warmup_epochs,
            "gradient_clip_norm": self.gradient_clip_norm,
        }


def default_recipe_grid() -> tuple[RecipeSpec, ...]:
    """Return the frozen experiment grid.

    T4 remains the strongest observed baseline. T5 is its controlled stability
    variant: identical data/loss/optimizer policy plus light label smoothing,
    learning-rate warmup, and gradient clipping.
    """
    moderate = AugmentationConfig()
    strong = AugmentationConfig(
        rotation=10.0,
        translate=0.08,
        scale_min=0.95,
        scale_max=1.05,
        brightness=0.15,
        contrast=0.15,
    )
    return (
        RecipeSpec("T0", moderate, "weighted_ce", 0.5, None, "base", "full"),
        RecipeSpec("T1", moderate, "weighted_ce", 0.5, None, "base", "classifier-only"),
        RecipeSpec("T2", moderate, "weighted_ce", 0.5, None, "low", "full"),
        RecipeSpec("T3", moderate, "weighted_ce", 0.5, None, "low", "classifier-only"),
        RecipeSpec("T4", strong, "weighted_ce", 0.5, None, "base", "full"),
        RecipeSpec(
            "T5",
            strong,
            "weighted_ce",
            0.5,
            None,
            "base",
            "full",
            label_smoothing=0.05,
            warmup_epochs=5,
            gradient_clip_norm=1.0,
        ),
        RecipeSpec("T6", moderate, "focal", 0.5, 2.0, "base", "full"),
        RecipeSpec("T7", moderate, "focal", 0.5, 2.0, "base", "classifier-only"),
        RecipeSpec("T8", moderate, "balanced_softmax", None, None, "base", "full"),
    )


def learning_rate_for(architecture: str, recipe: RecipeSpec) -> float:
    try:
        value = {"DS-CNN": 1e-3, "MobileNetV3-Small": 3e-4, "ResNet-18": 1e-4}[architecture]
    except KeyError as exc:
        raise ValueError(f"unsupported architecture {architecture!r}") from exc
    return value * (0.5 if recipe.learning_rate == "low" else 1.0)


def scheduled_learning_rate(base: float, epoch: int, total_epochs: int, warmup_epochs: int = 0) -> float:
    """Linear warmup followed by cosine decay to one percent of ``base``."""
    if epoch < 1 or total_epochs < 1 or epoch > total_epochs:
        raise ValueError("epoch must be between 1 and total_epochs")
    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup_epochs must be between 0 and total_epochs - 1")
    if warmup_epochs and epoch <= warmup_epochs:
        return base * epoch / warmup_epochs
    decay_steps = max(total_epochs - warmup_epochs - 1, 1)
    progress = min(max((epoch - warmup_epochs - 1) / decay_steps, 0.0), 1.0)
    return 0.01 * base + 0.99 * base * 0.5 * (1.0 + math.cos(math.pi * progress))


__all__ = [
    "AugmentationConfig",
    "RecipeSpec",
    "default_recipe_grid",
    "learning_rate_for",
    "scheduled_learning_rate",
]
