"""Public export facade for the single ONNX release path."""

from __future__ import annotations

from pathlib import Path

from ..core.config import load_training_config
from .release import ReleaseError, ReleasePublisher, ReleaseResult


class ExportValidationError(ReleaseError):
    pass


def export_frozen_run(
    run_id: str, *, config: str | Path = "configs/default.yaml", publisher: ReleasePublisher | None = None
) -> ReleaseResult:
    try:
        training = load_training_config(config)
        return (publisher or ReleasePublisher()).publish(run_id, training=training)
    except ReleaseError:
        raise
    except Exception as exc:
        raise ExportValidationError(str(exc)) from exc
