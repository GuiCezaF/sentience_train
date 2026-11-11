"""Immutable staging and finalisation of training run artifacts."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class RunError(RuntimeError):
    pass


@dataclass
class RunStaging:
    root: Path
    staging: Path
    final: Path
    run_id: str

    @classmethod
    def create(cls, root: Path) -> "RunStaging":
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_id = uuid.uuid4().hex[:8]
        run_id = f"{stamp}-{short_id}"
        final = root / run_id
        staging = root / ".staging" / run_id
        if final.exists() or staging.exists():
            raise RunError(f"run collision for {run_id}; refusing to overwrite")
        staging.mkdir(parents=True)
        for relative in ("checkpoints", "metrics", "logs"):
            (staging / relative).mkdir()
        return cls(root=root, staging=staging, final=final, run_id=run_id)

    def write_resolved_config(self, config: dict[str, Any]) -> None:
        (self.staging / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def write_manifest(self, document: dict[str, Any]) -> None:
        (self.staging / "manifest.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def stage_checkpoint(self, source: Path) -> Path:
        destination = self.staging / "checkpoint.pt"
        shutil.copy2(source, destination)
        return destination

    def finalize(self) -> Path:
        if self.final.exists():
            raise RunError(f"run destination exists: {self.final}; refusing to overwrite")
        os.replace(self.staging, self.final)
        try:
            self.staging.parent.rmdir()
        except OSError:
            pass
        return self.final

    def abort(self) -> None:
        shutil.rmtree(self.staging, ignore_errors=True)
        try:
            self.staging.parent.rmdir()
        except OSError:
            pass
