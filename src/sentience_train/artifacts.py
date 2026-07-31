"""Immutable staging and finalisation of training run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


class RunError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    event = {"timestamp": utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def write_public_test_index(
    run_directory: Path,
    *,
    dataset_path: Path,
    dataset_sha256: str,
    records: Iterable[Any],
) -> Path:
    """Persist the only dataset records an export is allowed to evaluate."""
    destination = run_directory / "data" / "public_test.index.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "offset": int(record.offset),
            "label": int(record.label),
            "line_number": int(record.line_number),
            "row_sha256": str(record.row_sha256),
        }
        for record in records
    ]
    payload = {
        "schema": "public-test-index/v1",
        "partition": "PublicTest",
        "labels": ["angry", "happy", "neutral", "sad"],
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha256},
        "count": len(entries),
        "records": entries,
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def git_state(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    revision = command("git", "rev-parse", "HEAD")
    dirty = command("git", "status", "--porcelain")
    return {"revision": revision, "dirty": bool(dirty) if dirty is not None else None}


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
        (self.staging / "manifest.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def finalize(self) -> Path:
        if self.final.exists():
            raise RunError(f"run destination exists: {self.final}; refusing to overwrite")
        os.replace(self.staging, self.final)
        try:
            self.staging.parent.rmdir()
        except OSError:
            pass
        return self.final
