from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentience_train.release import ReleaseError


def test_old_run_schema_is_rejected(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"schema": "frozen-" + "run/v1"}), encoding="utf-8")
    with pytest.raises(ReleaseError, match="old or unsupported"):
        from sentience_train.release import _resolve_run

        _resolve_run("run-1", runs)


def test_run_id_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="direct child"):
        from sentience_train.release import _resolve_run

        _resolve_run("../outside", tmp_path / "runs")
