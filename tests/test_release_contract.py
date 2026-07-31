from __future__ import annotations

from pathlib import Path

import pytest

from sentience_train.cli import _parser
from sentience_train.export import ExportInputError, ExportResult, export_frozen_run


def test_export_cli_defaults_to_the_complete_release() -> None:
    args = _parser().parse_args(["export", "--run-id", "run-1"])

    assert args.format == "all"
    assert args.run_id == "run-1"


def test_export_frozen_run_rejects_a_missing_frozen_run(tmp_path: Path) -> None:
    with pytest.raises(ExportInputError):
        export_frozen_run("run-1", runs_root=tmp_path / "runs", exports_root=tmp_path / "exports")


def test_export_result_exposes_release_state() -> None:
    result = ExportResult("run-1", "all", Path("exports/run-1/all"), False, "candidate", "accepted", True)

    assert result.status == "accepted"
    assert result.test_accessed is True
