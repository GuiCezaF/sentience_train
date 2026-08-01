from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def test_cli_exposes_only_the_two_public_workflows_and_formats() -> None:
    assert (
        _parser().parse_args(["train", "--config", "config.yaml", "--epochs", "1", "--device", "cpu"]).command
        == "train"
    )
    assert (
        _parser().parse_args(["export", "--run-id", "run-1", "--format", "onnx", "--runs-dir", "runs"]).format == "onnx"
    )
    for command in ("evaluate", "crossval", "baseline", "compare"):
        with pytest.raises(SystemExit):
            _parser().parse_args([command])


def test_cli_rejects_legacy_configuration_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    document["model"]["name"] = "ds_cnn_w025"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    from sentience_train.config import ConfigError, load_config

    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)
