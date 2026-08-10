from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentience_train.cli import main
from sentience_train.config import load_training_config
from sentience_train.engine import SeedEvidence, TrainingRequest
from sentience_train.metrics import GateResult, Metrics
from sentience_train.training import execute_training


def _passing_metrics() -> Metrics:
    return Metrics(
        0.9,
        0.9,
        0.1,
        (0.9,) * 4,
        (0.9,) * 4,
        (0.9,) * 4,
        (1,) * 4,
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        "PublicTest",
    )


class _PassingEngine:
    def __init__(self) -> None:
        self.requests: list[TrainingRequest] = []

    def run(self, request: TrainingRequest) -> SeedEvidence:
        self.requests.append(request)
        request.output.mkdir(parents=True, exist_ok=True)
        checkpoint = request.output / "checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{request.seed}".encode())
        metrics = _passing_metrics()
        return SeedEvidence(
            request.architecture,
            request.trial.trial_id,
            request.seed,
            checkpoint,
            metrics,
            GateResult(True, {"margin": 0.2}, {}, ()),
            1,
            {"seed": request.seed},
        )


class _FailingEngine:
    def run(self, request: TrainingRequest) -> SeedEvidence:
        raise RuntimeError("trainer failed")


def _config_with_campaign(tmp_path: Path, status: str = "completed"):
    config = load_training_config("configs/default.yaml")
    outputs = replace(
        config.output,
        campaigns_dir=tmp_path / "artifacts" / "campaigns",
        runs_dir=tmp_path / "artifacts" / "runs",
        exports_dir=tmp_path / "artifacts" / "exports",
    )
    config = replace(config, output=outputs)
    campaign = outputs.campaigns_dir / "campaign"
    campaign.mkdir(parents=True)
    document = {
        "schema": "tuning-campaign/v2",
        "campaign_id": "campaign",
        "status": status,
        "shortlist": [{"architecture": "DS-CNN", "recipe_id": "T5"}] if status == "completed" else [],
        "batch_choices": {"DS-CNN": {"selected": 128, "mode": "configured", "probes": []}},
    }
    if status == "exhausted":
        document["error"] = "no promoted recipe passed all 3 seeds"
    (campaign / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    return config


def test_training_interface_publishes_a_complete_atomic_run(tmp_path: Path) -> None:
    config = _config_with_campaign(tmp_path)
    engine = _PassingEngine()
    data = SimpleNamespace(
        provenance=SimpleNamespace(
            path=Path("dataset.csv"),
            sha256="dataset-sha",
            partition_counts={"Training": 4, "PublicTest": 4, "PrivateTest": 4},
        )
    )

    run = execute_training(config, engine=engine, data=data)

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["recipe_id"] == "T5"
    assert manifest["checkpoint"] == "checkpoint.pt"
    assert (run / "checkpoint.pt").is_file()
    assert (run / "config.resolved.yaml").is_file()
    assert [request.seed for request in engine.requests] == [42, 43, 44]
    assert not (config.output.runs_dir / ".staging").exists()


def test_training_interface_rolls_back_staging_after_failure(tmp_path: Path) -> None:
    config = _config_with_campaign(tmp_path)
    data = SimpleNamespace(
        provenance=SimpleNamespace(path=Path("dataset.csv"), sha256="dataset-sha", partition_counts={})
    )

    with pytest.raises(RuntimeError, match="trainer failed"):
        execute_training(config, engine=_FailingEngine(), data=data)

    assert not (config.output.runs_dir / ".staging").exists()
    assert not any(path.is_dir() and not path.name.startswith(".") for path in config.output.runs_dir.iterdir())


def test_cli_classifies_an_exhausted_campaign_as_training_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    config = _config_with_campaign(tmp_path, status="exhausted")
    monkeypatch.setattr("sentience_train.cli.load_training_config", lambda path: config)

    exit_code = main(["train", "--config", "ignored.yaml"])

    assert exit_code == 1
    assert "training error: tuning campaign campaign is exhausted" in capsys.readouterr().err
