from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import sentience_train.engine as engine_module
from sentience_train.campaign import BatchAutotuner, DefaultCampaignRunner, ResolvedPlan
from sentience_train.config import load_training_config, load_tuning_config
from sentience_train.data import TrainingData, TuningData
from sentience_train.engine import SeedEvidence, TorchTrainingEngine, TrainingRequest, _loss
from sentience_train.metrics import GateResult, Metrics
from sentience_train.training import CampaignExhaustedError, execute_training


class _TinyDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self) -> None:
        self._records = tuple(SimpleNamespace(label=label) for label in range(4))
        self.augmentations: list[Any] = []

    @property
    def records(self) -> tuple[Any, ...]:
        return self._records

    def with_augmentation(self, augmentation: Any) -> "_TinyDataset":
        self.augmentations.append(augmentation)
        return self

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.full((1, 2, 2), float(index) / 4.0), self._records[index].label


def _cpu_config(tmp_path: Path):
    config = load_training_config("configs/default.yaml")
    settings = replace(
        config.training,
        batch_size=4,
        workers=0,
        device="cpu",
        amp=False,
        tf32=False,
        cudnn_benchmark=False,
        channels_last=False,
        triage_epochs=1,
        triage_patience=1,
    )
    output = replace(config.output, campaigns_dir=tmp_path / "campaigns")
    return replace(config, training=settings, output=output)


@pytest.mark.parametrize(
    ("kind", "expected_partition"),
    (("tuning", "Tuning"), ("training", "PublicTest")),
)
def test_engine_applies_trial_augmentation_and_labels_validation_partition(
    tmp_path: Path, kind: str, expected_partition: str
) -> None:
    config = _cpu_config(tmp_path)
    trial = load_tuning_config("configs/tuning.yaml").trial_grid[0]
    optimization = _TinyDataset()
    validation = _TinyDataset()
    if kind == "tuning":
        data = TuningData(optimization, validation, SimpleNamespace(), SimpleNamespace())
    else:
        data = TrainingData(optimization, validation, SimpleNamespace(), tuple(range(4)))
    trainer = TorchTrainingEngine(model_factory=lambda *args, **kwargs: nn.Sequential(nn.Flatten(), nn.Linear(4, 4)))

    evidence = trainer.run(TrainingRequest(config, "DS-CNN", trial, 42, data, tmp_path / kind, False, 4))

    assert optimization.augmentations == [trial.augmentation]
    assert evidence.metrics.partition == expected_partition


def _metrics(score: float) -> Metrics:
    return Metrics(
        score,
        score,
        0.1,
        (score,) * 4,
        (score,) * 4,
        (score,) * 4,
        (1,) * 4,
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        "Tuning",
    )


class _EvidenceEngine:
    def __init__(self, passes: Callable[[TrainingRequest], bool] | None = None) -> None:
        self.passes = passes or (lambda request: True)
        self.requests: list[TrainingRequest] = []

    def run(self, request: TrainingRequest) -> SeedEvidence:
        self.requests.append(request)
        passed = self.passes(request)
        score = 0.9 if passed else 0.1
        return SeedEvidence(
            request.architecture,
            request.trial.trial_id,
            request.seed,
            request.output / "checkpoint.pt",
            _metrics(score),
            GateResult(passed, {"margin": score - 0.7}, {}, () if passed else ("accuracy",)),
            1,
            {"seed": request.seed},
        )


class _InterruptingEngine(_EvidenceEngine):
    def run(self, request: TrainingRequest) -> SeedEvidence:
        if len(self.requests) == 1:
            self.requests.append(request)
            raise KeyboardInterrupt
        return super().run(request)


def _runner_inputs(tmp_path: Path):
    training = load_training_config("configs/default.yaml")
    training = replace(training, output=replace(training.output, campaigns_dir=tmp_path / "campaigns"))
    tuning = load_tuning_config("configs/tuning.yaml")
    tuning = replace(tuning, trial_grid=(tuning.trial_grid[0],), seeds=(42, 43, 44))
    plan = ResolvedPlan({"schema": "tuning-plan/v2"}, "a" * 64)
    tuner = BatchAutotuner(probe=lambda batch: (0, 100, True))
    return training, tuning, plan, tuner


def _fake_tuning_data() -> SimpleNamespace:
    return SimpleNamespace()


def test_runner_aggregates_all_promoted_seeds_and_records_terminal_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, tuning, plan, tuner = _runner_inputs(tmp_path)
    monkeypatch.setattr("sentience_train.campaign.load_tuning_data", lambda *args, **kwargs: _fake_tuning_data())
    engine = _EvidenceEngine(lambda request: request.seed == 42)
    runner = DefaultCampaignRunner(
        tuning,
        training,
        engine=engine,
        batch_autotuner=tuner,
        campaign_directory=training.output.campaigns_dir / "campaign",
        preflight=lambda architecture: True,
    )

    manifest = runner.execute(plan).document

    assert manifest["status"] == "exhausted"
    assert manifest["shortlist"] == []
    assert "no promoted recipe passed all 3 seeds" in manifest["error"]
    assert {item["seed"] for item in manifest["evidence"]} == {42, 43, 44}


def test_runner_never_trains_an_architecture_that_failed_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, tuning, plan, tuner = _runner_inputs(tmp_path)
    monkeypatch.setattr("sentience_train.campaign.load_tuning_data", lambda *args, **kwargs: _fake_tuning_data())
    engine = _EvidenceEngine()
    runner = DefaultCampaignRunner(
        tuning,
        training,
        engine=engine,
        batch_autotuner=tuner,
        campaign_directory=training.output.campaigns_dir / "campaign",
        preflight=lambda architecture: architecture != "MobileNetV3-Small",
    )

    runner.execute(plan)

    assert "MobileNetV3-Small" not in {request.architecture for request in engine.requests}


def test_running_campaign_persists_progress_and_does_not_repeat_completed_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, tuning, plan, tuner = _runner_inputs(tmp_path)
    monkeypatch.setattr("sentience_train.campaign.load_tuning_data", lambda *args, **kwargs: _fake_tuning_data())
    directory = training.output.campaigns_dir / "campaign"
    interrupted = _InterruptingEngine()
    first_runner = DefaultCampaignRunner(
        tuning,
        training,
        engine=interrupted,
        batch_autotuner=tuner,
        campaign_directory=directory,
        preflight=lambda architecture: True,
    )

    with pytest.raises(KeyboardInterrupt):
        first_runner.execute(plan)

    saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    completed = {(item["architecture"], item["recipe_id"], item["seed"]) for item in saved["evidence"]}
    assert completed == {("DS-CNN", "T0", 42)}

    resumed = _EvidenceEngine()
    second_runner = DefaultCampaignRunner(
        tuning,
        training,
        engine=resumed,
        batch_autotuner=tuner,
        campaign_directory=directory,
        preflight=lambda architecture: True,
    )
    second_runner.execute(plan)

    repeated = {(item.architecture, item.trial.trial_id, item.seed) for item in resumed.requests}
    assert ("DS-CNN", "T0", 42) not in repeated


def test_train_explains_an_exhausted_campaign_without_a_stored_error(tmp_path: Path) -> None:
    training = load_training_config("configs/default.yaml")
    campaigns = tmp_path / "campaigns"
    campaign = campaigns / "exhausted"
    campaign.mkdir(parents=True)
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "tuning-campaign/v2",
                "status": "exhausted",
                "shortlist": [],
            }
        ),
        encoding="utf-8",
    )
    config = replace(training, output=replace(training.output, campaigns_dir=campaigns))

    with pytest.raises(CampaignExhaustedError, match="exhausted.*no recipe met all predictive gates"):
        execute_training(config)


def test_mobile_classifier_warmup_selects_the_actual_last_classifier_layer() -> None:
    model = nn.Module()
    model.classifier = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Dropout(), nn.Linear(4, 4))

    engine_module._freeze_for_classifier_warmup(model, "MobileNetV3-Small")

    assert not model.classifier[0].weight.requires_grad
    assert model.classifier[3].weight.requires_grad


def test_zero_class_weight_exponent_is_not_replaced_by_the_default() -> None:
    trial = replace(load_tuning_config("configs/tuning.yaml").trial_grid[0], class_weight_exponent=0.0)

    criterion = _loss(trial, (0, 0, 0, 1, 2, 3))

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert torch.equal(criterion.weight, torch.ones(4))
