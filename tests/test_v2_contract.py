from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import sentience_train.models as models
from sentience_train.campaign import BatchAutotuner, DefaultCampaignPlanner
from sentience_train.config import (
    ARCHITECTURES,
    ConfigError,
    load_training_config,
    load_tuning_config,
)
from sentience_train.metrics import order_candidates
from sentience_train.models import DSCNN, adapt_rgb_kernel
from sentience_train.recipes import default_recipe_grid, scheduled_learning_rate
from sentience_train.training import CampaignExhaustedError, execute_training


def test_project_configs_and_grid_are_v2() -> None:
    training = load_training_config("configs/default.yaml")
    tuning = load_tuning_config("configs/tuning.yaml")
    root = training.source_path.parent.parent

    assert training.schema == "training-config/v2"
    assert tuning.architectures == ARCHITECTURES
    assert [trial.trial_id for trial in tuning.trial_grid] == [f"T{i}" for i in range(9)]
    assert training.data.csv_path == root / "data" / "raw" / "fer2013.csv"
    assert training.model.weights_dir == root / "models" / "pretrained"
    assert training.output.runs_dir == root / "artifacts" / "runs"
    assert training.output.campaigns_dir == root / "artifacts" / "campaigns"
    assert training.output.exports_dir == root / "artifacts" / "exports"


def test_t5_is_a_controlled_stability_variant_of_the_best_observed_recipe() -> None:
    recipes = {recipe.trial_id: recipe for recipe in default_recipe_grid()}
    baseline = recipes["T4"]
    stability = recipes["T5"]

    assert stability.augmentation == baseline.augmentation
    assert stability.loss == baseline.loss
    assert stability.class_weight_exponent == baseline.class_weight_exponent
    assert stability.learning_rate == baseline.learning_rate
    assert stability.policy == baseline.policy == "full"
    assert stability.label_smoothing == 0.05
    assert stability.warmup_epochs == 5
    assert stability.gradient_clip_norm == 1.0


def test_stability_recipe_uses_linear_warmup_then_cosine_decay() -> None:
    assert scheduled_learning_rate(1e-3, 1, 60, 5) == pytest.approx(2e-4)
    assert scheduled_learning_rate(1e-3, 5, 60, 5) == pytest.approx(1e-3)
    assert scheduled_learning_rate(1e-3, 60, 60, 5) == pytest.approx(1e-5)


def test_old_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({"schema": "training-config/v1"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_training_config(path)


def test_rgb_kernel_is_summed_without_channel_duplication() -> None:
    import torch

    value = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    assert torch.equal(adapt_rgb_kernel(value), value.sum(1, keepdim=True))


def test_ds_cnn_head_and_output() -> None:
    import torch

    model = DSCNN()
    assert model.head.weight.std().item() == pytest.approx(0.01, abs=0.01)
    assert model(torch.zeros(1, 1, 48, 48)).shape == (1, 4)


def test_build_resnet_adapts_grayscale_input_and_output_head(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    source = torch.nn.Module()
    source.conv1 = torch.nn.Conv2d(3, 2, 7, 2, 3, bias=False)
    source.fc = torch.nn.Linear(2, 1_000)
    checkpoint = source.state_dict()
    monkeypatch.setattr(models, "_image_model", lambda architecture: source)
    monkeypatch.setattr(models, "validate_weight_file", lambda architecture, weights_dir: Path("checkpoint"))
    monkeypatch.setattr(models, "_load_checkpoint", lambda path: checkpoint)

    model = models.build_model("ResNet-18")

    assert model.conv1.in_channels == 1
    assert model.fc.out_features == 4


def test_focal_loss_moves_class_weights_to_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to verify focal-loss device placement")

    from sentience_train.engine import _loss

    trial = next(item for item in default_recipe_grid() if item.loss == "focal")
    device = torch.device("cuda:0")
    criterion = _loss(trial, (0, 0, 1, 2, 3)).to(device)
    logits = torch.randn(5, 4, device=device, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 2, 3], device=device)

    value = criterion(logits, targets)
    value.backward()

    assert value.device == device


def test_stability_recipe_applies_label_smoothing() -> None:
    from sentience_train.engine import _loss

    recipe = next(item for item in default_recipe_grid() if item.trial_id == "T5")
    criterion = _loss(recipe, (0, 0, 0, 1, 1, 2, 3))

    assert criterion.label_smoothing == 0.05


def test_batch_probe_has_no_fallback_after_exhaustion() -> None:
    tuner = BatchAutotuner(probe=lambda batch: (100, 100, False))
    with pytest.raises(Exception, match="no batch probe"):
        tuner.choose("DS-CNN")


def test_tie_policy_is_stable() -> None:
    candidates = [
        {
            "recipe_id": "T8",
            "median_margin": 0.100,
            "median_p95_ms": 2,
            "median_rss_increment_bytes": 3,
            "median_onnx_bytes": 4,
        },
        {
            "recipe_id": "T1",
            "median_margin": 0.104,
            "median_p95_ms": 9,
            "median_rss_increment_bytes": 9,
            "median_onnx_bytes": 9,
        },
    ]
    assert order_candidates(candidates, limit=1)[0]["recipe_id"] == "T8"


def test_plan_is_read_only() -> None:
    training = load_training_config("configs/default.yaml")
    tuning = load_tuning_config("configs/tuning.yaml")
    plan = DefaultCampaignPlanner(tuning, training).resolve()
    assert plan.document["creates_campaign"] is False
    assert plan.document["schema"] == "tuning-plan/v2"
    assert len(plan.document["tasks"]) == 3 + 3 + 3 * 9 * 3 + 3


def test_plan_identity_changes_when_local_checkpoint_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training = load_training_config("configs/default.yaml")
    tuning = load_tuning_config("configs/tuning.yaml")
    weights_dir = tmp_path / "weights"
    config = replace(training, model=replace(training.model, weights_dir=weights_dir))

    fake_data = SimpleNamespace(
        provenance=SimpleNamespace(path=Path("dataset.csv"), sha256="dataset-sha", partition_counts={}),
        split=SimpleNamespace(to_dict=lambda: {}),
    )
    monkeypatch.setattr("sentience_train.campaign.load_tuning_data", lambda *args, **kwargs: fake_data)
    tuner = BatchAutotuner(probe=lambda batch: (0, 100, True))
    missing = DefaultCampaignPlanner(tuning, config, batch_autotuner=tuner).resolve()

    weights_dir.mkdir()
    (weights_dir / "mobilenet_v3_small-047dcff4.pth").write_bytes(b"checkpoint")
    present = DefaultCampaignPlanner(tuning, config, batch_autotuner=tuner).resolve()

    assert missing.plan_hash != present.plan_hash
    assert missing.document["weights"]["MobileNetV3-Small"]["status"] == "missing"
    assert present.document["weights"]["MobileNetV3-Small"]["status"] == "invalid"


def test_fixed_training_batch_bypasses_memory_autotuning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training = load_training_config("configs/default.yaml")
    tuning = load_tuning_config("configs/tuning.yaml")
    fake_data = SimpleNamespace(
        provenance=SimpleNamespace(path=Path("dataset.csv"), sha256="dataset-sha", partition_counts={}),
        split=SimpleNamespace(to_dict=lambda: {}),
    )
    monkeypatch.setattr("sentience_train.campaign.load_tuning_data", lambda *args, **kwargs: fake_data)

    def unexpected_probe(batch: int):
        pytest.fail(f"memory probe must not run for configured batch {batch}")

    plan = DefaultCampaignPlanner(tuning, training, batch_autotuner=BatchAutotuner(probe=unexpected_probe)).resolve()

    assert plan.document["batch_candidates"] == [128]
    assert {choice["mode"] for choice in plan.document["batch_preflights"].values()} == {"configured"}


def test_train_reports_the_terminal_campaign_cause(tmp_path: Path) -> None:
    training = load_training_config("configs/default.yaml")
    campaigns = tmp_path / "artifacts" / "campaigns"
    campaign = campaigns / "exhausted"
    campaign.mkdir(parents=True)
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "tuning-campaign/v2",
                "status": "exhausted",
                "error": "CampaignError: fewer than two architectures passed preflight",
            }
        ),
        encoding="utf-8",
    )
    config = replace(training, output=replace(training.output, campaigns_dir=campaigns))

    with pytest.raises(CampaignExhaustedError, match="fewer than two architectures passed preflight"):
        execute_training(config)
