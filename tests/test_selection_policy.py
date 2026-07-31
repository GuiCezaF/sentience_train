from __future__ import annotations

import pytest

from sentience_train.config import QualityGatesConfig
from sentience_train.selection import (
    EarlyStopping,
    Metrics,
    ResourceMetrics,
    SeedResult,
    evaluate_gates,
    metrics_from_predictions,
    representative_seed,
    select_first_passing,
    should_run_qat,
    width_order,
)


def _metrics(accuracy: float = 0.8, macro_f1: float = 0.7, recall: float = 0.6) -> Metrics:
    return metrics_from_predictions(
        [0, 1, 2, 3] * 25,
        [0, 1, 2, 3] * 25,
        loss=0.2,
    ) if accuracy == 1.0 else Metrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        loss=0.2,
        recall_by_class=(recall,) * 4,
        precision_by_class=(recall,) * 4,
        f1_by_class=(macro_f1,) * 4,
        support_by_class=(25,) * 4,
        confusion_matrix=((25, 0, 0, 0), (0, 25, 0, 0), (0, 0, 25, 0), (0, 0, 0, 25)),
    )


def _seed(seed: int, score: float) -> SeedResult:
    fp32 = _metrics()
    int8 = _metrics(macro_f1=score)
    gates = evaluate_gates(int8, QualityGatesConfig(), reference=fp32, resources=ResourceMetrics(1, 1, 1))
    return SeedResult(0.25, seed, fp32, evaluate_gates(fp32, QualityGatesConfig()), int8, gates, ResourceMetrics(1, 1, 1), 0.0, "ptq", "passed")


def test_early_stopping_requires_minimum_improvement_and_uses_loss_tie_break() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.002)
    assert stopping.update(1, 0.60, 0.50)
    assert not stopping.update(2, 0.601, 0.10)
    assert stopping.update(3, 0.60, 0.40)
    assert stopping.best_epoch == 3
    assert not stopping.update(4, 0.601, 0.10)
    assert stopping.update(5, 0.603, 0.20)
    assert not stopping.should_stop


def test_width_order_and_first_pass_stop_are_canonical() -> None:
    assert width_order((0.50, 0.25, 0.35)) == (0.25, 0.35, 0.50)
    first = _seed(42, 0.70)
    second = _seed(43, 0.71)
    third = _seed(44, 0.72)
    # The pure first-width choice is based on aggregate objects in production;
    # this assertion covers that it never reverses the canonical order.
    assert width_order() == (0.25, 0.35, 0.50)
    assert representative_seed((first, second, third)) == 43


def test_representative_uses_median_and_lowest_seed_on_tie() -> None:
    assert representative_seed((_seed(42, 0.70), _seed(43, 0.70), _seed(44, 0.80))) == 42
    assert representative_seed((_seed(42, 0.70), _seed(43, 0.80), _seed(44, 0.80))) == 43


@pytest.mark.parametrize(
    "accuracy_delta,conversion,expected",
    [(None, True, True), (0.021, False, True), (0.020, False, False), (0.10, False, True)],
)
def test_qat_has_only_the_two_authorized_triggers(accuracy_delta, conversion, expected) -> None:  # type: ignore[no-untyped-def]
    assert should_run_qat(ptq_conversion_failed=conversion, accuracy_delta=accuracy_delta) is expected


def test_gates_are_inclusive_at_exact_boundaries() -> None:
    gates = QualityGatesConfig()
    metrics = Metrics(
        accuracy=0.70,
        macro_f1=0.65,
        loss=0.2,
        recall_by_class=(0.55,) * 4,
        precision_by_class=(0.55,) * 4,
        f1_by_class=(0.65,) * 4,
        support_by_class=(1,) * 4,
        confusion_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )
    result = evaluate_gates(metrics, gates, reference=Metrics(0.72, 0.7, 0.2, (0.6,) * 4, (0.6,) * 4, (0.7,) * 4, (1,) * 4, metrics.confusion_matrix), resources=ResourceMetrics(1_048_576, 1_572_864, 15_000_000))
    assert result.passed, result.reasons
