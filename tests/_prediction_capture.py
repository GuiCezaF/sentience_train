from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionBundle:
    """One evaluation run: ground-truth labels, predicted class indices, and row-wise probabilities."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    source: str


def _synthetic_bundle(num_classes: int = 4, seed: int = 42) -> PredictionBundle:
    rng = np.random.RandomState(seed)
    n_per = 64
    y_true = np.repeat(np.arange(num_classes, dtype=np.int64), n_per)
    scores = rng.dirichlet(np.ones(num_classes), size=y_true.size)
    scores = scores / scores.sum(axis=1, keepdims=True)
    y_pred = scores.argmax(axis=1).astype(np.int64)
    return PredictionBundle(
        y_true=y_true,
        y_pred=y_pred,
        y_score=scores.astype(np.float64),
        source="synthetic",
    )


def collect_predictions_for_metrics_tests() -> PredictionBundle:
    """Load checkpoint and test `ImageFolder` when possible; otherwise return the synthetic bundle."""
    try:
        import torch
        import config as cfg
        from data.transforms import eval_transform
        from model import MiniVGGNet
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
    except ImportError:
        return _synthetic_bundle()

    if not cfg.BEST_MODEL_PATH.is_file():
        return _synthetic_bundle()
    test_root = cfg.DATA_DIR / "test"
    if not test_root.is_dir():
        return _synthetic_bundle()

    try:
        ds = ImageFolder(root=str(test_root), transform=eval_transform)
        if len(ds) == 0:
            return _synthetic_bundle()
        loader = DataLoader(
            ds,
            batch_size=min(cfg.BATCH_SIZE, 128),
            shuffle=False,
            num_workers=0,
        )
        ckpt = torch.load(str(cfg.BEST_MODEL_PATH), map_location=cfg.DEVICE)
        model = MiniVGGNet(cfg.NUM_CLASSES).to(cfg.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        y_list: list[int] = []
        p_list: list[np.ndarray] = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(cfg.DEVICE)
                logits = model(images)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                y_list.extend(labels.numpy().tolist())
                p_list.append(prob)
        y_score = np.vstack(p_list)
        y_true = np.asarray(y_list, dtype=np.int64)
        y_pred = y_score.argmax(axis=1).astype(np.int64)
        return PredictionBundle(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            source="checkpoint",
        )
    except (OSError, RuntimeError, ValueError, KeyError):
        return _synthetic_bundle()
