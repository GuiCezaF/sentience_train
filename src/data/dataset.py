import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
import numpy as np

from data.transforms import train_transform, eval_transform


def _compute_class_weights(dataset):
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[targets]
    return torch.tensor(sample_weights, dtype=torch.float32), torch.tensor(class_weights, dtype=torch.float32)


def get_dataloaders(cfg):
    train_dataset = ImageFolder(root=str(cfg.DATA_DIR / "train"), transform=train_transform)
    test_dataset = ImageFolder(root=str(cfg.DATA_DIR / "test"), transform=eval_transform)

    sample_weights, class_weights = _compute_class_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader, class_weights


def get_kfold_datasets(cfg):
    full_train = ImageFolder(root=str(cfg.DATA_DIR / "train"), transform=None)
    indices = np.arange(len(full_train))
    targets = np.array(full_train.targets)

    kfold = KFold(n_splits=cfg.K_FOLDS, shuffle=True, random_state=42)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices, targets)):
        train_subset = Subset(
            ImageFolder(root=str(cfg.DATA_DIR / "train"), transform=train_transform),
            train_idx,
        )
        val_subset = Subset(
            ImageFolder(root=str(cfg.DATA_DIR / "train"), transform=eval_transform),
            val_idx,
        )

        fold_targets = targets[train_idx]
        class_counts = np.bincount(fold_targets, minlength=cfg.NUM_CLASSES)
        class_weights = torch.tensor(1.0 / (class_counts.astype(float) + 1e-6), dtype=torch.float32)
        sample_weights = class_weights[fold_targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        folds.append((fold_idx, train_loader, val_loader, class_weights))

    return folds
