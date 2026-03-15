import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from tabulate import tabulate

import config as cfg
from data.transforms import eval_transform
from model import BaselineCNN


def _load_flat_data(split):
    dataset = ImageFolder(root=str(cfg.DATA_DIR / split), transform=eval_transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    X, y = [], []
    for imgs, labels in tqdm(loader, desc=f"  Carregando {split}", ncols=90, leave=False):
        X.append(imgs.view(imgs.size(0), -1).numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)


def _train_baseline_cnn(train_loader, val_loader):
    model = BaselineCNN(cfg.NUM_CLASSES).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(cfg.DEVICE.type)
    best_acc = 0.0
    best_state = None

    for epoch in tqdm(range(30), desc="  BaselineCNN", ncols=90, leave=False):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=cfg.DEVICE.type):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds.extend(model(imgs.to(cfg.DEVICE)).argmax(1).cpu().numpy())
                true.extend(labels.numpy())
        acc = accuracy_score(true, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc


def run_baseline():
    print("Carregando dados para baseline...")
    X_train, y_train = _load_flat_data("train")
    X_test, y_test = _load_flat_data("test")

    print("\nTreinando Logistic Regression...")
    lr_model = LogisticRegression(max_iter=500, n_jobs=-1, C=0.1)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro", zero_division=0)
    print(f"  LR — Accuracy: {lr_acc:.4f} | F1-macro: {lr_f1:.4f}")

    train_dataset = ImageFolder(root=str(cfg.DATA_DIR / "train"), transform=eval_transform)
    test_dataset = ImageFolder(root=str(cfg.DATA_DIR / "test"), transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\nTreinando BaselineCNN...")
    cnn_model, _ = _train_baseline_cnn(train_loader, test_loader)
    cnn_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            all_preds.extend(cnn_model(imgs.to(cfg.DEVICE)).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    cnn_acc = accuracy_score(all_labels, all_preds)
    cnn_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"  BaselineCNN — Accuracy: {cnn_acc:.4f} | F1-macro: {cnn_f1:.4f}")

    results = {
        "logistic_regression": {"accuracy": lr_acc, "f1_macro": lr_f1},
        "baseline_cnn": {"accuracy": cnn_acc, "f1_macro": cnn_f1},
    }
    out_path = str(cfg.LOGS_DIR / "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    table = [
        ["Logistic Regression", f"{lr_acc:.4f}", f"{lr_f1:.4f}"],
        ["Baseline CNN", f"{cnn_acc:.4f}", f"{cnn_f1:.4f}"],
    ]
    print("\n=== Resultados Baseline ===")
    print(tabulate(table, headers=["Modelo", "Accuracy", "F1-macro"], tablefmt="rounded_outline"))
    print(f"\nResultados salvos em: {out_path}")
