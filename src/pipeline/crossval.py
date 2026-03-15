import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import config as cfg
from data import get_kfold_datasets
from model import MiniVGGNet


def _train_fold(model, train_loader, val_loader, class_weights, fold_idx):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = GradScaler(cfg.DEVICE.type)

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    bar = tqdm(range(1, cfg.EPOCHS + 1), ncols=90, desc=f"  Fold {fold_idx + 1}", leave=False)
    for epoch in bar:
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=cfg.DEVICE.type):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(cfg.DEVICE)
                preds = model(imgs).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        bar.set_postfix(val_acc=f"{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def _eval_fold(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(cfg.DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def run_kfold():
    folds = get_kfold_datasets(cfg)
    results = []

    fold_bar = tqdm(folds, ncols=90, desc="Cross-Validation")
    for fold_idx, train_loader, val_loader, class_weights in fold_bar:
        model = MiniVGGNet(cfg.NUM_CLASSES).to(cfg.DEVICE)
        model = _train_fold(model, train_loader, val_loader, class_weights, fold_idx)
        preds, labels = _eval_fold(model, val_loader)

        results.append({
            "fold": fold_idx + 1,
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        })
        fold_bar.set_postfix(acc=f"{results[-1]['accuracy']:.4f}")

    keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in results]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out_path = str(cfg.LOGS_DIR / "crossval_results.json")
    with open(out_path, "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)

    table = [[k, f"{v['mean']:.4f}", f"{v['std']:.4f}"] for k, v in summary.items()]
    print("\n=== Resultado Cross-Validation (5-Fold) ===")
    print(tabulate(table, headers=["Metrica", "Media", "Desvio Padrao"], tablefmt="rounded_outline"))
    print(f"\nResultados salvos em: {out_path}")
