import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from tqdm import tqdm
from tabulate import tabulate

import config as cfg
from data import get_dataloaders
from model import MiniVGGNet


def evaluate():
    _, test_loader, _ = get_dataloaders(cfg)

    checkpoint = torch.load(str(cfg.BEST_MODEL_PATH), map_location=cfg.DEVICE)
    model = MiniVGGNet(cfg.NUM_CLASSES).to(cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Avaliando", ncols=90):
            images = images.to(cfg.DEVICE)
            logits = model(images)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    summary = [
        ["Accuracy", f"{acc:.4f}"],
        ["Precision (macro)", f"{prec_macro:.4f}"],
        ["Recall (macro)", f"{rec_macro:.4f}"],
        ["F1 (macro)", f"{f1_macro:.4f}"],
        ["F1 (weighted)", f"{f1_weighted:.4f}"],
    ]
    print("\n=== Metricas Gerais ===")
    print(tabulate(summary, headers=["Metrica", "Valor"], tablefmt="rounded_outline"))

    print("\n=== Relatorio por Classe ===")
    print(classification_report(all_labels, all_preds, target_names=cfg.CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusao")
    fig.tight_layout()
    out_path = str(cfg.LOGS_DIR / "confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nMatriz de confusao salva em: {out_path}")

    return {"accuracy": acc, "f1_macro": f1_macro, "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_weighted": f1_weighted}
