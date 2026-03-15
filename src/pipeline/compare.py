import json
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import config as cfg


def run_compare():
    baseline_path = cfg.LOGS_DIR / "baseline_results.json"
    if not baseline_path.exists():
        print("ERRO: execute 'baseline' primeiro para gerar baseline_results.json")
        return
    if not cfg.BEST_MODEL_PATH.exists():
        print("ERRO: execute 'train' e 'evaluate' antes de comparar.")
        return

    with open(str(baseline_path)) as f:
        baselines = json.load(f)

    crossval_path = cfg.LOGS_DIR / "crossval_results.json"
    main_acc = main_f1 = None
    if crossval_path.exists():
        with open(str(crossval_path)) as f:
            cv = json.load(f)
        main_acc = cv["summary"]["accuracy"]["mean"]
        main_f1 = cv["summary"]["f1_macro"]["mean"]

    models = {
        "Logistic Regression": baselines["logistic_regression"],
        "Baseline CNN": baselines["baseline_cnn"],
    }
    if main_acc is not None:
        models["MiniVGGNet (CV media)"] = {"accuracy": main_acc, "f1_macro": main_f1}

    names = list(models.keys())
    accs = [v["accuracy"] for v in models.values()]
    f1s = [v["f1_macro"] for v in models.values()]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1-macro", color="#DD8452")
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Comparacao de Modelos — FER 2023")
    ax.legend()
    fig.tight_layout()
    out_path = str(cfg.LOGS_DIR / "comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    table = [[name, f"{v['accuracy']:.4f}", f"{v['f1_macro']:.4f}"] for name, v in models.items()]
    print("\n=== Comparacao de Modelos ===")
    print(tabulate(table, headers=["Modelo", "Accuracy", "F1-macro"], tablefmt="rounded_outline"))
    print(f"\nGrafico salvo em: {out_path}")
