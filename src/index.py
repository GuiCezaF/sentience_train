import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Sentience — Reconhecimento de Emocoes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Treina o MiniVGGNet principal")
    train_p.add_argument("--epochs", type=int, default=None, help="Numero de epochs (sobreescreve config)")

    subparsers.add_parser("evaluate", help="Avalia o melhor modelo no conjunto de teste")
    subparsers.add_parser("crossval", help="Executa 5-fold cross-validation")
    subparsers.add_parser("baseline", help="Treina e avalia modelos baseline")
    subparsers.add_parser("compare", help="Gera comparacao visual entre modelos")
    subparsers.add_parser("export", help="Exporta o melhor modelo para ONNX")

    if len(sys.argv) == 1:
        print("Erro: Nenhum comando foi fornecido. Por favor, utilize um dos comandos listados abaixo.\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    match args.command:
        case "train":
            from pipeline.train import train
            train(epochs_override=args.epochs)
        case "evaluate":
            from pipeline.evaluate import evaluate
            evaluate()
        case "crossval":
            from pipeline.crossval import run_kfold
            run_kfold()
        case "baseline":
            from pipeline.baseline import run_baseline
            run_baseline()
        case "compare":
            from pipeline.compare import run_compare
            run_compare()
        case "export":
            from pipeline.export import export
            export()
        case _:
            raise ValueError(f"Comando {args.command} nao e valido")

if __name__ == "__main__":
    main()