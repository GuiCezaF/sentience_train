import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Sentience — Emotion Recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Trains the main MiniVGGNet")
    train_p.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")

    subparsers.add_parser("evaluate", help="Evaluates the best model on the test set")
    subparsers.add_parser("crossval", help="Runs 5-fold cross-validation")
    subparsers.add_parser("baseline", help="Trains and evaluates baseline models")
    subparsers.add_parser("compare", help="Generates visual comparison between models")
    subparsers.add_parser("export", help="Exports the best model to ONNX")

    if len(sys.argv) == 1:
        print("Error: No command provided. Please use one of the commands listed below.\n", file=sys.stderr)
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
            raise ValueError(f"Command {args.command} is not valid")

if __name__ == "__main__":
    main()