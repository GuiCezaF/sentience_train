"""Public command line interface for the deterministic training foundation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .data import DataValidationError
from .engine import run_training
from .export import (
    ExportCollisionError,
    ExportInputError,
    ExportValidationError,
    export_frozen_run,
)
from .selection import SelectionFailed


def _epochs(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("--epochs must be between 1 and 100")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentience-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--epochs", type=_epochs)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"))
    export = subparsers.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--format", default="all", choices=("all", "onnx", "tflite"))
    export.add_argument("--runs-dir", default="./runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        try:
            result = export_frozen_run(
                args.run_id,
                args.format,
                runs_root=Path(args.runs_dir),
                exports_root=Path("./exports"),
            )
        except ExportInputError as exc:
            print(f"input/run error: {exc}", file=sys.stderr)
            return 2
        except (ExportCollisionError, ExportValidationError) as exc:
            print(f"export error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"export error: {exc}", file=sys.stderr)
            return 1
        action = "reused" if result.reused else "created"
        print(f"{action} {result.status} {result.format} export for {result.run_id}: {result.directory}")
        return 0 if result.status in {"development", "accepted"} else 1

    overrides = {
        f"training.{key}": value
        for key, value in (("epochs", args.epochs), ("device", args.device))
        if value is not None
    }
    try:
        config = load_config(args.config, overrides)
        completed = run_training(config)
    except (ConfigError, DataValidationError) as exc:
        print(f"configuration/data error: {exc}", file=sys.stderr)
        return 2
    except SelectionFailed as exc:
        print(f"selection failed; immutable diagnosis: {exc.diagnosis}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"execution error: {exc}", file=sys.stderr)
        return 1
    print(f"completed run {completed.run_id}: {completed.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
