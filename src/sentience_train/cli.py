"""Public command line interface for the deterministic training foundation."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .data import DataValidationError
from .engine import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentience-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = {f"training.{key}": value for key, value in (("epochs", args.epochs), ("seed", args.seed), ("device", args.device)) if value is not None}
    try:
        config = load_config(args.config, overrides)
        completed = run_training(config)
    except (ConfigError, DataValidationError) as exc:
        print(f"configuration/data error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"execution error: {exc}", file=sys.stderr)
        return 1
    print(f"completed run {completed.run_id}: {completed.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
