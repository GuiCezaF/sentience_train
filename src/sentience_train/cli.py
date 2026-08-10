"""Thin command-line adapter for the four public workflow entry points."""

from __future__ import annotations

import argparse
import sys

from .campaign import CampaignError, execute_tune
from .config import ConfigError, canonical_json, load_training_config, load_tuning_config
from .data import DataValidationError
from .export import export_frozen_run
from .training import CampaignExhaustedError, TrainingPipelineError, execute_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentience-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    tune = subparsers.add_parser("tune")
    tune.add_argument("--config", required=True)
    tune.add_argument("--plan", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "tune":
            tuning = load_tuning_config(args.config)
            training = load_training_config(tuning.training_config)
            plan, manifest = execute_tune(tuning, training, plan_only=args.plan)
            if args.plan:
                print(canonical_json(plan.document))
                return 0
            assert manifest is not None
            if manifest.document.get("status") == "completed" and manifest.document.get("shortlist"):
                print(f"completed tuning campaign {manifest.directory.name}: {manifest.directory}")
                return 0
            print(f"tuning campaign exhausted: {manifest.directory}", file=sys.stderr)
            return 1
        if args.command == "train":
            config = load_training_config(args.config)
            directory = execute_training(config)
            print(f"completed training run {directory.name}: {directory}")
            return 0
        result = export_frozen_run(args.run_id)
        print(f"{result.status} export {result.run_id}: {result.directory}")
        return 0 if result.status == "accepted" else 1
    except CampaignExhaustedError as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 1
    except (ConfigError, DataValidationError, CampaignError) as exc:
        print(f"input/configuration error: {exc}", file=sys.stderr)
        return 2
    except TrainingPipelineError as exc:
        print(f"execution error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"execution error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
