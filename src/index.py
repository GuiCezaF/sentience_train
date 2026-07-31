"""Compatibility launcher for the public ``sentience-train`` CLI."""

from sentience_train.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
