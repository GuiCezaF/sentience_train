# Local artifacts

This directory contains all state produced by the pipeline:

- `campaigns/`: tuning plans, evidence, histories, and checkpoints
- `runs/`: immutable full-training results
- `exports/`: ONNX bundles and the publication ledger
- `archive/campaigns/`: older campaigns preserved locally

Git ignores this content because it can be large, machine-specific, or contain
dataset metadata. Do not edit published manifests manually. Keep external
backups of results that must be preserved.
