# Changelog

## Unreleased

### Removed
- Removed the legacy entrypoint, ImageFolder/pipeline/evaluation modules, `requirements.txt`, stale confusion-matrix PNG, and fallback metric/capture tests.
- Removed public `evaluate`, `crossval`, `baseline`, and `compare` workflows, `export_development`, `model.name`, `training.seed`, `CompletedRun`, MiniVGG, and the distillation contingency.

### Changed
- Windows resolves `torch` and `torchvision` from PyTorch's CUDA 13.0 index,
  enabling NVIDIA RTX acceleration while preserving deterministic cuBLAS
  workspace configuration during training.
- `train` now creates only DS-CNN frozen runs; `export --run-id ID --format all|onnx|tflite` is the only export interface and `all` remains the default.
- Configuration is strict and YAML/CSV based; ImageFolder and old imports are no longer supported. Existing `frozen-run/v1` artifacts, including manifests with historical teacher metadata, remain readable.
- Metrics and report generation are centralized, with macro precision/recall, long confusion CSV, named JSON matrices, deterministic PNGs, host ESP32-S3 gates, and explicit pending hardware evidence.
- Acceptance is excluded from the default pytest command and fails when `SENTIENCE_RELEASE_RUN_ID` or real inputs are absent.

- Add immutable development ONNX and TFLite bundles for frozen runs, with
  PublicTest-only parity validation, atomic publication, and hash-checked reuse.
- Add the complete `all` release bundle with versioned manifests, deterministic
  PublicTest/PrivateTest evidence, resource gates, and a permanent access ledger.
- Add Matplotlib/Agg to the export dependency group for deterministic confusion
  matrix rendering.
- Add ONNX and ONNX Runtime to the `export` dependency group.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-04-15

### Added
- `build_evaluation_metrics_document` in `src/evaluation/metrics_schema.py` (and export in `evaluation.__init__`) to build version-6 metrics JSON from labels, predicted classes, and probability scores (`log_loss`, multiclass ROC AUC OvR, subgroup layout aligned with validation).
- `tests/_prediction_capture.py` to collect test-set predictions from `MiniVGGNet` when `models/best_model.pth` and `dataset/test` exist, with a deterministic synthetic fallback if PyTorch, data, or checkpoint are unavailable.
- `pytest.ini`: `tests` on `pythonpath` so test helpers import cleanly.

### Changed
- `tests/test_evaluation_metrics.py`: session-scoped fixture builds metrics at test time, writes `reports/evaluation_metrics.generated.json` and `reports/evaluation_run_meta.txt`, and no longer reads `tests/fixtures/model_44_evaluation_metrics.json`.
- `tests/conftest.py`: Markdown report uses only `reports/evaluation_metrics.generated.json` (plus `evaluation_run_meta.txt` when present) for the metrics section; no fallback to external JSON at the repository root.
- `test_wrong_version_is_rejected` assertion aligned with English validation messages from `validate_evaluation_metrics`.
- Tests and evaluation helpers: user-facing Markdown, module docstrings, comments, and test identifiers in English (`tests/conftest.py`, `tests/test_evaluation_metrics.py`, `tests/_prediction_capture.py`, `src/evaluation/metrics_schema.py`); session metadata key `prediction_source` replaces `fonte_predicoes`.

### Fixed
- Sklearn `UserWarning` (“y_prob values do not sum to one”) when calling `log_loss` on float32 softmax outputs: `build_evaluation_metrics_document` now renormalizes probability rows to sum to 1 in float64 before `log_loss` / `roc_auc_score`.

## [1.0.1] - 2026-04-14

- Translate entire Python codebase to English
- Translated comments, docstrings, and internal notes in all src/ files.
- Updated user-facing strings, log messages, and console outputs to English.
- Translated CLI argument help and error messages in index.py and pipeline scripts.


## [1.0.0] - 2026-04-14

### Added
- `LICENSE` file with the MIT open-source license definition for the project.
- Creation and structure of `README.md` with initial instructions and project execution architecture.
- Creation of `GITHUB_ABOUT.md` descriptive file with optimized presentation and GitHub tags.
- Initial project structure for the Python Machine Learning module.
- `.gitignore` file configured to ignore temporary files, virtual environments (like `.venv`), and Python compilations.
- Local dependency configuration with `requirements.txt`.
- Centralized logs directory (`logs/`).
- Main application module (`src/index.py`).
- Central configuration file (`src/config.py`).
- Data manipulation module (`src/data/`):
  - `dataset.py`: routines and abstractions for data loading.
  - `transforms.py`: functions for pre-processing and transformations.
- AI models definition module (`src/model/`):
  - `model.py`: ML model structure and base predictions.
- Complete processing and training flows and pipelines module (`src/pipeline/`):
  - `baseline.py`: creation and testing of comparative base models.
  - `compare.py`: comparison reports and metrics.
  - `crossval.py`: cross-validation strategies to ensure reliability.
  - `evaluate.py`: dedicated evaluation flow on test data.
  - `export.py`: generation of exports (artifacts) for ready models.
  - `train.py`: main training execution flow.
- `mise` manager configuration to pin Python version to `3.12.13` for the project.
- Installation and activation of native GPU support (NVIDIA RTX) via PyTorch with CUDA 12.1.
- Installation of general project dependencies listed in `requirements.txt`.
- `ColorJitter` data augmentation filter for brightness and contrast in transformations to increase robustness for native black and white format.

### Fixed
- Solution for model collapse on Edge Impulse: removed duplicate weighting penalty in `CrossEntropyLoss` that interacted with `WeightedRandomSampler`, which caused the network to massively predict the minority (`disgust` class).
- Security warning fix in `export.py` by enabling `weights_only=True` in `torch.load()`.
- Architecture and distribution conflict resolution by isolating the virtual environment from the global Python 3.14.
- Library activation error fix in `.venv\pyvenv.cfg` mapping direct binaries installed by `mise`.
- Log pipeline optimization in the training file, reducing verbosity (tlog for each epoch) and including global vital error checks.

### Removed
- Removed `disgust`, `surprise`, and `fear` classes from the dataset and model settings (`num_classes` adjusted from 7 to 4) to focus on stress and satisfaction measurement.
