# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An ONNX-only host contract with strict v2 configuration, tuning campaign,
  training run, release bundle, metrics, and benchmark schemas, plus the
  `onnx-contract/v1` graph contract.
- Deterministic campaign planning and tuning, immutable campaign reuse by
  `plan_hash`, local checkpoint validation, and transactional run and release
  publication.
- The T5 stability recipe with 0.05 label smoothing, five warmup epochs, and
  unit-norm gradient clipping.

### Changed

- The public pipeline now publishes one FP32 ONNX artifact with preprocessing
  embedded in the graph; validation, benchmarking, and publication use the same
  bytes.
- Full-training orchestration now lives in a dedicated training module backed
  by atomic run staging, and exhausted campaigns are execution failures.
- The default optimization batch size is 128, and campaign identity includes
  the configured batch semantics and complete recipe definition.
- On Windows, `torch` and `torchvision` resolve from PyTorch's CUDA 13.0 index
  for native NVIDIA RTX acceleration.
- Acceptance tests that require a real run are excluded from the default pytest
  command.
- Local ML state is organized under `data/`, `models/`, and `artifacts/`, while
  generated reports live under `reports/`.
- Current source text and documentation are standardized in English.

### Removed

- Embedded-runtime formats, schemas, dependencies, and documentation from the
  public release pipeline.
- The legacy entry point, ImageFolder and evaluation pipelines, the public
  `evaluate`, `crossval`, `baseline`, and `compare` workflows, MiniVGG, and the
  distillation contingency.
- Unused compatibility modules, aliases, helpers, synthetic adapters, and the
  unreferenced `tqdm` and `matplotlib` dependencies.

## [1.0.3] - 2026-04-15

### Added

- `build_evaluation_metrics_document` in `src/evaluation/metrics_schema.py`
  (and export in `evaluation.__init__`) to build version-6 metrics JSON from
  labels, predicted classes, and probability scores (`log_loss`, multiclass ROC
  AUC OvR, and subgroup layout aligned with validation).
- `tests/_prediction_capture.py` to collect test-set predictions from
  `MiniVGGNet` when `models/best_model.pth` and `dataset/test` exist, with a
  deterministic synthetic fallback if PyTorch, data, or a checkpoint are
  unavailable.
- `pytest.ini` configuration adding `tests` to `pythonpath` so test helpers
  import cleanly.

### Changed

- `tests/test_evaluation_metrics.py` now uses a session-scoped fixture to build
  metrics at test time, writes `reports/evaluation_metrics.generated.json` and
  `reports/evaluation_run_meta.txt`, and no longer reads
  `tests/fixtures/model_44_evaluation_metrics.json`.
- `tests/conftest.py` now uses only `reports/evaluation_metrics.generated.json`
  and, when present, `evaluation_run_meta.txt` for the Markdown metrics section,
  with no fallback to external JSON at the repository root.
- The `test_wrong_version_is_rejected` assertion is aligned with the English
  validation messages from `validate_evaluation_metrics`.
- User-facing Markdown, module docstrings, comments, and test identifiers in
  evaluation helpers are in English; the session metadata key
  `prediction_source` replaces `fonte_predicoes`.

### Fixed

- The scikit-learn warning that `y_prob` values do not sum to one when calling
  `log_loss` on FP32 softmax output. `build_evaluation_metrics_document` now
  renormalizes probability rows in FP64 before `log_loss` and `roc_auc_score`.

## [1.0.1] - 2026-04-14

### Changed

- Translated the entire Python codebase to English.
- Translated comments, docstrings, and internal notes in all source files.
- Updated user-facing strings, log messages, and console output to English.
- Translated CLI argument help and error messages in `index.py` and pipeline
  scripts.

## [1.0.0] - 2026-04-14

### Added

- `LICENSE` with the MIT open-source license definition for the project.
- The initial `README.md` with project instructions and execution architecture.
- `GITHUB_ABOUT.md` with an optimized project description and GitHub tags.
- The initial Python machine-learning project structure.
- `.gitignore` rules for temporary files, virtual environments, and Python
  compilation artifacts.
- Local dependency configuration through `requirements.txt`.
- A centralized `logs/` directory.
- The main application module at `src/index.py`.
- The central configuration module at `src/config.py`.
- Data-loading and preprocessing modules at `src/data/dataset.py` and
  `src/data/transforms.py`.
- The model definition at `src/model/model.py`.
- Complete processing and training workflows under `src/pipeline/`, including
  baseline creation, comparison, cross-validation, evaluation, export, and
  training.
- A `mise` configuration pinning Python 3.12.13.
- Native NVIDIA RTX acceleration through PyTorch with CUDA 12.1.
- General project dependencies listed in `requirements.txt`.
- `ColorJitter` brightness and contrast augmentation for monochrome input.

### Fixed

- Model collapse on Edge Impulse by removing the duplicate weighting penalty
  between `CrossEntropyLoss` and `WeightedRandomSampler` that caused excessive
  predictions of the minority `disgust` class.
- The `torch.load()` security warning by enabling `weights_only=True` in
  `export.py`.
- Architecture and distribution conflicts by isolating the virtual environment
  from the global Python 3.14 installation.
- Library activation errors by mapping the direct `mise` binaries in
  `.venv\pyvenv.cfg`.
- Training-log verbosity while adding checks for critical execution errors.

### Removed

- The `disgust`, `surprise`, and `fear` classes from the dataset and model
  configuration, reducing `num_classes` from seven to four to focus on stress
  and satisfaction measurement.

[unreleased]: https://github.com/GuiCezaF/sentience_train/compare/v1.0.3...HEAD
[1.0.3]: https://github.com/GuiCezaF/sentience_train/compare/v1.0.1...v1.0.3
[1.0.1]: https://github.com/GuiCezaF/sentience_train/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GuiCezaF/sentience_train/releases/tag/v1.0.0
