# FER2013 Four-Emotion Edge Model Refactoring Specification

**Status:** Accepted — approved for implementation on 2026-07-31.

This specification refactors the existing PyTorch project into a reproducible training
and export pipeline for a four-emotion classifier. It does not include camera
capture, face detection, an API server, or ESP-IDF firmware.

## Repository findings

- The existing project already separates data, models, pipelines, and metrics,
  but relies on global configuration and import-path manipulation.
- The loader expects pre-filtered `ImageFolder` directories. It neither reads
  the canonical CSV nor validates or remaps FER2013 labels.
- Training currently calls the test loader “validation” and selects checkpoints
  against it, leaking final-test information.
- The current main network has approximately 2,650,212 parameters, 10.1 MiB of
  raw FP32 weights, 2.53 MiB of raw INT8 weights, and 88 million MACs per image.
- Class weights are calculated but ignored, while weighted sampling changes the
  training distribution.
- Global reproducibility is absent. Cross-validation is not stratified and
  auxiliary pipelines use inconsistent evaluation semantics.
- Export produces only FP32 ONNX and does not verify full integer I/O, TFLite
  Micro operators, tensor-arena use, or the embedded memory budget.
- Existing metric tests can silently substitute synthetic predictions when the
  real dataset or checkpoint is unavailable.

## Confirmed assumptions and contracts

### Data

- Canonical source: `dataset/fer2013.csv`.
- Observed SHA-256:
  `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`.
- The file contains 35,887 valid rows with 2,304 pixels each.
- `Training` is the training partition, `PublicTest` is validation, and
  `PrivateTest` is the final reserved test partition.
- After filtering, the partitions contain 21,005 training, 2,622 validation,
  and 2,590 test examples.
- Output order is fixed as `angry`, `happy`, `neutral`, `sad`; raw FER2013 IDs
  map as `0→0`, `3→1`, `6→2`, and `4→3`. IDs 1, 2, and 5 are ignored.
- Dataset schema, pixel count, emotion IDs, partition names, class counts, and
  hash are recorded in each run manifest. The CSV remains outside Git.

### Responsibility and inputs

- The classifier accepts an already-located 48x48 grayscale face crop.
- Camera capture, face detection, alignment, and full-frame preprocessing are
  responsibilities of consuming projects.
- PyTorch training uses NCHW FP32 tensors normalized to `[-1, 1]`.
- ONNX accepts FP32 `[N, 1, 48, 48]` with dynamic batch size.
- TFLite accepts INT8 NHWC `[1, 48, 48, 1]` and publishes scale and zero point.

### Target and deployment budgets

- Target board assumption: ESP32-S3-CAM N8R8 or equivalent, with 8 MB flash
  and 8 MB PSRAM.
- Fully INT8 `.tflite` file: at most 1 MiB.
- Estimated TFLite Micro tensor arena, including 20% safety headroom: at most
  1.5 MiB PSRAM.
- Compute: at most 15 million MACs for one face crop.
- Embedded integration target: inference p95 at or below 200 ms at 240 MHz with
  ESP-NN. This is measured later on hardware, never inferred from a host run.

### Predictive quality

The frozen TFLite INT8 release candidate must pass on filtered `PrivateTest`:

- accuracy at least 0.70;
- macro-F1 at least 0.65;
- recall for every target emotion at least 0.55;
- accuracy loss from PyTorch FP32 no more than two percentage points.

The ONNX FP32 service model must remain equivalent to the PyTorch checkpoint.
Equivalent thresholds are used only as selection proxies on `PublicTest`.

### Test-access protocol

- Candidate development and selection use only `Training` and `PublicTest`.
- Architecture, configuration, and checkpoint are frozen before release.
- Only `export --format all` evaluates the export bundle on `PrivateTest`.
- A failed release is reported as failed and cannot be replaced based on
  observed test results.
- Test access and final results are persisted under an immutable run ID.

## Target project structure

```text
.
├── configs/
│   └── default.yaml
├── docs/
│   ├── adr/
│   └── refactor-plan.md
├── src/
│   └── sentience_train/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       │   ├── fer2013.py
│       │   └── transforms.py
│       ├── models/
│       │   ├── budgets.py
│       │   ├── ds_cnn.py
│       │   └── teacher.py
│       ├── training/
│       │   ├── distillation.py
│       │   ├── engine.py
│       │   ├── reproducibility.py
│       │   └── selection.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── reporting.py
│       └── exporting/
│           ├── bundle.py
│           ├── onnx.py
│           ├── parity.py
│           └── tflite.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── acceptance/
├── tools/
│   └── tflm_probe/
├── CONTEXT.md
├── pyproject.toml
└── uv.lock
```

## Configuration and environment

- Pin Python 3.11 in `mise.toml`.
- Replace loose runtime dependencies with `pyproject.toml` and committed
  `uv.lock`, separating training, export, and development groups.
- Load `configs/default.yaml` into immutable validated dataclasses.
- Reject unknown keys, invalid classes, invalid paths, and contradictory gates.
- Support `--config` plus a small set of explicit CLI overrides.
- Avoid filesystem creation, CUDA probing, or other side effects on import.
- Copy the fully resolved configuration into every run directory.

Initial configurable training defaults are batch size 128, AdamW with learning
rate `1e-3` and weight decay `1e-4`, a cosine schedule, and at most 100 epochs.
These are tunable defaults rather than architectural contracts.

## Data and training behavior

- Parse CSV rows lazily and validate all pixel and partition invariants.
- Build deterministic datasets directly from official partitions.
- Training augmentation: horizontal flip 0.5, rotation ±10 degrees,
  translation 8%, scale 0.95–1.05, and brightness/contrast ±15%.
- Validation, test, and quantization calibration are deterministic and do not
  use augmentation.
- Use normal shuffling and inverse-square-root class weights normalized to mean
  one in cross-entropy. Do not use weighted sampling or oversampling.
- Seed Python, NumPy, PyTorch, CUDA, DataLoader generators, and workers. Enable
  deterministic algorithms and fail on unsupported nondeterministic behavior.
- Early-stop on validation macro-F1 with patience 12 and minimum improvement
  0.002; use validation loss as the tie-breaker and restore the chosen state.
- Screen a candidate with seed 42. If it passes validation, repeat seeds 43 and
  44, apply proxy gates to their mean, and select the median macro-F1 checkpoint.

## Model selection

- Use a MobileNetV1-inspired depthwise-separable CNN with a standard stem,
  depthwise/pointwise blocks, foldable BatchNorm, ReLU6, strided downsampling,
  fixed global average pooling, and a four-output linear head.
- Try width multipliers 0.25, 0.35, and 0.50 from smallest to largest.
- Reject candidates statically exceeding a deployment constraint before train.
- Stop at the smallest candidate passing validation; compare passing candidates
  by TFLite size, arena estimate, MACs, then validation macro-F1.
- If all widths fail, train the legacy MiniVGG as a non-deployed teacher and
  retry distilled students from smallest to largest.
- If distillation also fails, report the failed gates and stop without reading
  `PrivateTest` or weakening acceptance criteria.

## Quantization and export

- Before full training, run direct PyTorch-to-LiteRT preflight on every width.
- Fail fast on conversion, full-INT8, operator, size, arena, or MAC problems.
  Never switch converter chains silently.
- Use PTQ first with 512 deterministic, class-stratified, unaugmented examples
  from `Training`.
- Use QAT only when FP32 passes validation but INT8 loses over two percentage
  points or fails another validation threshold.
- Export ONNX directly from the same checkpoint in FP32 with dynamic batch.
- Check PyTorch/ONNX numerical parity and PyTorch/TFLite predictive parity.
- Inspect TFLite input/output types and operator allowlist.
- Run a pinned host `RecordingMicroInterpreter` probe and gate the total plus
  20% headroom; report it as an estimate, not target measurement.

The TFLite handoff contains the `.tflite`, aligned const C++ array/header,
labels, input contract, operator list, and hashes. The ONNX handoff contains the
FP32 model and its input/output contract.

## CLI and immutable runs

Only two commands remain:

```bash
sentience-train train --config configs/default.yaml
sentience-train export --run-id <run-id> --format all
```

- `train` never reads `PrivateTest`; it writes checkpoint, resolved config,
  history, seed variance, and validation metrics to a new `runs/<run-id>/`.
- `export --format onnx|tflite` creates development artifacts and validates
  them only on `PublicTest`.
- `export --format all` is the sole release transaction and the only operation
  allowed to mark a run `accepted`.
- Existing runs and artifacts are never silently overwritten.

## Reports

Each completed run or release contains:

- `manifest.json` with configuration, provenance, versions, device, Git commit,
  dataset and artifact hashes;
- validation and final test metric JSON documents;
- Markdown gate report;
- confusion matrix in CSV and PNG;
- aggregate and per-emotion precision, recall, F1, and support;
- PyTorch, ONNX, and TFLite comparisons;
- parameter count, file sizes, MACs, quantization loss, operator list, and
  estimated TFLM arena breakdown;
- explicit pass/fail reasons and pending on-device measurements.

Release reporting never substitutes synthetic predictions.

## Tests

- Unit: configuration, CSV parsing, filtering/remapping, validation failures,
  transforms, weights, metrics, model shapes, and static budgets.
- Integration: tiny synthetic CSV, one CPU epoch, immutable run creation,
  ONNX/TFLite conversion, INT8 contracts, parity, and operator inspection.
- Acceptance: real FER2013, trained checkpoint, quality gates, size, MACs, and
  host arena estimate. Missing data or checkpoints fail explicitly.
- Hardware: ESP32-S3 arena and latency measurements remain in the embedded
  integration project.

## Planned diff

Add the package, YAML configuration, locked environment, tiered tests, TFLM
probe, run manifests, export bundle generation, glossary, ADRs, and this plan.

Migrate reusable metric logic from `src/evaluation/metrics_schema.py` into the
new package. Update `README.md`, `CHANGELOG.md`, `mise.toml`, `.gitignore`, and
test configuration.

After the new unit and integration suites pass, remove the old global config,
entrypoint, ImageFolder data path, MiniVGG deployment path, BaselineCNN, legacy
pipeline commands, old tests with release fallback behavior,
`requirements.txt`, and the checked-in stale confusion-matrix image. Retain the
MiniVGG definition only as an internal teacher if the fallback is activated.

## Implementation sequence

1. Establish the package, locked environment, config schema, run layout, and
   unit-test foundation.
2. Implement and test canonical CSV loading, remapping, transforms, weights,
   deterministic seeding, and manifests.
3. Implement the bounded DS-CNN family, parameter/MAC accounting, exporters,
   operator inspection, and host arena probe.
4. Run conversion preflight for all widths; stop and diagnose any failure.
5. Implement the training engine, early stopping, three-seed selection, PTQ,
   conditional QAT, and optional distillation.
6. Run integration tests and then remove the superseded pipeline.
7. Train against real `Training`, select exclusively with `PublicTest`, and
   produce the immutable release candidate.
8. Run `export --format all` once, evaluate `PrivateTest`, generate the handoff
   bundles and reports, and mark the run accepted only if every gate passes.
9. Update README and changelog with exact commands, outputs, assumptions,
   limitations, and on-device checks still pending.

## Completion conditions

Implementation is complete only when the two commands run end to end, all
non-hardware tests pass, the release bundle satisfies predictive, file-size,
arena-estimate, and MAC gates, and documentation matches the produced artifacts.
The 200 ms latency and real ESP32-S3 arena remain explicitly pending until the
embedded project measures them. A failed quality or conversion gate is a
reported failure, not grounds for an unsupported success claim.
