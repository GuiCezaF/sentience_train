# sentience-train

Deterministic training and immutable export of a four-class FER2013 classifier for the ESP32-S3-CAM. The only public workflows are `train` and `export`; selection creates DS-CNN candidates exclusively.

## Prerequisites and installation

- Python **3.11.15** and uv **0.11.32**, pinned in `mise.toml`;
- `mise` to provision the versions;
- the canonical FER2013 CSV and its local data in `dataset/`;
- a CPU for host validation. CUDA is optional for training.

Installation is locked by the lockfile:

```bash
mise install
uv sync --locked --group dev --group export
```

The `uv.lock` file must be used without local changes. The `dev` group includes pytest, Ruff, mypy, and YAML stubs; `export` includes ONNX, ONNX Runtime, Matplotlib/Agg, and LiteRT/AI Edge.

## Canonical dataset

The file is a UTF-8 CSV with the exact header `emotion,pixels,Usage`. Each record contains 2,304 integer pixels between 0 and 255, and `Usage` is `Training`, `PublicTest`, or `PrivateTest`. The accepted SHA-256 is:

```text
3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b
```

The stable mapping from emotions to model outputs is `0 -> angry`, `3 -> happy`, `6 -> neutral`, and `4 -> sad`. Unselected data remains in the provenance, but only the four canonical classes are used for training. `Training` is used for optimization; `PublicTest` is used for validation, early stopping, calibration, and selection; `PrivateTest` may be read only once by a complete export of an explicitly supplied canonical run.

## Configuration and selection

`configs/default.yaml` is the canonical configuration. The schema is strict: removed keys such as `model.name`, `training.seed`, and the entire `distillation` section are rejected. The CLI only allows overriding `training.epochs` with a value from 1 through 100 and `training.device` with `auto`, `cpu`, or `cuda`.

Selection evaluates widths `0.25`, `0.35`, and `0.50`, in that order, with seeds `42`, `43`, and `44`. It stops at the first width that passes every gate. The representative is the median by INT8 macro-F1, with the lower seed as the tie-breaker. Each training run uses macro-F1 for early stopping, loss as the tie-breaker, and restores the best checkpoint. The loss is CrossEntropy with fixed class weights based on inverse square-root frequency and normalized by their mean.

PTQ is attempted first. QAT is triggered only by PTQ conversion failure or an accuracy loss above the limit; there is no teacher, MiniVGG, or distillation. If every DS-CNN candidate fails, the command creates an immutable `candidate-comparison.json`/`.md` diagnostic, creates no release, and does not access `PrivateTest`.

## Training

```bash
sentience-train train --config configs/default.yaml
sentience-train train --config configs/default.yaml --epochs 10 --device cpu
```

The result is an immutable run in `runs/<id>/`, with a `manifest.json` that follows the `frozen-run/v1` schema, the resolved configuration, hashes, a `PublicTest` offset index, checkpoints, JSONL history, metrics, and candidate comparisons. Synthetic fixtures are useful for tests, but never serve as release evidence.

## Export

```bash
sentience-train export --run-id <id>
sentience-train export --run-id <id> --format onnx --runs-dir ./runs
sentience-train export --run-id <id> --format tflite --runs-dir ./runs
```

`all` is the default and publishes to `exports/<id>/all/`. `onnx` and `tflite` are verifiable handoffs derived from the same frozen run. No format modifies the run; manifests and payloads are hash-checked, and publication and reuse are atomic.

### ONNX FP32 handoff

- `float32` dtype;
- NCHW `[N,1,48,48]` layout;
- dynamic batch;
- `pixel / 127.5 - 1` normalization;
- FP32 `[N,4]` output;
- host parity validation against PyTorch using only the `PublicTest` index.

### TFLite INT8 handoff

- NHWC `[1,48,48,1]` layout;
- fixed batch;
- scale and zero point recorded in `input_contract.json`/`output_contract.json`;
- `labels.txt`, an operator allowlist, and resource metadata shipped with the FlatBuffer;
- `emotion_model_data.h` and `emotion_model_data.cc` generated from the same bytecode, with the same labels and size.

### ESP32-S3-CAM N8R8

Host gates enforce a FlatBuffer size of at most 1 MiB, an arena estimate with margin of at most 1.5 MiB, and at most 15 million MACs. Actual arena measurement and a p95 latency of 200 ms on the ESP32 remain `pending_external`; a host estimate is not presented as approval.

## Metrics and reports

`sentience_train.metrics` is the only metrics module used by training, selection, and release. Each document includes accuracy, macro-precision, macro-recall, macro-F1, loss, per-class precision/recall/F1, support, and the raw matrix in `angry,happy,neutral,sad` order. JSON matrices use `labels` and named `matrices`; CSV uses the long format `model,actual,predicted,count`; PNG is rendered with Matplotlib/Agg in a byte-deterministic manner.

The release keeps `PublicTest` and `PrivateTest` evidence separate. `PrivateTest` predictions are not saved: only metrics, support, the matrix, and an access marker are persisted. Final statuses are accurately reported as `accepted` or `failed`, with gates, hashes, and diagnostics.

## Real acceptance and integrity

Acceptance is never part of the default suite (`pytest` uses `-m 'not acceptance'`). To run it, explicitly provide `SENTIENCE_RELEASE_RUN_ID` and, optionally, `SENTIENCE_RELEASE_RUNS_ROOT`/`SENTIENCE_RELEASE_EXPORTS_ROOT`:

```bash
SENTIENCE_RELEASE_RUN_ID=<id> pytest -m acceptance tests/test_release_acceptance.py
```

Without the real run and inputs, `pytest -m acceptance` fails; there is no skip or synthetic fallback. The `all` export authorizes `PrivateTest` only after validating the `frozen-run/v1` schema, the canonical CSV SHA-256, the PublicTest index, the checkpoint, TFLite, operators, and host gates. Earlier frozen runs, including manifests that record a teacher, remain consumable; the removal affects only the creation of new runs.
