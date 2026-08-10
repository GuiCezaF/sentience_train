# ONNX-only host contract

The public classifier has one immutable artifact. It accepts `uint8` NHWC
images with shape `[N, 48, 48, 1]` through `images` and returns FP32
probabilities with shape `[N, 4]` through `probabilities`. The graph owns the
cast, `float32(pixel) / 127.5 - 1.0` normalization, layout conversion, network,
and softmax.

The fixed model set is DS-CNN, MobileNetV3-Small, and ResNet-18. ImageNet
weights are read only from `models/pretrained/` after filename, size, and
SHA-256 validation. No pipeline command downloads weights.

## Workflow

```text
tune --plan -> tune -> train -> export
```

`tune --plan` validates the CSV, deterministic 80/20 training split, batch
probes, identities, and every conditional task without creating state. `tune`
automatically reuses the only incomplete campaign with the same `plan_hash`.
`train` consumes the single completed shortlist and uses only `Training` and
`PublicTest`. `export` builds one ONNX file in staging, validates and benchmarks
that exact file, and then publishes it atomically.

The configured optimization batch is part of the recipe identity. Memory
probing may select a batch only when the configuration explicitly uses `auto`;
a fixed batch is never silently replaced. The default batch size is 128. T4 is
the strong-augmentation baseline, while T5 adds 0.05 label smoothing, five
warmup epochs, and unit-norm gradient clipping as a controlled stability
variant.

The release publisher is the only module allowed to read `PrivateTest` samples,
and it does so only after the identity ledger and all pre-private gates pass.
Predictions are never persisted; only aggregate metrics, support, and the
confusion matrix are included.

## Determinism

Seeds 42, 43, and 44, one `cuda:0` device, AMP FP16, TF32, cuDNN benchmarking,
channels-last memory format, AdamW, and the resolved DataLoader settings are
recorded. CUDA bit-for-bit reproducibility is not promised. Checkpoints maximize
predictive margin and macro-F1, then minimize loss, in lexicographic order.

## Code modules

`config.py`, `data.py`, `models.py`, and `recipes.py` define the experiment.
`engine.py` is the PyTorch adapter. `campaign.py` owns tuning and shortlist
freezing. `training.py` consumes a shortlist and publishes an immutable run
through `artifacts.py`. `cli.py` is a thin adapter; tests exercise the same
training interface as production.

Repository inputs live under `data/raw/` and `models/pretrained/`. Mutable
campaigns, runs, and release bundles live under `artifacts/`.
