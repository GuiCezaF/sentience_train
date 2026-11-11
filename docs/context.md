# Project context

This project classifies four emotions in 48-by-48 monochrome face crops: angry,
happy, neutral, and sad. The logit order is stable and independent of any
external source ordering.

The pipeline exposes four interfaces: campaign planning, campaign execution,
full training, and publication. Each interface hides its implementation behind
a deep module. Synthetic adapters exist only in tests and always identify the
evidence they produce.

The implementation is organized by responsibility under
`src/sentience_train/`: `core/` contains configuration, data, models, recipes,
and metrics; `engine/` contains the PyTorch trainer and atomic run staging;
`onnx/` contains graph construction and host benchmarking; and `workflows/`
contains campaign, full-training, export, and release orchestration. The root
`cli.py` remains the input and exit-code adapter. Its package initializer maps
legacy top-level module imports to the same module objects, preserving existing
imports and monkeypatch targets.

Tuning accesses only views of the `Training` partition, split deterministically
into 80% optimization and 20% tuning subsets. Full training accesses `Training`
and `PublicTest`. The scanner may validate `PrivateTest` bytes and counts but
does not expose its samples. Only the release module materializes `PrivateTest`,
after every other gate has passed.

The public artifact is an FP32 ONNX graph containing the
`float32(pixel) / 127.5 - 1.0` normalization, NHWC-to-NCHW layout conversion,
and softmax. Benchmarking and publication use the same bytes. Bundles are
immutable and contain a hash for every payload plus a canonical self-digest.

Input data and weights live under `data/raw/` and `models/pretrained/`. All
produced state lives under `artifacts/`, keeping code and configuration
independent of any specific execution. Tests are grouped under
`tests/contracts/`, `tests/workflows/`, and `tests/acceptance/`.
