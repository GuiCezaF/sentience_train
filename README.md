# sentience-train

A reproducible pipeline for tuning, training, and publishing a four-class FP32
facial-expression classifier in ONNX format. The public contract accepts
`uint8` NHWC images with shape `[N, 48, 48, 1]` and returns FP32 probabilities
with shape `[N, 4]`; preprocessing is embedded in the graph.

## Setup

The project uses Python 3.11.15 and uv 0.11.32, both pinned in `mise.toml`.
Install the locked development environment with:

```bash
uv sync --locked --group dev
```

Before running the pipeline, provide these files on the local machine:

- `data/raw/fer2013.csv`
- `models/pretrained/mobilenet_v3_small-047dcff4.pth`
- `models/pretrained/resnet18-f37072fd.pth`

The pipeline never downloads data or weights. It validates the expected
filenames, sizes, and SHA-256 digests of the ImageNet checkpoints before use.

## Workflow

```bash
uv run sentience-train tune --config configs/tuning.yaml --plan
uv run sentience-train tune --config configs/tuning.yaml
uv run sentience-train train --config configs/default.yaml
uv run sentience-train export --run-id <run-id>
```

`tune --plan` is read-only. `tune` creates or resumes a campaign with the same
identity. `train` consumes the frozen shortlist and atomically publishes a
training run. `export` validates and benchmarks the exact ONNX artifact
published to `artifacts/exports/<run-id>/`.

The CLI returns exit code 0 for success or reuse, 2 for invalid or incompatible
input, and 1 for execution failure, an exhausted campaign, or a failed gate.

## Repository layout

```text
configs/                  versioned experiment configurations
data/raw/                 local source dataset
models/pretrained/        validated source weights
artifacts/
  campaigns/              tuning plans, evidence, history, and checkpoints
  runs/                   final training runs and selected checkpoints
  exports/                published ONNX bundles
  archive/campaigns/      locally archived campaigns
reports/                  generated test and evaluation reports
schemas/                  public JSON contracts
src/sentience_train/      pipeline implementation
tests/                    public-interface tests
docs/                     context, decisions, and contracts
```

`data/`, `models/`, `artifacts/`, and generated files under `reports/` remain
local and are protected by `.gitignore`; only their `README.md` files are
versioned. This keeps reproducible code and configuration separate from large
or potentially private data, weights, checkpoints, and results.

## Training recipe

The default optimization batch size is 128. T4 is the strong-augmentation
baseline. T5 is a controlled stability variant with 0.05 label smoothing, five
linear warmup epochs, and gradient clipping with a unit norm. After warmup, the
learning rate follows cosine decay.

Every recipe change contributes to the `plan_hash`, creating a new campaign
without overwriting previous results. The fixed architectures are DS-CNN,
MobileNetV3-Small, and ResNet-18, trained with seeds 42, 43, and 44.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv build
```

Acceptance tests that require a real training run are disabled by default. See
[docs/onnx-only.md](docs/onnx-only.md) for partition isolation, determinism, and
publication details.
