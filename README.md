# Sentience - Emotion Recognition (Machine Learning)

Core Artificial Intelligence and Machine Learning module for the **Sentience** project, specialized in recognizing and classifying facial emotions to measure workplace stress and satisfaction. It uses Deep Learning architectures (like MiniVGGNet) built on PyTorch with native support for GPU acceleration via CUDA.

## Main Features

- **Classification**: Specialized in 4 key classes for professional environment monitoring: `angry`, `happy`, `neutral`, `sad`.
- **Isolation and Control**: Automatic management of environments and dependencies via [mise](https://github.com/jdx/mise), locking the language version to Python 3.12.
- **Pipeline Modules**: Deterministic training and explicit development exports from frozen runs.
- **Fast Deployment**: Hash-checked ONNX and byte-for-byte TFLite bundles for development consumers.

## Technologies and Dependencies

- **Language**: Python 3.12 (via `mise`)
- **Core ML**: PyTorch (CUDA 12.1+ support), Torchvision
- **Export and Inference**: ONNX, ONNXRuntime
- **Data Analysis and Metrics**: Scikit-Learn, Numpy, Matplotlib, Seaborn
- **Additional Utilities**: tqdm (progress bars), tabulate

## Installation and Setup

1. Make sure you have the universal installer [mise](https://github.com/jdx/mise) on your machine.
2. In the root of this project (`ml/`), `mise` will automatically detect the required version and provision the virtual environment `.venv`.
3. With the environment activated, install the dependencies:
   ```bash
   uv sync --locked --group dev --group export
   ```

## Usage

The public workflow is explicit and run-based:

```bash
sentience-train train --config configs/default.yaml
sentience-train export --run-id <run-id>
sentience-train export --run-id <run-id> --format onnx --runs-dir ./runs
sentience-train export --run-id <run-id> --format tflite --runs-dir ./runs
```

`--format all` is the default release flow. `onnx` and `tflite` are immutable
development exports and consume only `PublicTest`; they never create a
PrivateTest ledger. The complete bundle is published under
`exports/<run-id>/all/` and is reused only when its candidate identity and every
payload hash match. A final release always has status `accepted` or `failed`
and the frozen run remains untouched.

## Project Structure

- `src/config.py`: Global variables for metadata, paths, and hyperparameters (such as `BATCH_SIZE`, `LR`, `IMG_SIZE`).
- `src/data/`: Structural abstractions of the original dataset and basic transformations (numeric data augmentation) into tensors.
- `src/model/`: Classes containing the semantic structure of Neural Network weights.
- `src/pipeline/`: Orchestration layer for scripts triggered via CLI.
- `runs/`: Immutable frozen training runs, including indexed `PublicTest` records.
- `exports/`: Atomic development bundles and complete release bundles with the
  permanent PrivateTest access ledger
- `docs/deterministic-release.md`: deterministic release and retry rules
- `logs/`: Persistent terminal record of train or evaluation sessions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
