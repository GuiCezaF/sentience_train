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
sentience-train export --run-id <run-id> --format onnx
sentience-train export --run-id <run-id> --format tflite --runs-dir ./runs
```

`--format` is required and issue #5 accepts only `onnx` or `tflite`. `all` and
any access to `PrivateTest` are reserved for issue #6. Bundles are published
under `exports/<run-id>/<format>/`, atomically, and reused only when their
identity and every payload hash match.

## Project Structure

- `src/config.py`: Global variables for metadata, paths, and hyperparameters (such as `BATCH_SIZE`, `LR`, `IMG_SIZE`).
- `src/data/`: Structural abstractions of the original dataset and basic transformations (numeric data augmentation) into tensors.
- `src/model/`: Classes containing the semantic structure of Neural Network weights.
- `src/pipeline/`: Orchestration layer for scripts triggered via CLI.
- `runs/`: Immutable frozen training runs, including indexed `PublicTest` records.
- `exports/`: Atomic ONNX or copied INT8 TFLite development bundles.
- `logs/`: Persistent terminal record of train or evaluation sessions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
