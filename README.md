# Sentience - Emotion Recognition (Machine Learning)

Core Artificial Intelligence and Machine Learning module for the **Sentience** project, specialized in recognizing and classifying facial emotions using Deep Learning architectures (like MiniVGGNet). The entire flow is built on PyTorch and has native support for GPU acceleration via CUDA.

## Main Features

- **Classification**: The model can discern between 7 base emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- **Isolation and Control**: Automatic management of environments and dependencies via [mise](https://github.com/jdx/mise), locking the language version to Python 3.12.
- **Pipeline Modules**: End-to-end independent flows (training on bases and baselines, test set evaluation, 5-fold cross-validation, and metric comparisons).
- **Fast Deployment**: Automated export of `.pth` models to `.onnx` ready for production applications.

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
   pip install -r requirements.txt
   ```

## Usage

The modular entry point for the application and pipelines is the script `src/index.py`. Below are the available commands via CLI:

- **Train the MiniVGGNet model**: (Default: 80 epochs, with early stop of 12 patience epochs)
  ```bash
  python src/index.py train
  ```
  *(Optional parameter: `--epochs N` to override the amount).*

- **Test on the Evaluation set**: Run basic result validation metrics.
  ```bash
  python src/index.py evaluate
  ```

- **Map reliability via Cross-Validation (5-Fold)**:
  ```bash
  python src/index.py crossval
  ```

- **Determine Baseline**: Train and compare performance with linear base models.
  ```bash
  python src/index.py baseline
  ```

- **Graphical Comparisons**: View statistical curves.
  ```bash
  python src/index.py compare
  ```

- **Export Artifacts**: Consolidate the best global model into an optimized ONNX inference-ready structure.
  ```bash
  python src/index.py export
  ```

## Project Structure

- `src/config.py`: Global variables for metadata, paths, and hyperparameters (such as `BATCH_SIZE`, `LR`, `IMG_SIZE`).
- `src/data/`: Structural abstractions of the original dataset and basic transformations (numeric data augmentation) into tensors.
- `src/model/`: Classes containing the semantic structure of Neural Network weights.
- `src/pipeline/`: Orchestration layer for scripts triggered via CLI.
- `models/`: Persistence location for trained model checkpoints (`.pth`, `.onnx`).
- `logs/`: Persistent terminal record of train or evaluation sessions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
