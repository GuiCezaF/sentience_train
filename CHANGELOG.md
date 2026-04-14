# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
