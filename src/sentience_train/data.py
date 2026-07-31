"""Streaming FER2013 validation and lazy Training/PublicTest datasets."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from .config import AugmentationConfig, DataConfig, LABELS

CANONICAL_FER2013_SHA256 = "3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b"
EXPECTED_HEADER = ["emotion", "pixels", "Usage"]
USAGES = ("Training", "PublicTest", "PrivateTest")
EMOTION_TO_LABEL = {0: 0, 3: 1, 6: 2, 4: 3}


class DataValidationError(ValueError):
    """Raised for a CSV whose schema or FER2013 records are invalid."""


def _error(path: Path, line: int, field: str, received: Any, expected: str) -> DataValidationError:
    return DataValidationError(
        f"{path}: line {line}, field {field}: received {received!r}; expected {expected}"
    )


@dataclass(frozen=True)
class CsvProvenance:
    path: Path
    sha256: str
    release_evidence: bool
    header: tuple[str, str, str]
    total_rows: int
    partition_counts: dict[str, int]
    raw_class_counts: dict[str, dict[str, int]]
    selected_class_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class Record:
    offset: int
    label: int
    line_number: int
    row_sha256: str = ""


class FER2013Dataset(Dataset[tuple[torch.Tensor, int]]):
    """A dataset which reads exactly one selected record on each access."""

    def __init__(
        self,
        csv_path: Path,
        records: tuple[Record, ...],
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        self._csv_path = csv_path
        self._records = records
        self._augmentation = augmentation

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[Record, ...]:
        """Validated Training records, exposed for deterministic calibration only."""
        return self._records

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._records[index]
        with self._csv_path.open("rb") as handle:
            handle.seek(record.offset)
            raw = handle.readline()
        try:
            row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
            pixels = [int(value) for value in row[1].split()]
        except (UnicodeDecodeError, ValueError, csv.Error) as exc:  # protected by validation
            raise RuntimeError(f"CSV changed after validation at {self._csv_path}:{record.line_number}") from exc
        image = torch.tensor(pixels, dtype=torch.float32).reshape(1, 48, 48) / 255.0
        if self._augmentation is not None:
            image = _augment(image, self._augmentation)
        # Explicit FP32 conversion keeps the public transform independent of AMP.
        return image.to(dtype=torch.float32).mul_(2.0).sub_(1.0), record.label


def _augment(image: torch.Tensor, config: AugmentationConfig) -> torch.Tensor:
    if torch.rand(()) < config.horizontal_flip_probability:
        image = F.hflip(image)
    angle = float(torch.empty(()).uniform_(-config.rotation_degrees, config.rotation_degrees))
    translation = [
        int(torch.empty(()).uniform_(-config.translate_fraction, config.translate_fraction).item() * 48),
        int(torch.empty(()).uniform_(-config.translate_fraction, config.translate_fraction).item() * 48),
    ]
    scale = float(torch.empty(()).uniform_(config.scale_min, config.scale_max))
    image = F.affine(image, angle=angle, translate=translation, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR)
    brightness = 1.0 + float(torch.empty(()).uniform_(-config.brightness, config.brightness))
    contrast = 1.0 + float(torch.empty(()).uniform_(-config.contrast, config.contrast))
    return F.adjust_contrast(F.adjust_brightness(image, brightness), contrast).clamp_(0.0, 1.0)


@dataclass(frozen=True)
class TrainingData:
    """The engine's complete view of data: only Training and PublicTest datasets."""

    train: FER2013Dataset
    validation: FER2013Dataset
    provenance: CsvProvenance
    train_labels: tuple[int, ...]


def _parse_row(path: Path, line_number: int, raw: bytes) -> tuple[int, list[int], str]:
    try:
        decoded = raw.decode("utf-8").rstrip("\r\n")
        row = next(csv.reader([decoded]))
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise _error(path, line_number, "row", raw[:80], "a valid UTF-8 CSV row") from exc
    if len(row) != 3:
        raise _error(path, line_number, "row", row, "exactly 3 fields: emotion,pixels,Usage")
    try:
        emotion = int(row[0])
    except ValueError as exc:
        raise _error(path, line_number, "emotion", row[0], "an integer from 0 through 6") from exc
    if emotion < 0 or emotion > 6:
        raise _error(path, line_number, "emotion", row[0], "an integer from 0 through 6")
    if row[2] not in USAGES:
        raise _error(path, line_number, "Usage", row[2], "Training, PublicTest, or PrivateTest")
    tokens = row[1].split()
    if len(tokens) != 2304:
        raise _error(path, line_number, "pixels", f"{len(tokens)} values", "exactly 2304 integers from 0 through 255")
    pixels: list[int] = []
    for position, value in enumerate(tokens):
        try:
            pixel = int(value)
        except ValueError as exc:
            raise _error(path, line_number, "pixels", value, f"integer 0..255 at position {position}") from exc
        if pixel < 0 or pixel > 255:
            raise _error(path, line_number, "pixels", pixel, f"integer 0..255 at position {position}")
        pixels.append(pixel)
    return emotion, pixels, row[2]


def load_training_data(data_config: DataConfig, augmentation: AugmentationConfig | None = None) -> TrainingData:
    """Validate the whole CSV in one pass while retaining no PrivateTest offsets."""
    path = data_config.csv_path
    hasher = hashlib.sha256()
    partition_counts: Counter[str] = Counter()
    raw_counts: dict[str, Counter[int]] = {usage: Counter() for usage in USAGES}
    selected_counts: dict[str, Counter[int]] = {usage: Counter() for usage in USAGES}
    records: dict[Literal["Training", "PublicTest"], list[Record]] = {"Training": [], "PublicTest": []}
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise DataValidationError(f"cannot read dataset {path}: {exc}") from exc
    with handle:
        offset = 0
        header_raw = handle.readline()
        hasher.update(header_raw)
        try:
            header = next(csv.reader([header_raw.decode("utf-8").rstrip("\r\n")]))
        except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
            raise _error(path, 1, "header", header_raw[:80], "emotion,pixels,Usage") from exc
        if header != EXPECTED_HEADER:
            raise _error(path, 1, "header", header, "exactly emotion,pixels,Usage")
        offset += len(header_raw)
        line_number = 1
        while raw := handle.readline():
            line_number += 1
            hasher.update(raw)
            emotion, _pixels, usage = _parse_row(path, line_number, raw)
            partition_counts[usage] += 1
            raw_counts[usage][emotion] += 1
            if emotion in EMOTION_TO_LABEL:
                label = EMOTION_TO_LABEL[emotion]
                selected_counts[usage][label] += 1
                if usage in records:
                    records[usage].append(
                        Record(
                            offset=offset,
                            label=label,
                            line_number=line_number,
                            row_sha256=hashlib.sha256(raw).hexdigest(),
                        )
                    )
            offset += len(raw)
    for usage in USAGES:
        missing = [LABELS[index] for index in range(len(LABELS)) if selected_counts[usage][index] == 0]
        if missing:
            raise DataValidationError(f"{path}: partition {usage} is missing required class(es): {', '.join(missing)}")
    provenance = CsvProvenance(
        path=path,
        sha256=hasher.hexdigest(),
        release_evidence=hasher.hexdigest() == CANONICAL_FER2013_SHA256,
        header=tuple(EXPECTED_HEADER),
        total_rows=sum(partition_counts.values()),
        partition_counts={usage: partition_counts[usage] for usage in USAGES},
        raw_class_counts={usage: {str(label): raw_counts[usage][label] for label in range(7)} for usage in USAGES},
        selected_class_counts={usage: {LABELS[label]: selected_counts[usage][label] for label in range(4)} for usage in USAGES},
    )
    train_records = tuple(records["Training"])
    validation_records = tuple(records["PublicTest"])
    return TrainingData(
        train=FER2013Dataset(path, train_records, augmentation),
        validation=FER2013Dataset(path, validation_records),
        provenance=provenance,
        train_labels=tuple(record.label for record in train_records),
    )
