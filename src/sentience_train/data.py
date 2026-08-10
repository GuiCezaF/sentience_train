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

from .config import AugmentationConfig, DataConfig, LABELS, TUNING_SPLIT_SCHEMA, canonical_hash

CANONICAL_FER2013_SHA256 = "3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b"
EXPECTED_HEADER = ["emotion", "pixels", "Usage"]
USAGES = ("Training", "PublicTest", "PrivateTest")
EMOTION_TO_LABEL = {0: 0, 3: 1, 6: 2, 4: 3}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DataValidationError(ValueError):
    """Raised for a CSV whose schema or FER2013 records are invalid."""


def _error(path: Path, line: int, field: str, received: Any, expected: str) -> DataValidationError:
    return DataValidationError(f"{path}: line {line}, field {field}: received {received!r}; expected {expected}")


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
    global_index: int = -1


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

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def with_augmentation(self, augmentation: AugmentationConfig) -> "FER2013Dataset":
        """Return the same validated view with a trial-specific augmentation."""
        return FER2013Dataset(self._csv_path, self._records, augmentation)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._records[index]
        with self._csv_path.open("rb") as handle:
            handle.seek(record.offset)
            raw = handle.readline()
        try:
            row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
            pixels = [int(value) for value in row[1].split()]
        except (
            UnicodeDecodeError,
            ValueError,
            csv.Error,
        ) as exc:  # protected by validation
            raise RuntimeError(f"CSV changed after validation at {self._csv_path}:{record.line_number}") from exc
        image = torch.tensor(pixels, dtype=torch.float32).reshape(1, 48, 48) / 255.0
        if self._augmentation is not None:
            image = _augment(image, self._augmentation)
        # Explicit FP32 conversion keeps the public transform independent of AMP.
        return image.to(dtype=torch.float32).mul_(2.0).sub_(1.0), record.label


def _augment(image: torch.Tensor, config: AugmentationConfig) -> torch.Tensor:
    from torchvision.transforms import functional as F
    from torchvision.transforms.functional import InterpolationMode

    if torch.rand(()) < config.horizontal_flip_probability:
        image = F.hflip(image)
    angle = float(torch.empty(()).uniform_(-config.rotation_degrees, config.rotation_degrees))
    translation = [
        int(torch.empty(()).uniform_(-config.translate_fraction, config.translate_fraction).item() * 48),
        int(torch.empty(()).uniform_(-config.translate_fraction, config.translate_fraction).item() * 48),
    ]
    scale = float(torch.empty(()).uniform_(config.scale_min, config.scale_max))
    image = F.affine(
        image,
        angle=angle,
        translate=translation,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
    )
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


def _parse_row(
    path: Path, line_number: int, raw: bytes, *, materialize_pixels: bool = True, validate_pixels: bool = True
) -> tuple[int, list[int], str]:
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
    if not validate_pixels:
        return emotion, [], row[2]
    tokens = row[1].split()
    if len(tokens) != 2304:
        raise _error(path, line_number, "pixels", f"{len(tokens)} values", "exactly 2304 integers from 0 through 255")
    if not materialize_pixels:
        for position, value in enumerate(tokens):
            try:
                pixel = int(value)
            except ValueError as exc:
                raise _error(path, line_number, "pixels", value, f"integer 0..255 at position {position}") from exc
            if pixel < 0 or pixel > 255:
                raise _error(path, line_number, "pixels", pixel, f"integer 0..255 at position {position}")
        return emotion, [], row[2]
    pixels: list[int] = []
    for position, value in enumerate(tokens):
        try:
            pixel = int(value)
        except ValueError as exc:
            raise _error(
                path,
                line_number,
                "pixels",
                value,
                f"integer 0..255 at position {position}",
            ) from exc
        if pixel < 0 or pixel > 255:
            raise _error(
                path,
                line_number,
                "pixels",
                pixel,
                f"integer 0..255 at position {position}",
            )
        pixels.append(pixel)
    return emotion, pixels, row[2]


def load_training_data(data_config: DataConfig, augmentation: AugmentationConfig | None = None) -> TrainingData:
    """Validate the whole CSV in one pass while retaining no PrivateTest offsets."""
    path = data_config.csv_path
    hasher = hashlib.sha256()
    partition_counts: Counter[str] = Counter()
    raw_counts: dict[str, Counter[int]] = {usage: Counter() for usage in USAGES}
    selected_counts: dict[str, Counter[int]] = {usage: Counter() for usage in USAGES}
    records: dict[Literal["Training", "PublicTest"], list[Record]] = {
        "Training": [],
        "PublicTest": [],
    }
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
            emotion, _pixels, usage = _parse_row(path, line_number, raw, materialize_pixels=False)
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
                            global_index=len(records["Training"]),
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
        selected_class_counts={
            usage: {LABELS[label]: selected_counts[usage][label] for label in range(4)} for usage in USAGES
        },
    )
    train_records = tuple(records["Training"])
    validation_records = tuple(records["PublicTest"])
    return TrainingData(
        train=FER2013Dataset(path, train_records, augmentation),
        validation=FER2013Dataset(path, validation_records),
        provenance=provenance,
        train_labels=tuple(record.label for record in train_records),
    )


@dataclass(frozen=True)
class TuningSplitEntry:
    """A persisted reference to a Training row, never to PublicTest/PrivateTest."""

    global_index: int
    label: int
    row_sha256: str
    rank: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_index": self.global_index,
            "label": self.label,
            "row_sha256": self.row_sha256,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class TuningSplit:
    schema: str
    tuning_fraction: float
    tuning: tuple[TuningSplitEntry, ...]
    optimization: tuple[TuningSplitEntry, ...]
    class_counts: dict[str, int]
    tuning_class_counts: dict[str, int]
    optimization_class_counts: dict[str, int]
    tuning_hash: str
    optimization_hash: str
    split_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "partition": "Training",
            "tuning_fraction": self.tuning_fraction,
            "class_counts": self.class_counts,
            "tuning_class_counts": self.tuning_class_counts,
            "optimization_class_counts": self.optimization_class_counts,
            "tuning": [entry.to_dict() for entry in self.tuning],
            "optimization": [entry.to_dict() for entry in self.optimization],
            "tuning_hash": self.tuning_hash,
            "optimization_hash": self.optimization_hash,
            "split_hash": self.split_hash,
        }


@dataclass(frozen=True)
class TuningData:
    """Data view exposed to tuning; it has no PublicTest or PrivateTest view."""

    optimization: FER2013Dataset
    tuning: FER2013Dataset
    provenance: CsvProvenance
    split: TuningSplit

    @property
    def optimization_labels(self) -> tuple[int, ...]:
        return tuple(record.label for record in self.optimization.records)


def tuning_rank(row_sha256: str, *, schema: str = TUNING_SPLIT_SCHEMA) -> str:
    """Return the namespaced rank mandated by tuning-split/v1."""
    if schema != TUNING_SPLIT_SCHEMA:
        raise DataValidationError(f"unsupported tuning split schema {schema!r}")
    try:
        row_bytes = bytes.fromhex(row_sha256)
    except ValueError as exc:
        raise DataValidationError(f"row SHA-256 is not hexadecimal: {row_sha256!r}") from exc
    if len(row_bytes) != hashlib.sha256().digest_size:
        raise DataValidationError("row SHA-256 must contain 32 bytes")
    return hashlib.sha256(b"sentience-train:tuning-split/v1\0" + row_bytes).hexdigest()


def build_tuning_split(
    records: tuple[Record, ...] | list[Record], *, fraction: float = 0.20, schema: str = TUNING_SPLIT_SCHEMA
) -> TuningSplit:
    """Build the deterministic stratified split independently of model seeds."""
    if schema != TUNING_SPLIT_SCHEMA:
        raise DataValidationError(f"unsupported tuning split schema {schema!r}")
    if fraction != 0.20:
        raise DataValidationError("tuning fraction is fixed to 0.20")
    entries = [
        TuningSplitEntry(
            record.global_index if record.global_index >= 0 else index,
            record.label,
            record.row_sha256,
            tuning_rank(record.row_sha256, schema=schema),
        )
        for index, record in enumerate(records)
    ]
    by_class: dict[int, list[TuningSplitEntry]] = {index: [] for index in range(len(LABELS))}
    for entry in entries:
        by_class.setdefault(entry.label, []).append(entry)
    selected: list[TuningSplitEntry] = []
    for label in range(len(LABELS)):
        class_entries = sorted(
            by_class.get(label, ()), key=lambda item: (item.rank, item.row_sha256, item.global_index)
        )
        selected.extend(class_entries[: int(len(class_entries) * fraction)])
    tuning = tuple(sorted(selected, key=lambda item: item.global_index))
    tuning_indices = {entry.global_index for entry in tuning}
    optimization = tuple(
        sorted(
            (entry for entry in entries if entry.global_index not in tuning_indices), key=lambda item: item.global_index
        )
    )
    class_counts = {LABELS[label]: len(by_class.get(label, ())) for label in range(len(LABELS))}
    tuning_counts = {LABELS[label]: sum(entry.label == label for entry in tuning) for label in range(len(LABELS))}
    optimization_counts = {
        LABELS[label]: sum(entry.label == label for entry in optimization) for label in range(len(LABELS))
    }
    tuning_payload = [entry.to_dict() for entry in tuning]
    optimization_payload = [entry.to_dict() for entry in optimization]
    tuning_hash = canonical_hash(tuning_payload)
    optimization_hash = canonical_hash(optimization_payload)
    split_payload = {
        "schema": schema,
        "tuning_fraction": fraction,
        "class_counts": class_counts,
        "tuning": tuning_payload,
        "optimization": optimization_payload,
    }
    return TuningSplit(
        schema=schema,
        tuning_fraction=fraction,
        tuning=tuning,
        optimization=optimization,
        class_counts=class_counts,
        tuning_class_counts=tuning_counts,
        optimization_class_counts=optimization_counts,
        tuning_hash=tuning_hash,
        optimization_hash=optimization_hash,
        split_hash=canonical_hash(split_payload),
    )


def _scan_csv_for_tuning(data_config: DataConfig) -> tuple[CsvProvenance, tuple[Record, ...]]:
    """Validate/hash the complete CSV while retaining only target Training offsets."""
    path = data_config.csv_path
    hasher = hashlib.sha256()
    partition_counts: Counter[str] = Counter()
    raw_counts: dict[str, Counter[int]] = {usage: Counter() for usage in USAGES}
    selected_counts: dict[str, Counter[int]] = {usage: Counter() for usage in USAGES}
    training_records: list[Record] = []
    canonical_bytes = _file_sha256(path) == CANONICAL_FER2013_SHA256
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
            emotion, _pixels, usage = _parse_row(
                path, line_number, raw, materialize_pixels=False, validate_pixels=not canonical_bytes
            )
            partition_counts[usage] += 1
            raw_counts[usage][emotion] += 1
            if emotion in EMOTION_TO_LABEL:
                label = EMOTION_TO_LABEL[emotion]
                selected_counts[usage][label] += 1
                if usage == "Training":
                    training_records.append(
                        Record(
                            offset=offset,
                            label=label,
                            line_number=line_number,
                            row_sha256=hashlib.sha256(raw).hexdigest(),
                            global_index=len(training_records),
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
        selected_class_counts={
            usage: {LABELS[label]: selected_counts[usage][label] for label in range(4)} for usage in USAGES
        },
    )
    return provenance, tuple(training_records)


def load_tuning_data(
    data_config: DataConfig, *, augmentation: AugmentationConfig | None = None, fraction: float = 0.20
) -> TuningData:
    """Materialize optimization/tuning views from Training records only."""
    provenance, records = _scan_csv_for_tuning(data_config)
    split = build_tuning_split(records, fraction=fraction)
    by_index = {record.global_index: record for record in records}
    optimization_records = tuple(by_index[entry.global_index] for entry in split.optimization)
    tuning_records = tuple(by_index[entry.global_index] for entry in split.tuning)
    return TuningData(
        optimization=FER2013Dataset(data_config.csv_path, optimization_records, augmentation),
        tuning=FER2013Dataset(data_config.csv_path, tuning_records),
        provenance=provenance,
        split=split,
    )
