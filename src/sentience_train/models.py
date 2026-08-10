"""The three fixed FP32 model implementations and their input adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .config import ARCHITECTURES

WEIGHT_FILES = {
    "MobileNetV3-Small": (
        "mobilenet_v3_small-047dcff4.pth",
        10_306_551,
        "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
    ),
    "ResNet-18": (
        "resnet18-f37072fd.pth",
        46_830_571,
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    ),
}


class ModelError(ValueError):
    pass


def _conv(in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class _DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = _conv(in_channels, in_channels, stride=stride, groups=in_channels)
        self.pointwise = _conv(in_channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(value))


class DSCNN(nn.Module):
    """MobileNetV1-like DS-CNN at the mandated width 1.0."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv(1, 32, stride=2),
            _DepthwiseSeparable(32, 64, 1),
            _DepthwiseSeparable(64, 128, 2),
            _DepthwiseSeparable(128, 128, 1),
            _DepthwiseSeparable(128, 256, 2),
            _DepthwiseSeparable(256, 256, 1),
            _DepthwiseSeparable(256, 256, 2),
            *[_DepthwiseSeparable(256, 256, 1) for _ in range(2)],
            _DepthwiseSeparable(256, 256, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(value)).flatten(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_identity(architecture: str, weights_dir: str | Path) -> dict[str, object]:
    """Return the immutable local identity used by campaign planning."""
    if architecture == "DS-CNN":
        return {"status": "not-applicable", "filename": None}
    try:
        filename, expected_size, expected_hash = WEIGHT_FILES[architecture]
    except KeyError as exc:
        raise ModelError(f"{architecture!r} does not use an external checkpoint") from exc
    path = Path(weights_dir) / filename
    identity: dict[str, object] = {
        "filename": filename,
        "expected_size": expected_size,
        "expected_sha256": expected_hash,
    }
    if not path.is_file():
        identity["status"] = "missing"
        return identity
    size = path.stat().st_size
    actual_hash = _sha256(path)
    identity.update(
        {
            "status": "present" if size == expected_size and actual_hash == expected_hash else "invalid",
            "size": size,
            "sha256": actual_hash,
        }
    )
    return identity


def validate_weight_file(architecture: str, weights_dir: str | Path) -> Path:
    try:
        filename, expected_size, expected_hash = WEIGHT_FILES[architecture]
    except KeyError as exc:
        raise ModelError(f"{architecture!r} does not use an external checkpoint") from exc
    path = Path(weights_dir) / filename
    if not path.is_file():
        raise ModelError(f"required checkpoint is absent; downloads are disabled: {path}")
    if path.stat().st_size != expected_size:
        raise ModelError(f"{path.name} has size {path.stat().st_size}, expected {expected_size}")
    actual = _sha256(path)
    if actual != expected_hash:
        raise ModelError(f"{path.name} SHA-256 mismatch: {actual}")
    return path


def adapt_rgb_kernel(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 4 or weight.shape[1] != 3:
        raise ModelError("ImageNet stem must be a four-dimensional RGB convolution kernel")
    return weight.sum(dim=1, keepdim=True)


def _load_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ModelError("checkpoint must contain a state-dict object")
    return value.get("state_dict", value)  # type: ignore[return-value]


def _image_model(architecture: str) -> nn.Module:
    try:
        from torchvision.models import (
            mobilenet_v3_small,
            resnet18,
        )
    except Exception as exc:
        raise ModelError(f"torchvision is unavailable: {exc}") from exc
    if architecture == "MobileNetV3-Small":
        return mobilenet_v3_small(weights=None)
    if architecture == "ResNet-18":
        return resnet18(weights=None)
    raise ModelError(f"unsupported architecture {architecture!r}")


def build_model(architecture: str, *, weights_dir: str | Path = "models/pretrained", num_classes: int = 4) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ModelError(f"architecture must be one of {list(ARCHITECTURES)!r}")
    if architecture == "DS-CNN":
        return DSCNN(num_classes)
    checkpoint = validate_weight_file(architecture, weights_dir)
    model = _image_model(architecture)
    state = dict(_load_checkpoint(checkpoint))
    if architecture == "MobileNetV3-Small":
        first_key = "features.0.0.weight"
        first = state.pop(first_key, None)
        if first is None:
            raise ModelError(f"checkpoint missing {first_key}")
        features = getattr(model, "features", None)
        classifier = getattr(model, "classifier", None)
        if not isinstance(features, nn.Sequential) or not isinstance(classifier, nn.Sequential):
            raise ModelError("MobileNetV3-Small has an unexpected module structure")
        first_block = features[0]
        final_classifier = classifier[-1]
        if not isinstance(first_block, nn.Sequential) or not isinstance(first_block[0], nn.Conv2d):
            raise ModelError("MobileNetV3-Small has an unexpected input stem")
        if not isinstance(final_classifier, nn.Linear):
            raise ModelError("MobileNetV3-Small has an unexpected classifier head")
        first_conv = first_block[0]
        first_block[0] = nn.Conv2d(1, first_conv.out_channels, 3, 2, 1, bias=False)
        state[first_key] = adapt_rgb_kernel(first)
        classifier[-1] = nn.Linear(final_classifier.in_features, num_classes)
        head_prefix = f"classifier.{len(classifier) - 1}."
        state = {key: value for key, value in state.items() if not key.startswith(head_prefix)}
        allowed_missing = {f"{head_prefix}weight", f"{head_prefix}bias"}
    else:
        first_key = "conv1.weight"
        first = state.pop(first_key, None)
        if first is None:
            raise ModelError(f"checkpoint missing {first_key}")
        first_conv = getattr(model, "conv1", None)
        classifier = getattr(model, "fc", None)
        if not isinstance(first_conv, nn.Conv2d) or not isinstance(classifier, nn.Linear):
            raise ModelError("ResNet-18 has an unexpected module structure")
        model.conv1 = nn.Conv2d(1, first_conv.out_channels, 7, 2, 3, bias=False)
        state[first_key] = adapt_rgb_kernel(first)
        model.fc = nn.Linear(classifier.in_features, num_classes)
        state = {key: value for key, value in state.items() if not key.startswith("fc.")}
        allowed_missing = {"fc.weight", "fc.bias"}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(key not in allowed_missing for key in missing):
        raise ModelError(f"checkpoint adaptation failed; missing={missing}, unexpected={unexpected}")
    return model


class PublicOnnxWrapper(nn.Module):
    """The complete public contract: uint8 NHWC to FP32 probabilities."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = images.to(dtype=torch.float32) / 127.5 - 1.0
        value = value.permute(0, 3, 1, 2).contiguous()
        return torch.softmax(self.model(value), dim=1)
