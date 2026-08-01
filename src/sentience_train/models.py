"""Deployable depthwise-separable CNN candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig


@dataclass(frozen=True)
class DSCNNBaseline:
    name: str
    width: float
    parameters: int
    macs: int


DSCNN_BASELINES: tuple[DSCNNBaseline, ...] = (
    DSCNNBaseline("ds_cnn_w025", 0.25, 213_956, 1_968_448),
    DSCNNBaseline("ds_cnn_w035", 0.35, 403_908, 3_857_256),
    DSCNNBaseline("ds_cnn_w050", 0.50, 820_356, 7_377_536),
)
_BY_NAME = {candidate.name: candidate for candidate in DSCNN_BASELINES}


def make_divisible(value: float, divisor: int = 8, min_ratio: float = 0.9) -> int:
    """Return MobileNet's standard channel-width rounding.

    MobileNet does not blindly ceil a scaled channel count.  It rounds to the
    nearest divisor and only moves up when that would shrink the requested
    width by more than ten percent.  The distinction matters for the 0.35
    candidate (179.2 becomes 176, not 184) and keeps the checked-in structural
    baselines reproducible.
    """
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < min_ratio * value:
        rounded += divisor
    return int(rounded)


def _channels(value: int, width: float) -> int:
    """Scale a MobileNet channel count with the canonical make-divisible rule."""
    return make_divisible(value * width)


class _ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=False),
        )


class _DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=False),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(value))


class DSCNN(nn.Module):
    """MobileNetV1 topology constrained to the TFLM deployment operator set."""

    def __init__(self, width: float, num_classes: int = 4) -> None:
        super().__init__()
        channels = [_channels(value, width) for value in (32, 64, 128, 256, 512, 1024)]
        stem, c64, c128, c256, c512, c1024 = channels
        blocks: tuple[tuple[int, int], ...] = ((c64, 1), (c128, 2), (c128, 1), (c256, 2), (c256, 1), (c512, 2))
        blocks += ((c512, 1),) * 5
        blocks += ((c1024, 2), (c1024, 1))
        layers: list[nn.Module] = [_ConvBNReLU6(1, stem, 2)]
        current = stem
        for out_channels, stride in blocks:
            layers.append(_DepthwiseSeparable(current, out_channels, stride))
            current = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(current, num_classes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.features(value)
        value = self.pool(value)
        return self.classifier(torch.flatten(value, 1))


class NHWCModel(nn.Module):
    """Conversion-only adapter: LiteRT sees NHWC while PyTorch remains NCHW."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value.permute(0, 3, 1, 2).contiguous())


def candidate_for_name(name: str) -> DSCNNBaseline:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model {name!r}") from exc


def build_candidate(name: str, num_classes: int = 4) -> DSCNN:
    candidate = candidate_for_name(name)
    return DSCNN(candidate.width, num_classes)


def build_model(config: ModelConfig) -> nn.Module:
    return build_candidate("ds_cnn_w025", config.num_classes)
