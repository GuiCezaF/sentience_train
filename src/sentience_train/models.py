"""Deployable depthwise-separable CNN candidates.

MiniVGG deliberately does not appear at this public seam.  It may return as a
private teacher implementation, but every model users can select is one of the
three bounded DS-CNN widths below.
"""

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


def _channels(value: int, width: float) -> int:
    """Scale a MobileNet channel count and round up to a multiple of eight."""
    return max(8, ((int(value * width) + 7) // 8) * 8)


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
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
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
        blocks = ((c64, 1), (c128, 2), (c128, 1), (c256, 2), (c256, 1), (c512, 2))
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
    return build_candidate(config.name, config.num_classes)
