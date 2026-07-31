"""The model seam for the current provisional MiniVGG architecture."""

from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )

    def forward(self, value):  # type: ignore[no-untyped-def]
        return self.layers(value)


class MiniVGG(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(_ConvBlock(1, 32), _ConvBlock(32, 64), _ConvBlock(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6 * 6, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, value):  # type: ignore[no-untyped-def]
        return self.classifier(self.features(value))


def build_model(config: ModelConfig) -> nn.Module:
    if config.name != "minivgg":  # also defend direct calls that bypass load_config
        raise ValueError(f"unsupported model {config.name!r}")
    return MiniVGG(config.num_classes)
