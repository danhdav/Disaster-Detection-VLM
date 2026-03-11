"""
6-channel building damage classifier: [preRGB || postRGB] -> 4 damage classes.
Matches the SixChannelCNN architecture from Benchmark-Model-xView2.
"""
from __future__ import annotations

import torch
import torch.nn as nn


DAMAGE_CLASSES  = ["no-damage", "minor-damage", "major-damage", "destroyed"]
DAMAGE_SEVERITY = ["LOW",       "MODERATE",     "HIGH",         "SEVERE"]


class SixChannelCNN(nn.Module):
    """
    4-block CNN on 6-channel (pre+post RGB) input -> 4 damage classes.
    Input shape: (B, 6, H, W)  channels: preRGB(0:3) | postRGB(3:6).
    Architecture matches Benchmark-Model-xView2 SixChannelCNN.
    """

    def __init__(self, num_classes: int = len(DAMAGE_CLASSES), dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 6->32
            nn.Conv2d(6, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32->64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64->128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 128->256
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# Alias kept for backward compatibility (test_cnn_only.py uses DisasterCNN)
DisasterCNN = SixChannelCNN
