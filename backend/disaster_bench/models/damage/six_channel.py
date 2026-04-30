"""
6-channel classifier: [preRGB || postRGB] -> 4-class damage.
Ref §2.1 Supervised (non-LLM) baselines — simplest supervised model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SixChannelCNN(nn.Module):
    """
    Simple CNN on 6-channel (pre+post RGB) 128x128 input -> 4 damage classes.
    Architecture: 4 conv blocks + global avg pool + MLP head.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 6->32, 128x128 -> 64x64
            nn.Conv2d(6, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32->64, 64x64 -> 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64->128, 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 128->256, 16x16 -> 8x8
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

    def forward_features(self, x: torch.Tensor) -> tuple:
        """Return (logits, features) where features is the 128-dim embedding after ReLU."""
        enc = self.encoder(x)
        feats  = self.head[4](self.head[3](self.head[2](self.head[1](self.head[0](enc)))))
        logits = self.head[6](self.head[5](feats))
        return logits, feats


class SixChannelCNNLight(nn.Module):
    """
    Lightweight 6-channel CNN for CPU training (2 conv blocks + global pool + MLP).
    Faster than SixChannelCNN while still learning discriminative features.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 6->64, stride 2
            nn.Conv2d(6, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 64->128, stride 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, groups=128), nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def forward_features(self, x: torch.Tensor) -> tuple:
        """Return (logits, features) where features is the 64-dim embedding after ReLU."""
        enc = self.encoder(x)
        feats  = self.head[4](self.head[3](self.head[2](self.head[1](self.head[0](enc)))))
        logits = self.head[5](feats)
        return logits, feats


def build_model(num_classes: int = 4, dropout: float = 0.4, light: bool = True) -> nn.Module:
    if light:
        return SixChannelCNNLight(num_classes=num_classes, dropout=dropout)
    return SixChannelCNN(num_classes=num_classes, dropout=dropout)


def load_checkpoint(path: str, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    num_classes = ckpt.get("num_classes", 4)
    # Detect architecture from first conv weight shape
    first_key = next(k for k in state if "encoder.0.weight" in k)
    first_out_ch = state[first_key].shape[0]  # 32=full, 64=light
    use_light = (first_out_ch == 64)
    model = build_model(num_classes=num_classes, light=use_light)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def predict_crop(
    model: SixChannelCNN,
    x: "np.ndarray",  # (6, H, W) float32 [0,1]
    device: str = "cpu",
) -> tuple[str, float]:
    """Return (damage_class, confidence) for a single 6-channel crop array."""
    import torch
    import numpy as np
    from disaster_bench.data.dataset import DAMAGE_CLASSES
    t = torch.from_numpy(x).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return DAMAGE_CLASSES[idx], float(probs[idx])
