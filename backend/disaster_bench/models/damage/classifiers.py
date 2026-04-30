"""
Additional damage classifiers — Ref §2.1 Supervised (non-LLM) baselines.

Models:
  PrePostDiffCNN  — 9-channel [preRGB | postRGB | |post-pre|] -> 4-class
  SiameseCNN      — separate encoders for pre/post, feature fusion -> 4-class
  CentroidPatchCNN— same architecture as SixChannelCNN but accepts raw pre+post crops

All share a common _ConvEncoder backbone for consistency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared encoder backbone (re-used by all classifiers)
# ---------------------------------------------------------------------------

def _make_encoder(in_ch: int, base: int = 32) -> nn.Sequential:
    """4-block conv encoder: in_ch -> base*8 channels, global avg pool."""
    b = base
    return nn.Sequential(
        nn.Conv2d(in_ch, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(b, b*2, 3, padding=1), nn.BatchNorm2d(b*2), nn.ReLU(inplace=True),
        nn.Conv2d(b*2, b*2, 3, padding=1), nn.BatchNorm2d(b*2), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(b*2, b*4, 3, padding=1), nn.BatchNorm2d(b*4), nn.ReLU(inplace=True),
        nn.Conv2d(b*4, b*4, 3, padding=1), nn.BatchNorm2d(b*4), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(b*4, b*8, 3, padding=1), nn.BatchNorm2d(b*8), nn.ReLU(inplace=True),
        nn.Conv2d(b*8, b*8, 3, padding=1), nn.BatchNorm2d(b*8), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def _make_head(feat_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(feat_dim, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
    )


# ---------------------------------------------------------------------------
# Pre/Post/Diff — 9 channels: pre RGB | post RGB | |post-pre| RGB
# Ref §2.1: "Pre/Post/Diff classifier"
# ---------------------------------------------------------------------------

class PrePostDiffCNN(nn.Module):
    """
    9-channel input: [preRGB || postRGB || |postRGB - preRGB|] -> 4-class damage.
    The explicit difference channel forces the model to attend to change regions.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.4, base: int = 32) -> None:
        super().__init__()
        self.encoder = _make_encoder(9, base=base)
        self.head     = _make_head(base * 8, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, H, W)  channels: pre(0:3) | post(3:6) | diff(6:9)"""
        return self.head(self.encoder(x))

    @staticmethod
    def from_six_channel(x6: torch.Tensor) -> torch.Tensor:
        """Build 9-channel tensor from 6-channel (pre+post) input."""
        pre  = x6[:, :3]
        post = x6[:, 3:]
        diff = torch.abs(post - pre)
        return torch.cat([pre, post, diff], dim=1)


# ---------------------------------------------------------------------------
# Siamese — encode pre and post separately, fuse features -> 4-class
# Ref §2.1: "Siamese classifier: encode pre and post separately → fuse → 4-class"
# ---------------------------------------------------------------------------

class SiameseCNN(nn.Module):
    """
    Dual-stream: shared CNN encodes pre crop and post crop independently,
    features are concatenated then classified.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.4, base: int = 32) -> None:
        super().__init__()
        # Shared-weight encoder: 3 channels per stream
        self.stream = _make_encoder(3, base=base)
        feat_dim = base * 8 * 2  # pre feat + post feat concatenated
        self.head = _make_head(feat_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W)  channels: pre(0:3) | post(3:6)"""
        pre_feat  = self.stream(x[:, :3])
        post_feat = self.stream(x[:, 3:])
        fused = torch.cat([pre_feat, post_feat], dim=1)
        return self.head(fused)


# ---------------------------------------------------------------------------
# Centroid-patch — fixed square crop around polygon centroid from full image
# Same architecture as 6-channel but dataset feeds centroid crops.
# Ref §2.1: "Centroid-patch classifier: centroid from polygon/mask → fixed patch"
# ---------------------------------------------------------------------------

class CentroidPatchCNN(nn.Module):
    """
    Identical architecture to SixChannelCNN.
    The 'centroid-patch' distinction is in the *dataset*, not the network:
    inputs are larger fixed-size patches centered on polygon centroids rather
    than tight bbox crops.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.4, base: int = 32) -> None:
        super().__init__()
        self.encoder = _make_encoder(6, base=base)
        self.head     = _make_head(base * 8, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W)  channels: pre(0:3) | post(3:6) from centroid patch"""
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Multi-head cascade model — shared backbone + 3 task-specific heads
# Used with --cascade_mode multihead + --threshold_policy cascade_threshold
# ---------------------------------------------------------------------------

class MultiHeadCNN(nn.Module):
    """
    Shared-backbone cascade classifier for building damage assessment.

    Heads:
      head_damage:   binary logit  — any-damage (minor+major+destroyed) vs no-damage
      head_severe:   binary logit  — severe (major+destroyed) vs rest
      head_severity: 3-class logits — minor(0) / major(1) / destroyed(2)
                     trained only on damaged samples (damage_target==1)

    Input: 6 channels [preRGB || postRGB] or 9 channels [pre||post||diff]
    (determined by cascade_mode backbone; pass in_ch=9 for pre_post_diff backbone).
    """
    def __init__(self, in_ch: int = 6, dropout: float = 0.4, base: int = 32) -> None:
        super().__init__()
        self.encoder       = _make_encoder(in_ch, base=base)
        feat_dim           = base * 8
        self.head_damage   = _make_head(feat_dim, 1, dropout)
        self.head_severe   = _make_head(feat_dim, 1, dropout)
        self.head_severity = _make_head(feat_dim, 3, dropout)  # minor/major/destroyed

    def forward(self, x: torch.Tensor):
        """
        Returns:
          damage_logit:    (B,)   — binary logit for any-damage task
          severe_logit:    (B,)   — binary logit for severe task
          severity_logits: (B, 3) — logits for minor(0)/major(1)/destroyed(2)
        """
        feat = self.encoder(x)
        return (
            self.head_damage(feat).squeeze(-1),    # (B,)
            self.head_severe(feat).squeeze(-1),    # (B,)
            self.head_severity(feat),              # (B, 3)
        )


def build_multihead_classifier(
    model_type: str = "six_channel",
    dropout: float = 0.4,
) -> MultiHeadCNN:
    """Build a MultiHeadCNN cascade classifier. Input channels match model_type backbone."""
    in_ch = 9 if model_type == "pre_post_diff" else 6
    return MultiHeadCNN(in_ch=in_ch, dropout=dropout)


# ---------------------------------------------------------------------------
# ResNet-18 with pretrained ImageNet weights — 6-channel input
# ---------------------------------------------------------------------------

class ResNet18SixChannel(nn.Module):
    """
    ImageNet-pretrained ResNet-18 adapted for 6-channel (pre+post RGB) input.

    First conv is replaced with a 6-channel version; weights initialized by
    copying pretrained 3-channel weights into both halves scaled by 0.5.
    All other backbone layers keep their pretrained ImageNet weights.

    Exposes self.encoder (backbone) and self.head (FC) so that the existing
    --encoder_lr_scale and --freeze_epochs flags in train_damage.py apply
    automatically for proper transfer learning.

    Recommended fine-tune recipe:
      --freeze_epochs 5 --encoder_lr_scale 0.1 --warmup_epochs 3
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)

        # Replace conv1: (64, 3, 7, 7) -> (64, 6, 7, 7)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight * 0.5
            new_conv.weight[:, 3:] = old_conv.weight * 0.5
        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()  # strip original head; features pass through

        self.encoder = backbone          # backbone exposed for differential LR
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type] = {
    "six_channel":         None,               # loaded from six_channel.py
    "pre_post_diff":       PrePostDiffCNN,
    "siamese":             SiameseCNN,
    "siamese_convnextv2":  None,               # loaded from siamese_convnext.py
    "centroid_patch":      CentroidPatchCNN,
    "vit_finetune":        None,               # loaded from models.mae.vit_classifier
    "resnet18_finetune":   ResNet18SixChannel,
}

MODEL_TYPES = list(_MODEL_REGISTRY.keys())


def build_classifier(
    model_type: str,
    num_classes: int = 4,
    dropout: float = 0.4,
) -> nn.Module:
    """Build any damage classifier by name."""
    if model_type == "six_channel":
        from disaster_bench.models.damage.six_channel import build_model
        return build_model(num_classes=num_classes, dropout=dropout, light=False)
    if model_type == "vit_finetune":
        from disaster_bench.models.mae.vit_classifier import ViTDamageClassifier
        return ViTDamageClassifier(num_classes=num_classes, dropout=dropout)
    if model_type == "siamese_convnextv2":
        from disaster_bench.models.damage.siamese_convnext import SiameseConvNeXtV2
        return SiameseConvNeXtV2(num_classes=num_classes, dropout=dropout)
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {MODEL_TYPES}")
    return cls(num_classes=num_classes, dropout=dropout)


def save_checkpoint(
    model: nn.Module,
    path: str,
    model_type: str,
    epoch: int,
    val_macro_f1: float,
    per_class_f1: dict,
    input_size: int,
    num_classes: int = 4,
) -> None:
    import torch
    torch.save({
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_macro_f1": val_macro_f1,
        "per_class_f1": per_class_f1,
        "num_classes": num_classes,
        "input_size": input_size,
    }, path)


def load_classifier(path: str, device: str = "cpu") -> tuple[nn.Module, str, int]:
    """Load any classifier checkpoint. Returns (model, model_type, input_size)."""
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "six_channel")
    num_classes = ckpt.get("num_classes", 4)
    input_size  = ckpt.get("input_size", 128)
    model = build_classifier(model_type, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), model_type, input_size
