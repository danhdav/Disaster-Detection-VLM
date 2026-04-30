"""
Siamese U-Net for pixel-wise building damage severity.
Ref §2B learned models: "Siamese U-Net pixel-wise damage severity {0..4}"
                        "Post-only pixel-wise damage segmentation {0..4}"
                        "Polygon aggregation from a learned damage map"

Architecture:
  - Dual-stream U-Net encoder (pre and post images share weights)
  - Difference feature at each skip connection level
  - Decoder outputs per-pixel severity map (5-class: 0=background, 1-4=damage levels)
  - OR post-only mode (single stream, for ablation)

Pixel labels (xView2 standard):
  0 = background / non-building
  1 = no-damage
  2 = minor-damage
  3 = major-damage
  4 = destroyed
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SEVERITY_CLASSES = 5          # 0=bg, 1-4=damage
DAMAGE_SEVERITY  = [          # class-index → damage label
    "background", "no-damage", "minor-damage", "major-damage", "destroyed"
]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _double_conv(in_ch: int, out_ch: int) -> "nn.Sequential":
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Siamese U-Net
# ---------------------------------------------------------------------------

def build_siamese_unet(
    num_classes: int = SEVERITY_CLASSES,
    base: int = 32,
    post_only: bool = False,
) -> "nn.Module":
    """
    Build Siamese U-Net.
    post_only=True → single-stream (only post image), for ablation study.
    Input channels per stream: 3 (RGB).
    Skip connection fusion: concatenate pre+post features, then 1×1 conv to halve channels.
    """
    import torch
    import torch.nn as nn

    pool = nn.MaxPool2d(2)
    b    = base

    class SiameseUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_only = post_only
            # Shared encoder (weights tied for pre/post if not post_only)
            self.enc1 = _double_conv(3, b)
            self.enc2 = _double_conv(b, b*2)
            self.enc3 = _double_conv(b*2, b*4)
            self.enc4 = _double_conv(b*4, b*8)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _double_conv(b*8, b*16)

            # Fusion 1x1 convs (halve merged skip channels)
            if not post_only:
                self.fuse4 = nn.Conv2d(b*8*2,  b*8,  1)
                self.fuse3 = nn.Conv2d(b*4*2,  b*4,  1)
                self.fuse2 = nn.Conv2d(b*2*2,  b*2,  1)
                self.fuse1 = nn.Conv2d(b*2,    b,    1)  # b + b → b (post_only has b enc)
                btn_in = b*16 * 2
            else:
                btn_in = b*16

            # Decoder
            self.up4  = nn.ConvTranspose2d(btn_in, b*8,  2, stride=2)
            self.dec4 = _double_conv(b*8 + b*8,  b*8)
            self.up3  = nn.ConvTranspose2d(b*8,   b*4,  2, stride=2)
            self.dec3 = _double_conv(b*4 + b*4,  b*4)
            self.up2  = nn.ConvTranspose2d(b*4,   b*2,  2, stride=2)
            self.dec2 = _double_conv(b*2 + b*2,  b*2)
            self.up1  = nn.ConvTranspose2d(b*2,   b,    2, stride=2)
            self.dec1 = _double_conv(b + b,       b)
            self.out  = nn.Conv2d(b, num_classes,  1)

        def _encode(self, x: torch.Tensor):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            bn = self.bottleneck(self.pool(e4))
            return e1, e2, e3, e4, bn

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, 6, H, W) — pre(0:3) | post(3:6)"""
            if self.post_only:
                post = x[:, 3:]
                e1, e2, e3, e4, bn = self._encode(post)
                s4, s3, s2, s1 = e4, e3, e2, e1
            else:
                pre  = x[:, :3]
                post = x[:, 3:]
                p1, p2, p3, p4, pbn = self._encode(pre)
                e1, e2, e3, e4, ebn = self._encode(post)
                bn  = torch.cat([pbn, ebn], dim=1)
                # Fuse skip: concat + 1x1
                s4 = self.fuse4(torch.cat([p4, e4], dim=1))
                s3 = self.fuse3(torch.cat([p3, e3], dim=1))
                s2 = self.fuse2(torch.cat([p2, e2], dim=1))
                s1 = self.fuse1(torch.cat([p1, e1], dim=1))

            d4 = self.dec4(torch.cat([self.up4(bn), s4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
            return self.out(d1)   # (B, num_classes, H, W) logits

    return SiameseUNet()


# ---------------------------------------------------------------------------
# Polygon aggregation from learned damage map
# ---------------------------------------------------------------------------

def aggregate_damage_from_map(
    severity_logits: np.ndarray,
    mask: np.ndarray,
    mode: str = "mean_softmax",
) -> tuple[str, float]:
    """
    Aggregate pixel-wise severity predictions inside a building polygon mask.
    severity_logits: (num_classes, H, W) or (H, W) for argmax map.
    mask: (H, W) binary mask for this building.
    Returns (damage_label, confidence).
    Ref §2B: "Polygon aggregation from a learned damage map".
    """
    from scipy.special import softmax as scipy_softmax

    pixels = mask > 0
    if not pixels.any():
        return "no-damage", 0.0

    if severity_logits.ndim == 3:
        # (C, H, W) — aggregate per-class logits inside mask, then softmax
        pix_logits = severity_logits[:, pixels]          # (C, N_px)
        mean_logits = pix_logits.mean(axis=1)            # (C,)
        probs = scipy_softmax(mean_logits.astype(np.float64)).astype(np.float32)
        pred_idx = int(np.argmax(probs[1:]) + 1)         # exclude bg class 0
        conf = float(probs[pred_idx])
    else:
        # (H, W) argmax map
        classes, counts = np.unique(severity_logits[pixels], return_counts=True)
        best_cls = int(classes[np.argmax(counts)])
        pred_idx = best_cls
        conf = float(counts.max()) / float(counts.sum())

    label = DAMAGE_SEVERITY[min(pred_idx, len(DAMAGE_SEVERITY) - 1)]
    if label == "background":
        label = "no-damage"
    return label, conf


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_siamese_unet_checkpoint(
    model: "nn.Module",
    path: str | Path,
    post_only: bool,
    epoch: int,
    val_miou: float,
) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type":        "siamese_unet",
        "post_only":         post_only,
        "num_classes":       SEVERITY_CLASSES,
        "model_state_dict":  model.state_dict(),
        "epoch":             epoch,
        "val_miou":          val_miou,
    }, path)


def load_siamese_unet(path: str | Path, device: str = "cpu") -> tuple["nn.Module", bool]:
    """Returns (model, post_only)."""
    import torch
    ckpt      = torch.load(path, map_location=device, weights_only=False)
    post_only = ckpt.get("post_only", False)
    n_cls     = ckpt.get("num_classes", SEVERITY_CLASSES)
    model     = build_siamese_unet(num_classes=n_cls, post_only=post_only)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), post_only


# ---------------------------------------------------------------------------
# Pixel-wise dataset (tile-level, returns (pre_post_6ch, label_mask) pairs)
# ---------------------------------------------------------------------------

class PixelDamageDataset:
    """
    Dataset for pixel-wise damage segmentation training.
    Each item: (6, H, W) float32 pre+post crop, (H, W) int64 severity label.
    Severity labels: 0=bg, 1=no-damage, 2=minor, 3=major, 4=destroyed.

    Reads from oracle crops + rasterized polygon masks.
    For each building crop, the mask is the polygon filled with the damage severity.
    """
    def __init__(
        self,
        records: list[dict[str, Any]],
        size: int = 256,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.size    = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        import random
        from PIL import Image
        from disaster_bench.data.dataset import LABEL2IDX
        r = self.records[idx]
        sev = r.get("severity", 0)  # 0-4

        def _load(p: str) -> np.ndarray:
            with Image.open(p) as im:
                return np.array(im.convert("RGB").resize(
                    (self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0

        pre  = _load(r["pre_path"])   # (H,W,3)
        post = _load(r["post_path"])  # (H,W,3)
        x    = np.concatenate([pre, post], axis=2).transpose(2, 0, 1)  # (6,H,W)

        # Label map: building pixels = severity class, background = 0
        # For bbox crops, the entire crop is the building
        lbl_map = np.full((self.size, self.size), sev, dtype=np.int64)

        if self.augment:
            if random.random() > 0.5:
                x       = x[:, :, ::-1].copy()
                lbl_map = lbl_map[:, ::-1].copy()
            if random.random() > 0.5:
                x       = x[:, ::-1, :].copy()
                lbl_map = lbl_map[::-1, :].copy()

        return x, lbl_map


def build_pixel_records(
    crop_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert CropDataset records (label 0-3) to pixel-dataset records (severity 1-4).
    damage label_idx 0-3 → severity 1-4 (class 0 is background).
    """
    from disaster_bench.data.dataset import DAMAGE_CLASSES
    out = []
    for r in crop_records:
        out.append({
            **r,
            "severity": r["label_idx"] + 1,  # 1=no-damage ... 4=destroyed
        })
    return out
