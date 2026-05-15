"""
ViT-based 4-class damage classifier using a pretrained TemporalMAE encoder.

Input: (B, 6, H, W) = [preRGB || postRGB]  (same format as SixChannelCNN)
Architecture:
  1. Split into pre (B, 3, H, W) and post (B, 3, H, W)
  2. Encode both frames together through shared TemporalMAEEncoder
     (cross-temporal attention is preserved, same as pretraining)
  3. Mean-pool per frame:  f_pre, f_post ∈ (B, D)
  4. Fuse: z = concat(f_pre, f_post, |f_post - f_pre|, f_pre * f_post) → (B, 4D)
  5. Classification head → 4-class logits

The model's `encoder` attribute is a TemporalMAEEncoder, so the existing
_load_encoder_weights() helper in train_damage.py loads MAE checkpoint weights
correctly by stripping the "encoder." prefix.

Ref: prompt.md §Build §4 Fine-tuning (4-class)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from disaster_bench.models.mae.temporal_mae import TemporalMAEEncoder, BACKBONE_CONFIGS


class ViTDamageClassifier(nn.Module):
    """
    4-class damage classifier built on a TemporalMAEEncoder backbone.

    Compatible with the existing train_damage.py training loop:
    - Takes (B, 6, H, W) tensors (preRGB concatenated with postRGB)
    - Returns (B, num_classes) logits
    - Has a `.encoder` attribute for pretrained weight loading
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.4,
        backbone: str = "vit_small",
        img_size: int = 128,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        cfg = BACKBONE_CONFIGS[backbone]
        embed_dim = cfg["encoder_embed_dim"]

        self.encoder = TemporalMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=cfg["encoder_depth"],
            num_heads=cfg["encoder_num_heads"],
        )

        # Fusion: concat(f_pre, f_post, |f_post-f_pre|, f_pre*f_post) → 4*D
        fuse_dim = embed_dim * 4
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) — first 3 channels = pre, last 3 = post

        Returns:
            logits: (B, num_classes)
        """
        pre = x[:, :3]   # (B, 3, H, W)
        post = x[:, 3:]  # (B, 3, H, W)

        f_pre, f_post = self.encoder.forward_features(pre, post)

        diff = (f_post - f_pre).abs()
        prod = f_pre * f_post
        z = torch.cat([f_pre, f_post, diff, prod], dim=1)  # (B, 4*D)

        return self.head(z)
