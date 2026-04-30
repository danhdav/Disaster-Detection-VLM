"""
Siamese ConvNeXtV2-Femto — pretrained backbone, feature-level fusion.

Architecture:
  - Shared ConvNeXtV2-Femto backbone (FCMAE pretrained, 4.85M params)
  - Same backbone runs twice: once on pre-image, once on post-image
  - Fusion: [f_pre || f_post || |f_pre - f_post|]  (3 × 384 = 1152-d)
  - Head: LayerNorm(1152) → Linear(1152, 384) → GELU → Dropout → Linear(384, 4)

Normalization:
  Dataset yields raw [0, 1] pixels. ImageNet mean/std applied inside forward()
  so existing CropDataset and transforms stay unchanged.

head[4] = final Linear(384, num_classes) — compatible with _apply_tau_norm()
in export_cnn_probs.py (same index convention as _make_head()).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SiameseConvNeXtV2(nn.Module):
    """
    Siamese pretrained ConvNeXtV2-Femto for 4-class building damage.

    Input: (B, 6, H, W) — [preRGB || postRGB] from CropDataset, values in [0, 1].
    Output: (B, 4) logits.

    freeze_backbone() / unfreeze_backbone() support the LP-FT-cRT training recipe.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for SiameseConvNeXtV2. "
                "Install it with: pip install timm"
            ) from e

        self.backbone = timm.create_model(
            "convnextv2_femto.fcmae_ft_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim  = self.backbone.num_features  # 384
        fused_dim = feat_dim * 3                # 1152

        # head[4] = final classifier — same index convention as _make_head() so
        # export_cnn_probs.py _apply_tau_norm (targets head[4]) works unchanged.
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),            # 0
            nn.Linear(fused_dim, feat_dim),     # 1
            nn.GELU(),                          # 2
            nn.Dropout(dropout),                # 3
            nn.Linear(feat_dim, num_classes),   # 4  ← tau-norm target
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        # ImageNet normalization constants (buffers → move with .to(device))
        self.register_buffer(
            "_imgnet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imgnet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet mean/std to a 3-channel [0, 1] tensor."""
        return (x - self._imgnet_mean) / self._imgnet_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, H, W)  channels: pre(0:3) | post(3:6)
        Accepts 9-channel input (pre+post+diff) — only first 6 channels used.
        """
        pre  = self._normalize(x[:, :3])   # (B, 3, H, W)
        post = self._normalize(x[:, 3:6])  # (B, 3, H, W)
        f_pre  = self.backbone(pre)         # (B, 384)
        f_post = self.backbone(post)        # (B, 384)
        f_diff = (f_pre - f_post).abs()     # (B, 384)
        fused  = torch.cat([f_pre, f_post, f_diff], dim=1)  # (B, 1152)
        return self.head(fused)             # (B, 4)

    # ------------------------------------------------------------------
    # LP-FT-cRT helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze backbone weights and set to eval mode (preserves LayerNorm stats)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()

    def reinit_classifier(self) -> None:
        """Re-initialize the final classifier layer (used before cRT stage)."""
        layer = self.head[-1]
        nn.init.trunc_normal_(layer.weight, std=0.01)
        nn.init.zeros_(layer.bias)
