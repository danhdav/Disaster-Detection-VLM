"""
Temporal MAE (Masked Autoencoder) for paired pre/post building crops.

Architecture:
  Encoder: ViT-Small (embed_dim=384, depth=12, num_heads=6)
  Decoder: Lightweight ViT (embed_dim=256, depth=4, num_heads=8)

Key design choices:
  - Time embeddings (t=0 for pre, t=1 for post) distinguish the two frames
  - Independent random masking per frame (mask_ratio default 0.75)
  - Encoder sees ALL tokens concatenated: [vis_pre_tokens ; vis_post_tokens]
  - Decoder reconstructs masked patches in both frames
  - Loss: MSE on masked patches only (normalized per patch)

Ref: prompt.md §B Temporal MAE (Format A: two-frame tokens)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Patchify a single-frame RGB image into patch tokens."""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)           # (B, D, H/P, W/P)
        x = x.flatten(2)           # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Two-layer feed-forward network with GELU."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    """Pre-norm ViT transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TemporalMAEEncoder(nn.Module):
    """
    ViT encoder for paired pre/post frames with time embeddings.

    During pretraining: called with mask_ratio > 0 (random masking per frame).
    During fine-tuning: called via forward_features() with no masking.
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable positional embedding (one per patch location, shared across frames)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # Time embedding: t=0 for pre, t=1 for post
        self.time_embed = nn.Embedding(2, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Sinusoidal init for pos_embed
        num_patches = self.pos_embed.shape[1]
        dim = self.pos_embed.shape[2]
        pos = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
        pe = torch.zeros(num_patches, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:dim // 2])
        self.pos_embed.data.copy_(pe.unsqueeze(0))

        # Time embedding: small random init
        nn.init.normal_(self.time_embed.weight, std=0.02)

        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def random_masking(
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking: keep (1 - mask_ratio) fraction of tokens.

        Returns:
            x_visible: (B, N_keep, D) — visible tokens
            mask:      (B, N)         — 1=masked, 0=visible (original order)
            ids_restore:(B, N)        — permutation to restore original order
        """
        B, N, D = x.shape
        N_keep = max(1, int(N * (1.0 - mask_ratio)))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)          # (B, N)
        ids_restore = torch.argsort(ids_shuffle, dim=1)   # (B, N)

        # Keep first N_keep (lowest noise = most likely kept)
        ids_keep = ids_shuffle[:, :N_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Build mask in original token order (1=masked)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N_keep] = 0.0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        mask_ratio_pre: float = 0.75,
        mask_ratio_post: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Encode paired pre/post frames with independent masking.

        Args:
            pre:  (B, 3, H, W)
            post: (B, 3, H, W)
            mask_ratio_pre:  fraction of pre tokens to mask
            mask_ratio_post: fraction of post tokens to mask

        Returns:
            encoded:         (B, N_vis_pre + N_vis_post, D) encoder output
            mask_pre:        (B, N) — 1=masked (original order)
            ids_restore_pre: (B, N) — indices to restore original pre order
            mask_post:       (B, N) — 1=masked (original order)
            ids_restore_post:(B, N) — indices to restore original post order
            n_vis_pre:       int    — number of visible pre tokens
        """
        # Patch embed
        pre_tok = self.patch_embed(pre)    # (B, N, D)
        post_tok = self.patch_embed(post)  # (B, N, D)

        # Add positional + time embeddings
        pre_tok = pre_tok + self.pos_embed + self.time_embed.weight[0]
        post_tok = post_tok + self.pos_embed + self.time_embed.weight[1]

        # Independent masking per frame
        pre_tok, mask_pre, ids_restore_pre = self.random_masking(pre_tok, mask_ratio_pre)
        post_tok, mask_post, ids_restore_post = self.random_masking(post_tok, mask_ratio_post)
        n_vis_pre = pre_tok.shape[1]

        # Concatenate visible tokens and encode
        tokens = torch.cat([pre_tok, post_tok], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens, mask_pre, ids_restore_pre, mask_post, ids_restore_post, n_vis_pre

    def forward_features(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both frames without masking (used during fine-tuning).

        Returns:
            f_pre:  (B, D) — mean-pooled pre features
            f_post: (B, D) — mean-pooled post features
        """
        pre_tok = self.patch_embed(pre) + self.pos_embed + self.time_embed.weight[0]
        post_tok = self.patch_embed(post) + self.pos_embed + self.time_embed.weight[1]

        tokens = torch.cat([pre_tok, post_tok], dim=1)  # (B, 2N, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        N = pre_tok.shape[1]
        f_pre = tokens[:, :N].mean(dim=1)   # (B, D)
        f_post = tokens[:, N:].mean(dim=1)  # (B, D)
        return f_pre, f_post


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TemporalMAEDecoder(nn.Module):
    """Lightweight decoder for reconstruction of masked patches."""

    def __init__(
        self,
        num_patches: int,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_embed_dim: int = 384,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Project encoder dim → decoder dim
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learnable mask token (stands in for masked patches in decoder input)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional + time embeddings (separate from encoder's)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        self.decoder_time_embed = nn.Embedding(2, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Predict P*P*C pixel values per patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.zeros_(self.decoder_pos_embed)
        nn.init.normal_(self.decoder_time_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _restore_and_add_embeddings(
        self,
        vis_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
        time_idx: int,
    ) -> torch.Tensor:
        """
        Restore masked positions with mask tokens, then add pos+time embeddings.

        Args:
            vis_tokens: (B, N_vis, D_dec) — projected visible tokens
            ids_restore:(B, N)            — indices to restore original order
            time_idx:   0=pre, 1=post

        Returns:
            full_tokens: (B, N, D_dec)
        """
        B, N_vis, D = vis_tokens.shape
        N = ids_restore.shape[1]
        N_mask = N - N_vis

        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        # Concatenate vis + mask in shuffled order, then restore original order
        x = torch.cat([vis_tokens, mask_tokens], dim=1)  # (B, N, D)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))

        # Add decoder positional + time embeddings
        x = x + self.decoder_pos_embed + self.decoder_time_embed.weight[time_idx]
        return x

    def forward(
        self,
        encoded: torch.Tensor,
        ids_restore_pre: torch.Tensor,
        ids_restore_post: torch.Tensor,
        n_vis_pre: int,
    ) -> torch.Tensor:
        """
        Decode encoder output back to patch pixel values.

        Args:
            encoded:          (B, N_vis_pre + N_vis_post, D_enc) encoder output
            ids_restore_pre:  (B, N) restore indices for pre frame
            ids_restore_post: (B, N) restore indices for post frame
            n_vis_pre:        int — how many encoder tokens belong to pre

        Returns:
            pred: (B, 2*N, P*P*C) predicted pixel values for all patches (pre then post)
        """
        # Project to decoder dimension
        x = self.decoder_embed(encoded)  # (B, N_vis_pre + N_vis_post, D_dec)

        # Split into pre and post visible tokens
        x_pre_vis = x[:, :n_vis_pre]
        x_post_vis = x[:, n_vis_pre:]

        # Restore both frames
        x_pre = self._restore_and_add_embeddings(x_pre_vis, ids_restore_pre, time_idx=0)
        x_post = self._restore_and_add_embeddings(x_post_vis, ids_restore_post, time_idx=1)

        # Concatenate and decode
        x_all = torch.cat([x_pre, x_post], dim=1)  # (B, 2N, D_dec)
        for blk in self.decoder_blocks:
            x_all = blk(x_all)
        x_all = self.decoder_norm(x_all)

        pred = self.decoder_pred(x_all)  # (B, 2N, P*P*C)
        return pred


# ---------------------------------------------------------------------------
# Full Temporal MAE
# ---------------------------------------------------------------------------

class TemporalMAE(nn.Module):
    """
    Full Temporal MAE: encoder + decoder with reconstruction loss.

    Usage:
        model = build_temporal_mae('vit_small')
        loss = model(pre, post)                    # pretraining
        f_pre, f_post = model.encoder.forward_features(pre, post)  # fine-tuning
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 12,
        encoder_num_heads: int = 6,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio_pre: float = 0.75,
        mask_ratio_post: float = 0.75,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio_pre = mask_ratio_pre
        self.mask_ratio_post = mask_ratio_post
        self.norm_pix_loss = norm_pix_loss

        self.encoder = TemporalMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
        )
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder = TemporalMAEDecoder(
            num_patches=num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, C, H, W) → patches: (B, N, P*P*C)
        """
        P = self.patch_size
        C = imgs.shape[1]
        H, W = imgs.shape[2], imgs.shape[3]
        h, w = H // P, W // P
        x = imgs.reshape(imgs.shape[0], C, h, P, w, P)
        x = x.permute(0, 2, 4, 3, 5, 1)   # (B, h, w, P, P, C)
        x = x.reshape(imgs.shape[0], h * w, P * P * C)
        return x

    def reconstruction_loss(
        self,
        pred: torch.Tensor,
        target_pre: torch.Tensor,
        target_post: torch.Tensor,
        mask_pre: torch.Tensor,
        mask_post: torch.Tensor,
        objective: str = "v1",
    ) -> torch.Tensor:
        """
        MSE loss on masked patches only.

        Objective v1 (default): reconstruct both pre and post frames equally.
        Objective v2 (change-aligned): reconstruct post only, pre is less masked
            and acts as conditioning context. Loss is only on masked post patches.

        Args:
            pred:       (B, 2N, P*P*C)
            target_pre: (B, N,  P*P*C)
            target_post:(B, N,  P*P*C)
            mask_pre:   (B, N)  — 1=masked
            mask_post:  (B, N)  — 1=masked
            objective:  "v1" or "v2"
        """
        N = target_pre.shape[1]
        pred_pre = pred[:, :N]
        pred_post = pred[:, N:]

        if self.norm_pix_loss:
            mean_pre = target_pre.mean(dim=-1, keepdim=True)
            var_pre = target_pre.var(dim=-1, keepdim=True)
            target_pre = (target_pre - mean_pre) / (var_pre + 1e-6).sqrt()

            mean_post = target_post.mean(dim=-1, keepdim=True)
            var_post = target_post.var(dim=-1, keepdim=True)
            target_post = (target_post - mean_post) / (var_post + 1e-6).sqrt()

        loss_post_raw = ((pred_post - target_post) ** 2).mean(dim=-1)  # (B, N)
        n_masked_post = mask_post.sum()
        loss_post = (loss_post_raw * mask_post).sum() / (n_masked_post + 1e-8)

        if objective == "v2":
            # Reconstruct post only — pre is context, not a reconstruction target
            return loss_post

        # v1: reconstruct both frames equally
        loss_pre_raw = ((pred_pre - target_pre) ** 2).mean(dim=-1)
        n_masked_pre = mask_pre.sum()
        loss_pre = (loss_pre_raw * mask_pre).sum() / (n_masked_pre + 1e-8)
        return loss_pre + loss_post

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        objective: str = "v1",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for pretraining.

        Args:
            pre:       (B, 3, H, W)
            post:      (B, 3, H, W)
            objective: "v1" (reconstruct both) or "v2" (reconstruct post only,
                       change-aligned — set mask_ratio_pre lower when using v2)

        Returns:
            loss:      scalar reconstruction loss
            mask_pre:  (B, N) binary mask for pre  (1=masked)
            mask_post: (B, N) binary mask for post (1=masked)
        """
        encoded, mask_pre, ids_restore_pre, mask_post, ids_restore_post, n_vis_pre = \
            self.encoder(pre, post, self.mask_ratio_pre, self.mask_ratio_post)

        pred = self.decoder(encoded, ids_restore_pre, ids_restore_post, n_vis_pre)

        target_pre = self.patchify(pre)
        target_post = self.patchify(post)

        loss = self.reconstruction_loss(pred, target_pre, target_post,
                                        mask_pre, mask_post, objective=objective)
        return loss, mask_pre, mask_post


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BACKBONE_CONFIGS = {
    "vit_small": dict(
        encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
    ),
    "vit_base": dict(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    ),
    "vit_tiny": dict(
        encoder_embed_dim=192, encoder_depth=12, encoder_num_heads=3,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
    ),
}


def build_temporal_mae(
    backbone: str = "vit_small",
    img_size: int = 128,
    patch_size: int = 16,
    mask_ratio_pre: float = 0.75,
    mask_ratio_post: float = 0.75,
    norm_pix_loss: bool = True,
) -> TemporalMAE:
    if backbone not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONE_CONFIGS)}")
    cfg = BACKBONE_CONFIGS[backbone]
    return TemporalMAE(
        img_size=img_size,
        patch_size=patch_size,
        mask_ratio_pre=mask_ratio_pre,
        mask_ratio_post=mask_ratio_post,
        norm_pix_loss=norm_pix_loss,
        **cfg,
    )
