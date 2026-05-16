"""
SSL pretraining dataset for Temporal MAE.

Returns (pre_crop, post_crop) pairs — no labels needed.
Reads from buildings_v2.csv so it covers all 8,316 buildings.
Supports filtering by disaster name for LOWO leakage prevention.

Augmentations (conservative per prompt.md §D):
  - Random H/V flips
  - Small rotations (multiples of 90° or small affine)
  - Mild brightness/contrast jitter (same params for pre & post)
  - Slight crop jitter to handle alignment noise

Ref: prompt.md §A.1, §D
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def _load_rgb(path: str | Path, size: int) -> np.ndarray:
    """Load an image, resize to (size, size), return float32 (3, H, W) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return arr.transpose(2, 0, 1)  # (3, H, W)


class MAECropDataset:
    """
    SSL pretraining dataset: returns (pre_crop, post_crop) pairs.

    Args:
        buildings_csv:    Path to data/processed/buildings_v2.csv
        size:             Crop size in pixels (default 128)
        train_disasters:  If given, only include buildings from these disasters
        augment:          Apply conservative augmentations
        preload:          Cache all images in RAM
        seed:             RNG seed for augmentation
    """

    def __init__(
        self,
        buildings_csv: str | Path,
        size: int = 128,
        train_disasters: list[str] | None = None,
        augment: bool = True,
        preload: bool = False,
        seed: int = 42,
    ) -> None:
        self.size = size
        self.augment = augment
        self._rng = random.Random(seed)

        self.records = self._load_records(buildings_csv, train_disasters)
        if not self.records:
            raise RuntimeError(
                f"No records found in {buildings_csv} "
                f"(train_disasters={train_disasters})"
            )

        self._cache: dict[int, tuple] = {}
        if preload:
            print(f"[MAECropDataset] Preloading {len(self.records)} crop pairs...")
            for i, rec in enumerate(self.records):
                self._cache[i] = (
                    _load_rgb(rec["pre_path"], size),
                    _load_rgb(rec["post_path"], size),
                )

    def _load_records(
        self,
        csv_path: str | Path,
        train_disasters: list[str] | None,
    ) -> list[dict]:
        records = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if train_disasters and row["disaster"] not in train_disasters:
                    continue
                records.append({
                    "pre_path": row["pre_path"],
                    "post_path": row["post_path"],
                    "disaster": row["disaster"],
                    "tile_id": row["tile_id"],
                    "building_id": row["building_id"],
                })
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pre, post) — each (3, H, W) float32 in [0, 1]."""
        if idx in self._cache:
            pre, post = self._cache[idx]
        else:
            rec = self.records[idx]
            pre = _load_rgb(rec["pre_path"], self.size)
            post = _load_rgb(rec["post_path"], self.size)

        if self.augment:
            pre, post = self._augment(pre, post)
        return pre, post

    def _augment(
        self,
        pre: np.ndarray,
        post: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply identical geometric transforms and correlated photometric jitter."""
        # --- Geometric transforms (same for both frames to preserve alignment) ---

        # Random horizontal flip
        if self._rng.random() < 0.5:
            pre = pre[:, :, ::-1].copy()
            post = post[:, :, ::-1].copy()

        # Random vertical flip
        if self._rng.random() < 0.5:
            pre = pre[:, ::-1, :].copy()
            post = post[:, ::-1, :].copy()

        # Random 90° rotation
        k = self._rng.randint(0, 3)
        if k > 0:
            pre = np.rot90(pre, k, axes=(1, 2)).copy()
            post = np.rot90(post, k, axes=(1, 2)).copy()

        # Slight crop jitter (handles alignment noise): random ±4px shift + re-pad
        jitter = 4
        if self._rng.random() < 0.5:
            H, W = pre.shape[1], pre.shape[2]
            dx = self._rng.randint(-jitter, jitter)
            dy = self._rng.randint(-jitter, jitter)
            # Shift pre independently (simulates alignment noise between acquisitions)
            pre = self._shift(pre, dx, dy)
            # Post shift is smaller (mostly same-pass registration)
            post = self._shift(post, dx // 2, dy // 2)

        # --- Photometric jitter (same magnitude for pre & post to avoid spurious change) ---
        if self._rng.random() < 0.5:
            brightness = self._rng.uniform(0.8, 1.2)
            contrast = self._rng.uniform(0.8, 1.2)
            pre = self._color_jitter(pre, brightness, contrast)
            # Apply similar but slightly varied params to post
            b2 = brightness * self._rng.uniform(0.95, 1.05)
            c2 = contrast * self._rng.uniform(0.95, 1.05)
            post = self._color_jitter(post, b2, c2)

        return pre, post

    @staticmethod
    def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Shift image by (dx, dy) pixels, padding with zeros."""
        _, H, W = img.shape
        out = np.zeros_like(img)
        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(W, W + dx)
        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(H, H + dy)
        if src_x1 > src_x0 and src_y1 > src_y0:
            out[:, dst_y0:dst_y1, dst_x0:dst_x1] = img[:, src_y0:src_y1, src_x0:src_x1]
        return out

    @staticmethod
    def _color_jitter(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        """Apply brightness and contrast jitter in-place (clamp to [0,1])."""
        img = img * brightness
        mean = img.mean()
        img = (img - mean) * contrast + mean
        return np.clip(img, 0.0, 1.0)
