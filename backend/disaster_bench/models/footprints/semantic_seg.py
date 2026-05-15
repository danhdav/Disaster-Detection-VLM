"""
Building footprint detector — semantic segmentation.
Ref §A2 footprint stage: "YOLO boxes / instance seg / semantic seg → instances"
Ref §05 §A2: Footprint detector candidates — U-Net / DeepLabV3+ / SegFormer.

Implementation: lightweight encoder-decoder U-Net (single RGB image → binary building mask).
Works with any pretrained ImageNet encoder via torchvision or from scratch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight U-Net for semantic building segmentation
# ---------------------------------------------------------------------------

class ConvBlock(object):
    pass  # defined below with torch imports inside functions to avoid hard dep at import time


def build_unet(in_ch: int = 3, out_ch: int = 1, base: int = 32) -> "nn.Module":
    """
    Build a lightweight U-Net for building segmentation.
    Input: (B, 3, H, W)  Output: (B, 1, H, W) logits (sigmoid for probability).
    """
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, c_in: int, c_out: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.block(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            b = base
            self.enc1 = DoubleConv(in_ch, b)
            self.enc2 = DoubleConv(b, b*2)
            self.enc3 = DoubleConv(b*2, b*4)
            self.enc4 = DoubleConv(b*4, b*8)
            self.pool = nn.MaxPool2d(2)

            self.bottleneck = DoubleConv(b*8, b*16)

            self.up4  = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
            self.dec4 = DoubleConv(b*16, b*8)
            self.up3  = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
            self.dec3 = DoubleConv(b*8, b*4)
            self.up2  = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
            self.dec2 = DoubleConv(b*4, b*2)
            self.up1  = nn.ConvTranspose2d(b*2, b, 2, stride=2)
            self.dec1 = DoubleConv(b*2, b)

            self.out_conv = nn.Conv2d(b, out_ch, 1)

        def forward(self, x):
            import torch
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            bn = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out_conv(d1)

    return UNet()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_mask(
    model: "nn.Module",
    image: np.ndarray,
    device: str = "cpu",
    threshold: float = 0.5,
    tile_size: int | None = 512,
) -> np.ndarray:
    """
    Run semantic segmentation on a (H, W, 3) uint8 image.
    Returns binary mask (H, W) uint8.
    If tile_size is set, runs in overlapping tiles for large images.
    """
    import torch
    import torch.nn.functional as F

    h, w = image.shape[:2]
    model.eval()

    def _run_patch(patch_rgb: np.ndarray) -> np.ndarray:
        ph, pw = patch_rgb.shape[:2]
        # Pad to multiples of 32 (U-Net stride requirement)
        pad_h = (32 - ph % 32) % 32
        pad_w = (32 - pw % 32) % 32
        padded = np.pad(patch_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        t = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1) / 255.0)
        t = t.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t)[0, 0]        # (H', W')
            prob   = torch.sigmoid(logits).cpu().numpy()
        return (prob[:ph, :pw] > threshold).astype(np.uint8)

    if tile_size is None or (h <= tile_size and w <= tile_size):
        return _run_patch(image)

    # Sliding window with 64-pixel overlap
    overlap  = 64
    step     = tile_size - overlap
    out_mask = np.zeros((h, w), dtype=np.float32)
    count    = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y, h - 1); y2 = min(y + tile_size, h)
            x1 = min(x, w - 1); x2 = min(x + tile_size, w)
            patch = image[y1:y2, x1:x2]
            prob  = _run_patch(patch).astype(np.float32)
            out_mask[y1:y2, x1:x2] += prob
            count[y1:y2, x1:x2]    += 1.0

    return (out_mask / np.maximum(count, 1.0) > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Instance extraction from binary mask
# ---------------------------------------------------------------------------

def mask_to_instances(
    binary_mask: np.ndarray,
    min_area: int = 50,
    max_area: int = 500_000,
) -> list[dict[str, Any]]:
    """
    Extract building instances from binary segmentation mask using connected components.
    Returns list of dicts with: uid, bbox (x0,y0,x1,y1), area, centroid, mask_patch.
    Ref §A3: pred_instances — list of (uid, poly/mask, confidence).
    """
    import cv2
    import uuid

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

    instances = []
    for i in range(1, n_labels):  # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        x1 = x0 + bw
        y1 = y0 + bh
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        inst_mask = (labels == i).astype(np.uint8)
        instances.append({
            "uid":      str(uuid.uuid4()),
            "bbox":     (x0, y0, x1, y1),
            "area":     area,
            "centroid": (cx, cy),
            "mask":     inst_mask,
        })
    return instances


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_footprint_checkpoint(model: "nn.Module", path: str | Path, metadata: dict | None = None) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "unet_footprint",
        "model_state_dict": model.state_dict(),
        **(metadata or {}),
    }, path)


def load_footprint_checkpoint(path: str | Path, device: str = "cpu") -> "nn.Module":
    import torch
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = build_unet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Training: record builder, tile split, patch dataset
# ---------------------------------------------------------------------------

def build_footprint_records(
    index_csv: str | Path,
    use_pre_fallback: bool = True,
) -> list[dict]:
    """
    Build one record per tile for footprint U-Net training.
    Prefers post-disaster image; falls back to pre if post is missing.
    Returns list of {tile_id, image_path, label_json_path, disaster}.
    Skips tiles with no valid image or no label JSON.
    """
    import csv as _csv

    records = []
    with open(index_csv, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            label_path = row.get("label_json_path", "")
            if not label_path or not Path(label_path).is_file():
                continue
            post_path = row.get("post_path", "")
            pre_path  = row.get("pre_path", "")
            image_path = ""
            if post_path and Path(post_path).is_file():
                image_path = post_path
            elif use_pre_fallback and pre_path and Path(pre_path).is_file():
                image_path = pre_path
            if not image_path:
                continue
            records.append({
                "tile_id":         row["tile_id"],
                "image_path":      image_path,
                "label_json_path": label_path,
                "disaster":        row.get("disaster", ""),
            })
    return records


def footprint_tile_split(
    records: list[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Tile-level train/val split (no leakage)."""
    import random as _random

    tile_ids = sorted(set(r["tile_id"] for r in records))
    rng = _random.Random(seed)
    rng.shuffle(tile_ids)
    n_val = max(1, int(len(tile_ids) * val_fraction))
    val_tiles = set(tile_ids[:n_val])
    train = [r for r in records if r["tile_id"] not in val_tiles]
    val   = [r for r in records if r["tile_id"] in val_tiles]
    return train, val


def _make_patch_positions(
    img_h: int,
    img_w: int,
    ps: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Generate (y0, x0) top-left corners for all patches covering the image."""
    ys = sorted(set(
        list(range(0, max(img_h - ps, 0) + 1, stride)) + [max(0, img_h - ps)]
    ))
    xs = sorted(set(
        list(range(0, max(img_w - ps, 0) + 1, stride)) + [max(0, img_w - ps)]
    ))
    return [(y, x) for y in ys for x in xs]


class FootprintTileDataset:
    """
    Patch-based dataset for footprint U-Net training.

    Patchifies full satellite tiles into patch_size x patch_size crops with
    configurable stride overlap. Rasterizes ALL building polygons from the label
    JSON into a binary foreground mask (building=1, background=0). Un-classified
    buildings are included — we detect building presence, not damage.

    Returns:
        img_patch:  (3, patch_size, patch_size) float32 in [0, 1]
        mask_patch: (1, patch_size, patch_size) float32 binary
    """

    def __init__(
        self,
        records: list[dict],
        patch_size: int = 512,
        stride: int = 256,
        augment: bool = False,
        preload: bool = True,
    ) -> None:
        from PIL import Image as _PILImage

        self.records    = records
        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment

        # Pre-compute (rec_idx, y0, x0) patch positions using PIL header reads
        self._patches: list[tuple[int, int, int]] = []
        for i, r in enumerate(records):
            try:
                with _PILImage.open(r["image_path"]) as im:
                    img_w, img_h = im.size  # PIL: (width, height)
            except Exception:
                img_h = img_w = 1024
            for y, x in _make_patch_positions(img_h, img_w, patch_size, stride):
                self._patches.append((i, y, x))

        # Optional full preload into RAM
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        if preload:
            print(f"  Preloading {len(records)} tiles...", flush=True)
            self._cache = {i: self._load_tile(i) for i in range(len(records))}
            print("  Preload done.", flush=True)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec_idx, y0, x0 = self._patches[idx]
        if self._cache is not None:
            img, mask = self._cache[rec_idx]
        else:
            img, mask = self._load_tile(rec_idx)

        ps = self.patch_size
        h, w = img.shape[:2]
        y1 = min(y0 + ps, h)
        x1 = min(x0 + ps, w)
        img_p  = img[y0:y1, x0:x1]   # (ph, pw, 3)
        mask_p = mask[y0:y1, x0:x1]  # (ph, pw)

        # Zero-pad if patch is smaller than ps (image smaller than patch_size)
        if img_p.shape[0] < ps or img_p.shape[1] < ps:
            tmp_i = np.zeros((ps, ps, 3), dtype=np.uint8)
            tmp_m = np.zeros((ps, ps),    dtype=np.uint8)
            tmp_i[:img_p.shape[0], :img_p.shape[1]] = img_p
            tmp_m[:mask_p.shape[0], :mask_p.shape[1]] = mask_p
            img_p, mask_p = tmp_i, tmp_m

        img_t  = img_p.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3,H,W)
        mask_t = mask_p.astype(np.float32)                              # (H,W)

        if self.augment:
            img_t, mask_t = self._augment(img_t, mask_t)

        return img_t, mask_t[np.newaxis]  # (3,H,W), (1,H,W)

    def _load_tile(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Load full tile image and rasterize all building polygons into a binary mask."""
        import json as _json
        from disaster_bench.data.io import load_image
        from disaster_bench.data.polygons import parse_and_scale_building, scale_factors
        from disaster_bench.data.rasterize import rasterize_polygon

        r = self.records[i]
        img = load_image(r["image_path"])  # (H, W, 3) uint8
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        try:
            with open(r["label_json_path"], encoding="utf-8") as jf:
                data = _json.load(jf)
            meta   = data.get("metadata", {})
            json_w = int(meta.get("width",  1024))
            json_h = int(meta.get("height", 1024))
            sx, sy = scale_factors(w, h, json_w, json_h)
            for feat in data.get("features", {}).get("xy", []):
                wkt_str = feat.get("wkt", "")
                if not wkt_str:
                    continue
                poly, _ = parse_and_scale_building(wkt_str, sx, sy)
                if poly is None:
                    continue
                poly_mask = rasterize_polygon(poly, (h, w))
                np.maximum(mask, poly_mask, out=mask)
        except Exception:
            pass

        return img, mask

    def _augment(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spatial augmentation applied consistently to image (3,H,W) and mask (H,W)."""
        import random as _random

        if _random.random() > 0.5:           # horizontal flip
            img  = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if _random.random() > 0.5:           # vertical flip
            img  = img[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        k = _random.randint(0, 3)            # 90° rotation
        if k > 0:
            img  = np.rot90(img,  k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return img, mask

    def compute_pos_weight(self, max_tiles: int | None = None) -> float:
        """
        Estimate pos_weight = bg_pixels / building_pixels for BCEWithLogitsLoss.
        Uses cached data if preloaded; otherwise samples up to max_tiles tiles.
        """
        bg_total = fg_total = 0
        if self._cache is not None:
            items = list(self._cache.values())
        else:
            n = len(self.records) if max_tiles is None else min(max_tiles, len(self.records))
            items = [self._load_tile(i) for i in range(n)]
        for _, mask in items:
            fg = int(mask.sum())
            fg_total += fg
            bg_total += mask.size - fg
        return float(bg_total) / max(float(fg_total), 1.0)
