"""
PyTorch dataset for oracle crops: loads (pre_bbox.png, post_bbox.png) pairs + label.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Augmentation config dataclass (plain dict is fine for CLI → dataset bridge)
# ---------------------------------------------------------------------------
#
# AugConfig fields (all optional, default = off):
#   rotate90           float  — if > 0, sample k uniformly from {0,90,180,270} (true uniform
#                               4-orientation aug; value acts as on/off switch, not probability)
#   affine             float  — probability of small affine warp (translate ±10%, scale 0.9-1.1)
#   color_jitter       float  — probability of brightness/contrast jitter (SAME params pre+post)
#   color_jitter_indep float  — probability of independent jitter per image (pre != post params;
#                               simulates real satellite pre/post acquisition differences)
#   noise              float  — probability of additive Gaussian noise
#   class_conditional  bool   — if True, minor(1) and major(2) always get full aug regardless
#                               of probability flags; common classes keep probabilistic behavior
#
# H/V flips: rare classes always flip; common classes 50% prob each (existing behaviour).
# All geometric ops applied identically to all 6 channels so pre/post stay aligned.
# Leakage note: augmentation is on-the-fly (one view per building per epoch) — no sample
# multiplication, so tile-level train/val split fully prevents leakage.

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
LABEL2IDX = {c: i for i, c in enumerate(DAMAGE_CLASSES)}
# "" (missing label) maps to no-damage.
# "un-classified" is intentionally NOT remapped here — it is treated as a missing
# label and skipped in training/eval so the model never learns from unknown GT.
SUBTYPE_REMAP = {"": "no-damage"}


def remap_subtype(subtype: str) -> str:
    return SUBTYPE_REMAP.get(subtype, subtype)


def build_crop_records(
    index_csv: str | Path,
    crops_dir: str | Path,
    use_raw_crops: bool = False,
) -> list[dict[str, Any]]:
    """
    Walk crops_oracle and pair each crop pair with its GT label.

    use_raw_crops=False (default): loads pre_bbox.png / post_bbox.png (outlined, legacy).
    use_raw_crops=True: loads pre_raw.png / post_raw.png (no outline, clean input).
      Falls back to pre_bbox.png with a warning if pre_raw.png is missing.

    Returns list of {tile_id, uid, pre_path, post_path, label, label_idx}.
    """
    import sys as _sys
    crops_dir = Path(crops_dir)
    pre_fname  = "pre_raw.png"  if use_raw_crops else "pre_bbox.png"
    post_fname = "post_raw.png" if use_raw_crops else "post_bbox.png"

    # Build gt_damage lookup: (tile_id, uid) -> subtype
    # Also collect tile_id -> official_split (default "test" for backward compat)
    gt: dict[tuple[str, str], str] = {}
    tile_split: dict[str, str] = {}
    with open(index_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tile_id = row["tile_id"]
            tile_split[tile_id] = row.get("official_split", "test")
            label_path = row.get("label_json_path", "")
            if not label_path or not Path(label_path).is_file():
                continue
            with open(label_path, encoding="utf-8") as jf:
                data = json.load(jf)
            for feat in data.get("features", {}).get("xy", []):
                uid = feat["properties"].get("uid", "")
                subtype = feat["properties"].get("subtype", "")
                if uid and subtype != "un-classified":
                    gt[(tile_id, uid)] = remap_subtype(subtype)

    records = []
    n_fallback = 0
    for pre_path in crops_dir.rglob(pre_fname):
        uid_dir = pre_path.parent
        post_path = uid_dir / post_fname
        if not post_path.is_file():
            if use_raw_crops:
                # Fall back to outlined if raw is missing
                fb_pre  = uid_dir / "pre_bbox.png"
                fb_post = uid_dir / "post_bbox.png"
                if fb_pre.is_file() and fb_post.is_file():
                    pre_path  = fb_pre
                    post_path = fb_post
                    n_fallback += 1
                else:
                    continue
            else:
                continue
        uid = uid_dir.name
        tile_id = uid_dir.parent.name
        label = gt.get((tile_id, uid))
        if label is None or label not in LABEL2IDX:
            continue
        records.append({
            "tile_id": tile_id,
            "uid": uid,
            "pre_path": str(pre_path),
            "post_path": str(post_path),
            "label": label,
            "label_idx": LABEL2IDX[label],
            "official_split": tile_split.get(tile_id, "test"),
        })
    if use_raw_crops and n_fallback > 0:
        print(f"  WARNING [build_crop_records]: {n_fallback} buildings fell back to"
              f" pre_bbox.png (pre_raw.png missing). Run make_oracle_crops.py to backfill.",
              file=_sys.stderr)
    return records


def train_val_split(
    records: list[dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tile-level split: hold out val_fraction of tile_ids."""
    tile_ids = sorted(set(r["tile_id"] for r in records))
    rng = random.Random(seed)
    rng.shuffle(tile_ids)
    n_val = max(1, int(len(tile_ids) * val_fraction))
    val_tiles = set(tile_ids[:n_val])
    train = [r for r in records if r["tile_id"] not in val_tiles]
    val = [r for r in records if r["tile_id"] in val_tiles]
    return train, val


def load_crop_pair(
    pre_path: str | Path,
    post_path: str | Path,
    size: int = 128,
) -> np.ndarray:
    """Load and resize pre+post crops; stack to (6, H, W) float32 in [0,1]."""
    def _load(p: str | Path) -> np.ndarray:
        with Image.open(p) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.array(im, dtype=np.float32) / 255.0

    pre = _load(pre_path)   # (H,W,3)
    post = _load(post_path) # (H,W,3)
    combined = np.concatenate([pre, post], axis=2)  # (H,W,6)
    return combined.transpose(2, 0, 1)              # (6,H,W)


# ---------------------------------------------------------------------------
# Augmentation helpers (module-level so they can be imported by tests)
# ---------------------------------------------------------------------------

def _apply_affine(x: np.ndarray) -> np.ndarray:
    """
    Small affine warp: random translate (±10% of H/W) + scale (0.9–1.1).
    Applied identically to all channels via scipy.ndimage.affine_transform.
    Falls back to identity if scipy is unavailable.
    """
    try:
        from scipy.ndimage import affine_transform
    except ImportError:
        return x

    _, H, W = x.shape
    # Random scale and translate — same for all channels
    scale = random.uniform(0.9, 1.1)
    tx    = random.uniform(-0.1 * W, 0.1 * W)
    ty    = random.uniform(-0.1 * H, 0.1 * H)
    # Affine matrix (for ndimage: output[o] = input[matrix @ o + offset])
    matrix = np.array([[scale, 0], [0, scale]])
    offset = np.array([ty + H * (1 - scale) / 2, tx + W * (1 - scale) / 2])
    out = np.stack([
        affine_transform(x[c], matrix, offset=offset,
                         order=1, mode="reflect", prefilter=False)
        for c in range(x.shape[0])
    ], axis=0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_color_jitter(x: np.ndarray) -> np.ndarray:
    """
    Random brightness + contrast jitter.
    SAME random parameters applied to pre (ch 0-2) AND post (ch 3-5) to avoid
    injecting spurious change signal.
    x: (6, H, W) float32 in [0, 1]
    """
    brightness = random.uniform(-0.15, 0.15)
    contrast   = random.uniform(0.85, 1.15)
    out = x * contrast + brightness
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_color_jitter_indep(x: np.ndarray) -> np.ndarray:
    """
    Independent brightness + contrast jitter for pre (ch 0-2) vs post (ch 3-5).
    Simulates real satellite acquisition differences: different sun angles,
    atmospheric conditions, and sensor states between pre and post passes.
    x: (6, H, W) float32 in [0, 1]
    """
    def _jitter(channels: np.ndarray) -> np.ndarray:
        brightness = random.uniform(-0.15, 0.15)
        contrast   = random.uniform(0.85, 1.15)
        return np.clip(channels * contrast + brightness, 0.0, 1.0)

    out = x.copy()
    out[:3] = _jitter(x[:3])
    out[3:] = _jitter(x[3:])
    return out.astype(np.float32)


def _resize_6ch(x: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a (6, H', W') float32 array back to (6, H, W) using PIL bilinear interpolation."""
    out = np.zeros((6, H, W), dtype=np.float32)
    for c in range(6):
        ch_uint8 = (np.clip(x[c], 0.0, 1.0) * 255).astype(np.uint8)
        im = Image.fromarray(ch_uint8, mode="L")
        im = im.resize((W, H), Image.BILINEAR)
        out[c] = np.array(im, dtype=np.float32) / 255.0
    return out


def tta_transforms(x: np.ndarray, n_views: int = 4) -> list[np.ndarray]:
    """
    Return deterministic test-time augmentation views of x (6, H, W).
    n_views=4: four 90-degree rotations {0°, 90°, 180°, 270°}.
    n_views=8: four rotations × horizontal flip (8 views total).
    Average the softmax outputs over all views at inference time.
    """
    assert n_views in (4, 8), "n_views must be 4 or 8"
    views = []
    for k in range(4):
        rot = np.ascontiguousarray(np.rot90(x, k, axes=(1, 2)))
        views.append(rot)
        if n_views == 8:
            views.append(np.ascontiguousarray(rot[:, :, ::-1]))  # horizontal flip
    return views


class CropDataset:
    """Minimal dataset (no torch dependency at import time).

    Set preload=True to load all images into RAM once (much faster for repeated epochs).

    aug_config: dict controlling extra augmentations (all default off).
      Keys: rotate90 (float prob), affine (float prob),
            color_jitter (float prob), noise (float prob).
      H/V flips are always on when augment=True (existing behaviour).
    """
    def __init__(
        self,
        records: list[dict[str, Any]],
        size: int = 128,
        augment: bool = False,
        preload: bool = True,
        aug_config: dict | None = None,
    ) -> None:
        self.records    = records
        self.size       = size
        self.augment    = augment
        self.aug_config = aug_config or {}
        self._cache: list[np.ndarray] | None = None
        if preload:
            print(f"  Preloading {len(records)} crops into RAM...", flush=True)
            self._cache = [
                load_crop_pair(r["pre_path"], r["post_path"], size) for r in records
            ]
            print("  Preload done.", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        if self._cache is not None:
            x = self._cache[idx].copy()
        else:
            r = self.records[idx]
            x = load_crop_pair(r["pre_path"], r["post_path"], self.size)
        label_idx = self.records[idx]["label_idx"]
        if self.augment:
            x = self._augment(x, label_idx)
        return x, label_idx

    def _augment(self, x: np.ndarray, label_idx: int = 0) -> np.ndarray:
        """
        Apply augmentations to a (6, H, W) array.
        Channels 0-2 = pre RGB, 3-5 = post RGB.
        ALL geometric transforms applied identically to all 6 channels (pre/post stay aligned).
        Only called when augment=True — never on val/test splits.

        class_conditional mode (minor=1, major=2):
          - Rare classes: always apply geometric aug regardless of probability flags.
          - Common classes (no-damage=0, destroyed=3): keep probabilistic behavior.
          is_rare is always from ground-truth label_idx, never a model prediction.
        """
        cfg = self.aug_config
        is_rare = cfg.get("class_conditional", False) and label_idx in (1, 2)

        # --- H/V flips ---
        # Rare: uniform over {none, hflip, vflip, both} — avoids always-double-flip artifacts.
        # Common: independent 50% prob per axis (original behaviour).
        if is_rare:
            flip = random.randint(0, 3)   # 0=none, 1=hflip, 2=vflip, 3=both
            if flip & 1:
                x = x[:, :, ::-1].copy()
            if flip & 2:
                x = x[:, ::-1, :].copy()
        else:
            if random.random() > 0.5:
                x = x[:, :, ::-1].copy()
            if random.random() > 0.5:
                x = x[:, ::-1, :].copy()

        # --- Rotation: uniform {0°, 90°, 180°, 270°} ---
        # Semantics of --aug_rotate90 p (probability flag) are preserved:
        #   Rare:   always rotate (ignore p); sample k uniformly from {0,1,2,3}.
        #   Common: gate by probability first (random() < p), then sample k uniformly.
        # This keeps "rotate90=0.5" meaning "rotate 50% of the time" for common classes.
        p_rot = cfg.get("rotate90", 0.0)
        if is_rare:
            # Rare: always apply a non-identity rotation (k in {1,2,3} = 90/180/270°).
            # k=randint(1,3) ensures every rare-class epoch sees a distinct orientation.
            k = random.randint(1, 3)
            x = np.rot90(x, k, axes=(1, 2)).copy()
        elif p_rot > 0.0 and random.random() < p_rot:
            k = random.randint(0, 3)    # uniform: 0=identity, 1=90°, 2=180°, 3=270°
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy()

        # --- Small affine warp (translate + scale) ---
        p_aff = cfg.get("affine", 0.0)
        if p_aff > 0.0 and random.random() < p_aff:
            x = _apply_affine(x)

        # --- Shared color jitter (same params pre+post) ---
        p_cj = cfg.get("color_jitter", 0.0)
        eff_cj = 1.0 if (is_rare and p_cj > 0.0) else p_cj
        if eff_cj > 0.0 and random.random() < eff_cj:
            x = _apply_color_jitter(x)

        # --- Independent color jitter (different params pre vs post) ---
        p_indep = cfg.get("color_jitter_indep", 0.0)
        eff_indep = 1.0 if (is_rare and p_indep > 0.0) else p_indep
        if eff_indep > 0.0 and random.random() < eff_indep:
            x = _apply_color_jitter_indep(x)

        # --- Additive Gaussian noise ---
        p_noise = cfg.get("noise", 0.0)
        if p_noise > 0.0 and random.random() < p_noise:
            sigma = random.uniform(0.005, 0.02)
            noise = np.random.normal(0.0, sigma, x.shape).astype(np.float32)
            x = np.clip(x + noise, 0.0, 1.0)

        # --- Multi-scale crop/pad (enabled via aug_config["multiscale"]=True) ---
        # Randomly selects tight (60% FOV), normal (unchanged), or wide (140% FOV padded).
        # After crop/pad, resizes back to the original H×W so the tensor shape is preserved.
        # Intended for Stage 3 (minor vs major) where building context matters.
        if cfg.get("multiscale", False):
            scale = random.choice(["tight", "normal", "wide"])
            _, H, W = x.shape
            if scale == "tight":
                ch = int(H * 0.20)   # crop 20% off each edge → 60% of H remains
                cw = int(W * 0.20)
                x = x[:, ch:H - ch, cw:W - cw]
                x = _resize_6ch(x, H, W)
            elif scale == "wide":
                ph = int(H * 0.40)   # pad 40% on each side → 180% of H total
                pw = int(W * 0.40)
                x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)),
                           mode="constant", constant_values=0)
                x = _resize_6ch(x, H, W)
            # normal: no change

        # Ensure contiguous memory layout — np.rot90 and flip slices can produce
        # negative-stride views that cause torch.from_numpy() to fail.
        return np.ascontiguousarray(x)

    def as_nine_channel(self) -> "NineChannelDataset":
        """Return a view of this dataset with 9-channel (pre+post+diff) inputs."""
        return NineChannelDataset(self)

    def class_weights(self) -> np.ndarray:
        """Inverse-frequency class weights for weighted cross-entropy."""
        import numpy as np
        counts = np.zeros(len(DAMAGE_CLASSES), dtype=np.float32)
        for r in self.records:
            counts[r["label_idx"]] += 1
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights /= weights.sum()
        return weights * len(DAMAGE_CLASSES)


class NineChannelDataset:
    """Wraps CropDataset, returning 9-channel (pre+post+|diff|) tensors."""
    def __init__(self, base: CropDataset) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        x6, label = self._base[idx]   # (6,H,W)
        pre  = x6[:3]
        post = x6[3:]
        diff = np.abs(post - pre)
        return np.concatenate([pre, post, diff], axis=0), label  # (9,H,W)

    def class_weights(self) -> np.ndarray:
        return self._base.class_weights()


def load_centroid_patch(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    cx: int,
    cy: int,
    size: int = 128,
) -> np.ndarray:
    """
    Extract fixed (size x size) patch centred at (cx, cy) from pre and post images.
    Pads with zeros if patch extends outside image boundaries.
    Returns (6, size, size) float32 in [0,1].
    """
    h, w = pre_img.shape[:2]
    half = size // 2

    def _extract(img: np.ndarray) -> np.ndarray:
        canvas = np.zeros((size, size, 3), dtype=np.float32)
        # Source coords
        sx0 = max(cx - half, 0);       sy0 = max(cy - half, 0)
        sx1 = min(cx + half, w);       sy1 = min(cy + half, h)
        # Dest coords
        dx0 = sx0 - (cx - half);       dy0 = sy0 - (cy - half)
        dx1 = dx0 + (sx1 - sx0);       dy1 = dy0 + (sy1 - sy0)
        patch = img[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
        canvas[dy0:dy1, dx0:dx1] = patch
        return canvas

    pre_p  = _extract(pre_img)   # (H,W,3)
    post_p = _extract(post_img)  # (H,W,3)
    combined = np.concatenate([pre_p, post_p], axis=2)  # (H,W,6)
    return combined.transpose(2, 0, 1)                   # (6,H,W)


def build_centroid_records(
    index_csv: str | Path,
) -> list[dict[str, Any]]:
    """
    Build records for centroid-patch training.
    Each record stores full image paths + polygon centroid + label.
    """
    import json, csv
    from disaster_bench.data.polygons import parse_wkt_polygon

    records = []
    with open(index_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tile_id    = row["tile_id"]
            pre_path   = row.get("pre_path", "")
            post_path  = row.get("post_path", "")
            label_path = row.get("label_json_path", "")
            if not label_path or not Path(label_path).is_file():
                continue
            with open(label_path, encoding="utf-8") as jf:
                data = json.load(jf)
            meta   = data.get("metadata", {})
            json_w = meta.get("width",  1024)
            json_h = meta.get("height", 1024)
            for feat in data.get("features", {}).get("xy", []):
                props       = feat.get("properties", {})
                uid         = props.get("uid", "")
                raw_subtype = props.get("subtype", "")
                if raw_subtype == "un-classified":
                    continue  # skip: unknown GT, don't train on it
                subtype = remap_subtype(raw_subtype)
                if not uid or subtype not in LABEL2IDX:
                    continue
                wkt = feat.get("wkt") or ""
                if not wkt:
                    continue
                try:
                    poly = parse_wkt_polygon(wkt)
                    cx, cy = int(poly.centroid.x), int(poly.centroid.y)
                except Exception:
                    continue
                records.append({
                    "tile_id":   tile_id,
                    "uid":       uid,
                    "pre_path":  pre_path,
                    "post_path": post_path,
                    "cx": cx, "cy": cy,
                    "json_w": json_w, "json_h": json_h,
                    "label":     subtype,
                    "label_idx": LABEL2IDX[subtype],
                })
    return records


def collate(batch):
    """Standard DataLoader collate for (image, label) crop batches."""
    import torch
    xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys
