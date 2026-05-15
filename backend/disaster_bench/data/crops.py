"""
Oracle crop generation: use GT polygons to crop pre/post per building (uid).
Also supports cropping from predicted bboxes/masks (for tracks).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from disaster_bench.data.io import (
    get_buildings_from_label,
    get_label_canvas_size,
    load_image,
    load_label_json,
)
from disaster_bench.data.polygons import (
    parse_and_scale_building,
    scale_factors,
)
from disaster_bench.data.rasterize import mask_to_bbox, rasterize_polygon


def _translate_polygon(poly: Polygon, dx: float, dy: float) -> Polygon:
    """Shift a polygon by (dx, dy) — maps image-space coords into padded-crop-local coords."""
    from shapely import affinity
    return affinity.translate(poly, xoff=dx, yoff=dy)


def _draw_polygon_outline(
    crop: np.ndarray,
    poly: Polygon,
    color: tuple[int, int, int] = (255, 60, 60),
    line_width: int = 2,
) -> np.ndarray:
    """
    Draw a thin polygon outline on a uint8 RGB crop using PIL.
    Polygon coordinates must already be in crop-local pixel space.
    Returns a new array (does not modify crop in-place).
    """
    img = Image.fromarray(crop.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
    n = len(coords)
    for i in range(n):
        draw.line([coords[i], coords[(i + 1) % n]], fill=color, width=line_width)
    return np.array(img)


def make_oracle_crops_for_tile(
    tile_id: str,
    pre_path: str | Path,
    post_path: str | Path,
    label_json_path: str | Path,
    out_dir: str | Path,
    *,
    pad_fraction: float = 0.25,
    overwrite: bool = False,
) -> int:
    """
    For one tile: load pre/post images and post label JSON; for each building
    parse WKT (xy), scale to image, apply bbox-relative padding, crop pre and post.

    Saves per building uid:
      pre_bbox.png  / post_bbox.png  : padded crop with polygon outline drawn in red
      pre_masked.png/ post_masked.png: same padded region, non-building pixels blacked out

    pad_fraction: padding added on each side relative to building size.
        pad_px = max(pad_fraction * max(bbox_w, bbox_h), 8)
        e.g. 0.25 → 25% of the larger bbox dimension on every side.

    Returns number of buildings written.
    """
    out_base = Path(out_dir) / tile_id
    label_data = load_label_json(label_json_path)
    buildings = get_buildings_from_label(label_data, use_xy=True)
    json_w, json_h = get_label_canvas_size(label_data)
    if json_w <= 0 or json_h <= 0:
        json_w, json_h = 1024, 1024  # fallback

    try:
        pre_img = load_image(pre_path)
    except Exception:
        pre_img = None
    try:
        post_img = load_image(post_path)
    except Exception:
        post_img = None

    if pre_img is None and post_img is None:
        return 0

    # Use post image shape for scaling if available; else pre; else json size
    if post_img is not None:
        img_h, img_w = post_img.shape[:2]
    elif pre_img is not None:
        img_h, img_w = pre_img.shape[:2]
    else:
        img_w, img_h = json_w, json_h
    sx, sy = scale_factors(img_w, img_h, json_w, json_h)

    count = 0
    for b in buildings:
        uid = b["uid"]
        poly, bbox = parse_and_scale_building(b["wkt"], sx, sy)
        if poly is None or bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # Clamp tight bbox to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue

        # --- bbox-relative padding ---
        bbox_w, bbox_h = x2 - x1, y2 - y1
        pad = max(int(pad_fraction * max(bbox_w, bbox_h)), 8)
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(img_w, x2 + pad)
        py2 = min(img_h, y2 + pad)

        uid_dir = out_base / uid
        uid_dir.mkdir(parents=True, exist_ok=True)

        # Rasterize mask at full image resolution, then slice to padded region
        mask = rasterize_polygon(poly, (img_h, img_w))
        mask_roi = mask[py1:py2, px1:px2]           # (crop_h, crop_w)
        mask_3ch = np.stack([mask_roi] * 3, axis=2) # broadcast for RGB multiply

        # Polygon translated into padded-crop local coords for outline drawing
        poly_local = _translate_polygon(poly, -px1, -py1)

        for src_img, prefix in ((pre_img, "pre"), (post_img, "post")):
            if src_img is None:
                continue
            crop = src_img[py1:py2, px1:px2].copy()

            # Raw crop: no outline — clean training input
            raw_path = uid_dir / f"{prefix}_raw.png"
            if overwrite or not raw_path.exists():
                Image.fromarray(crop).save(raw_path)

            # Context crop: padded + thin red polygon outline (legacy training input)
            outlined = _draw_polygon_outline(crop, poly_local)
            bbox_path = uid_dir / f"{prefix}_bbox.png"
            if overwrite or not bbox_path.exists():
                Image.fromarray(outlined).save(bbox_path)

            # Explicit outlined name for documentation clarity
            outlined_path = uid_dir / f"{prefix}_outlined.png"
            if overwrite or not outlined_path.exists():
                Image.fromarray(outlined).save(outlined_path)

            # Masked crop: non-building pixels blacked out (RGB, not RGBA)
            masked = np.where(mask_3ch > 0, crop, np.uint8(0)).astype(np.uint8)
            masked_path = uid_dir / f"{prefix}_masked.png"
            if overwrite or not masked_path.exists():
                Image.fromarray(masked).save(masked_path)

        count += 1
    return count


def crop_from_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop image by (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()
