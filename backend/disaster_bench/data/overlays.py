"""
Debug overlays: draw building masks/outlines on pre/post images for sanity checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import cv2

from disaster_bench.data.io import (
    get_buildings_from_label,
    get_label_canvas_size,
    load_image,
    load_label_json,
)
from disaster_bench.data.polygons import parse_and_scale_building, scale_factors
from disaster_bench.data.rasterize import rasterize_polygon


# Colors BGR for opencv (one per class for outlines, optional)
DAMAGE_COLORS = {
    "no-damage": (0, 255, 0),
    "minor-damage": (0, 255, 255),
    "major-damage": (0, 165, 255),
    "destroyed": (0, 0, 255),
    "un-classified": (128, 128, 128),
}


def draw_mask_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay binary mask on image with given color and alpha."""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    overlay = out.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def draw_polygon_outline(
    img: np.ndarray,
    poly_xy: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw polygon outline on image (in-place). poly_xy shape (N, 2) as (x, y)."""
    pts = np.array(poly_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def make_overlay_for_tile(
    tile_id: str,
    pre_path: str | Path,
    post_path: str | Path,
    label_json_path: str | Path,
    out_path: str | Path,
    *,
    which: str = "post",
    alpha: float = 0.35,
) -> bool:
    """
    Draw building masks (from label JSON) over pre or post image; save to out_path.
    which: "pre" | "post"
    """
    label_data = load_label_json(label_json_path)
    buildings = get_buildings_from_label(label_data, use_xy=True)
    json_w, json_h = get_label_canvas_size(label_data)
    if json_w <= 0 or json_h <= 0:
        json_w, json_h = 1024, 1024

    img_path = post_path if which == "post" else pre_path
    try:
        img = load_image(img_path)
    except Exception:
        return False
    img_h, img_w = img.shape[:2]
    sx, sy = scale_factors(img_w, img_h, json_w, json_h)

    for b in buildings:
        poly, _ = parse_and_scale_building(b["wkt"], sx, sy)
        if poly is None:
            continue
        mask = rasterize_polygon(poly, (img_h, img_w))
        subtype = b.get("subtype", "no-damage")
        color = DAMAGE_COLORS.get(subtype, (0, 255, 0))
        img = draw_mask_overlay(img, mask, color=color, alpha=alpha)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # img is RGB from load_image; cv2 uses BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), img_bgr)
    return True
