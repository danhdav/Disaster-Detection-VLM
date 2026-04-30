"""
Rasterize polygon -> binary mask (numpy, same shape as image or given shape).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from shapely.geometry import Polygon


def rasterize_polygon(
    poly: Polygon,
    shape: Tuple[int, int],
    origin_upper_left: bool = True,
) -> np.ndarray:
    """
    Rasterize polygon to binary mask of shape (height, width).
    Uses integer coordinates; polygon coords are (x, y) with x=col, y=row.
    If origin_upper_left True, row 0 is top; else row 0 is bottom.
    """
    import cv2
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    # Get exterior coords
    if poly.is_empty:
        return mask
    exterior = poly.exterior
    if exterior is None:
        return mask
    coords = np.array(exterior.xy).T  # (N, 2) as (x, y)
    if len(coords) < 3:
        return mask
    # OpenCV fillPoly expects (N, 1, 2) int32
    pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
    # Clip to image bounds
    pts = np.clip(pts, [0, 0], [w - 1, h - 1])
    cv2.fillPoly(mask, [pts], 1)
    if not origin_upper_left:
        mask = np.flipud(mask)
    return mask


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (minx, miny, maxx, maxy) of the bounding box of non-zero pixels. None if empty."""
    if mask.size == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not (rows.any() and cols.any()):
        return None
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return cmin, rmin, cmax + 1, rmax + 1
