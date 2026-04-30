"""
Geometry features: building area, %changed, %damaged_pixels, area_change.
Used for Track 2 baselines and Track 3 guardrails.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from disaster_bench.data.rasterize import rasterize_polygon


def mask_area(mask: np.ndarray) -> int:
    """Total non-zero pixels."""
    return int(np.sum(mask > 0))


def polygon_area_pixels(poly: Polygon, shape: tuple[int, int]) -> int:
    """Rasterize polygon and return area in pixels."""
    m = rasterize_polygon(poly, shape)
    return mask_area(m)


def pct_changed(
    pre_mask_roi: np.ndarray,
    post_mask_roi: np.ndarray,
    diff_mask: np.ndarray,
) -> float:
    """
    Percentage of building (post_mask_roi) that is in diff_mask.
    All arrays same shape; values 0/1 or 0/255.
    """
    building_pixels = np.sum(post_mask_roi > 0)
    if building_pixels == 0:
        return 0.0
    changed = np.logical_and(post_mask_roi > 0, diff_mask > 0)
    return float(np.sum(changed)) / float(building_pixels) * 100.0


def pct_damaged_pixels(
    damage_mask: np.ndarray,
    building_mask: np.ndarray,
    severity_min: int = 1,
) -> float:
    """
    Percentage of building pixels with damage severity >= severity_min.
    damage_mask: per-pixel severity (0=no, 1=minor, 2=major, 3=destroyed).
    building_mask: binary building footprint.
    """
    building_pixels = np.sum(building_mask > 0)
    if building_pixels == 0:
        return 0.0
    damaged = np.logical_and(building_mask > 0, damage_mask >= severity_min)
    return float(np.sum(damaged)) / float(building_pixels) * 100.0


def area_change_ratio(
    pre_area: int | float,
    post_area: int | float,
) -> float:
    """(post_area - pre_area) / pre_area. Returns 0 if pre_area == 0."""
    if pre_area <= 0:
        return 0.0
    return (float(post_area) - float(pre_area)) / float(pre_area)
