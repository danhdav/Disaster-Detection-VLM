"""
WKT parsing and coordinate scaling (sx, sy) from JSON canvas to image resolution.
"""
from __future__ import annotations

from typing import Any

from shapely import wkt
from shapely.geometry import Polygon


def parse_wkt_polygon(wkt_str: str) -> Polygon | None:
    """Parse WKT string to Shapely Polygon. Returns None if invalid or empty."""
    if not (wkt_str or wkt_str.strip()):
        return None
    try:
        g = wkt.loads(wkt_str)
        if isinstance(g, Polygon) and not g.is_empty:
            return g
        return None
    except Exception:
        return None


def scale_polygon(
    poly: Polygon,
    sx: float,
    sy: float,
) -> Polygon:
    """Scale polygon by sx (x) and sy (y)."""
    from shapely import affinity
    return affinity.scale(poly, xfact=sx, yfact=sy, origin=(0, 0))


def polygon_to_bbox(poly: Polygon) -> tuple[int, int, int, int]:
    """Return (minx, miny, maxx, maxy) as integers (floor min, ceil max)."""
    from math import ceil, floor
    minx, miny, maxx, maxy = poly.bounds
    return int(floor(minx)), int(floor(miny)), int(ceil(maxx)), int(ceil(maxy))


def scale_factors(
    img_width: int,
    img_height: int,
    json_width: int,
    json_height: int,
) -> tuple[float, float]:
    """
    Scale from JSON canvas to image: sx = W_img / W_json, sy = H_img / H_json.
    """
    if json_width <= 0 or json_height <= 0:
        return 1.0, 1.0
    return img_width / json_width, img_height / json_height


def parse_and_scale_building(
    wkt_str: str,
    sx: float = 1.0,
    sy: float = 1.0,
) -> tuple[Polygon | None, tuple[int, int, int, int] | None]:
    """
    Parse WKT, optionally scale, return (polygon, bbox).
    bbox is (minx, miny, maxx, maxy) in scaled coords; None if poly invalid.
    """
    poly = parse_wkt_polygon(wkt_str)
    if poly is None:
        return None, None
    if sx != 1.0 or sy != 1.0:
        poly = scale_polygon(poly, sx, sy)
    bbox = polygon_to_bbox(poly)
    return poly, bbox
