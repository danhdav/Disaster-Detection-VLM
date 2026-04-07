'''
This file contains functions for parsing the image JSON files.
This includes calculating the polygons for the bounding boxes and creating the data schema to store VLM predictions in the database.
'''

from typing import Any


def parse_polygon_wkt_bounds(wkt: str) -> list[float] | None:
    # Simple POLYGON ((lng lat, ...)) as in xView2 labels → [min_lng, min_lat, max_lng, max_lat].
    if not wkt.startswith("POLYGON"):
        return None

    content = wkt.replace("POLYGON", "", 1).strip()
    if not (content.startswith("((") and content.endswith("))")):
        return None

    points = content[2:-2].split(",")
    min_lng = float("inf")
    min_lat = float("inf")
    max_lng = float("-inf")
    max_lat = float("-inf")

    for point in points:
        parts = point.strip().split()
        if len(parts) < 2:
            continue
        lng = float(parts[0])
        lat = float(parts[1])
        min_lng = min(min_lng, lng)
        min_lat = min(min_lat, lat)
        max_lng = max(max_lng, lng)
        max_lat = max(max_lat, lat)

    if min_lng == float("inf"):
        return None
    return [min_lng, min_lat, max_lng, max_lat]


def merge_bounds(base: list[float] | None, nxt: list[float] | None) -> list[float] | None:
    if base is None:
        return nxt
    if nxt is None:
        return base
    return [
        min(base[0], nxt[0]),
        min(base[1], nxt[1]),
        max(base[2], nxt[2]),
        max(base[3], nxt[3]),
    ]


def extract_label_bounds(label_data: dict[str, Any]) -> list[float] | None:
    bounds: list[float] | None = None
    lng_lat_features = label_data.get("features", {}).get("lng_lat", [])
    for feature in lng_lat_features:
        feature_bounds = parse_polygon_wkt_bounds(feature.get("wkt", ""))
        bounds = merge_bounds(bounds, feature_bounds)
    return bounds


def find_feature_by_uid(
    pre_phase: dict[str, Any] | None,
    post_phase: dict[str, Any] | None,
    feature_id: str,
) -> dict[str, Any] | None:
    """Return the lng_lat feature whose ``properties.uid`` matches ``feature_id``."""
    for phase_data in (pre_phase, post_phase):
        if not phase_data:
            continue
        for feature in phase_data.get("features", {}).get("lng_lat", []):
            uid = feature.get("properties", {}).get("uid")
            if uid == feature_id:
                return feature
    return None
