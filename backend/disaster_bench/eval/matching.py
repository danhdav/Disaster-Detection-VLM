"""
IoU one-to-one matching: predicted instances <-> GT instances.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Polygon

from disaster_bench.data.rasterize import rasterize_polygon


def polygon_iou(poly_a: Polygon, poly_b: Polygon, shape: tuple[int, int]) -> float:
    """Compute IoU of two polygons by rasterizing to shape (h, w)."""
    if poly_a.is_empty and poly_b.is_empty:
        return 1.0
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    m_a = rasterize_polygon(poly_a, shape)
    m_b = rasterize_polygon(poly_b, shape)
    inter = np.logical_and(m_a > 0, m_b > 0).sum()
    union = np.logical_or(m_a > 0, m_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def match_predictions_to_gt(
    pred_instances: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
    shape: tuple[int, int],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Greedy one-to-one matching by IoU. Each pred matches at most one GT, each GT at most one pred.
    pred_instances / gt_instances: list of { "id" or "uid", "polygon" (Shapely) or "wkt" }.
    Returns list of { pred_id, matched_gt_uid, iou } for matched pairs; unmatched preds have matched_gt_uid None.
    """
    from disaster_bench.data.polygons import parse_wkt_polygon

    def get_poly(obj: dict) -> Polygon | None:
        if "polygon" in obj and obj["polygon"] is not None:
            return obj["polygon"]
        if "wkt" in obj:
            return parse_wkt_polygon(obj["wkt"])
        return None

    def get_id(obj: dict, key: str = "uid") -> str:
        return str(obj.get(key, obj.get("id", "")))

    pred_polys: list[tuple[str, Polygon]] = []
    for p in pred_instances:
        poly = get_poly(p)
        if poly is not None:
            pred_polys.append((get_id(p, "id"), poly))
    gt_polys: list[tuple[str, Polygon]] = []
    for g in gt_instances:
        poly = get_poly(g)
        if poly is not None:
            gt_polys.append((get_id(g), poly))

    # Compute IoU matrix
    n_p, n_g = len(pred_polys), len(gt_polys)
    iou_mat = np.zeros((n_p, n_g))
    for i, (_, pa) in enumerate(pred_polys):
        for j, (_, ga) in enumerate(gt_polys):
            iou_mat[i, j] = polygon_iou(pa, ga, shape)

    # Greedy: repeatedly take max IoU pair above threshold
    matched_gt = set()
    results: list[dict[str, Any]] = []
    used_pred = set()
    while True:
        best = -1.0
        best_p, best_g = -1, -1
        for i in range(n_p):
            if i in used_pred:
                continue
            for j in range(n_g):
                if j in matched_gt:
                    continue
                if iou_mat[i, j] >= iou_threshold and iou_mat[i, j] > best:
                    best = iou_mat[i, j]
                    best_p, best_g = i, j
        if best_p < 0:
            break
        used_pred.add(best_p)
        matched_gt.add(best_g)
        results.append({
            "pred_id": pred_polys[best_p][0],
            "matched_gt_uid": gt_polys[best_g][0],
            "iou": round(best, 4),
        })
    for i in range(n_p):
        if i not in used_pred:
            results.append({
                "pred_id": pred_polys[i][0],
                "matched_gt_uid": None,
                "iou": None,
            })
    return results
