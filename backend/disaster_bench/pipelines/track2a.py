"""
Track 2A — Unsupervised numeric baseline.
Ref §B Track 2A: change signals (abs-diff, SSIM, edge-diff) aggregated inside GT footprint
-> threshold rules -> damage class.  No GT damage labels at inference.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from disaster_bench.data.io import (
    get_buildings_from_label,
    get_label_canvas_size,
    load_image,
    load_label_json,
    read_index_csv,
)
from disaster_bench.data.polygons import parse_and_scale_building, scale_factors
from disaster_bench.data.rasterize import rasterize_polygon
from disaster_bench.geometry.features import mask_area
from disaster_bench.geometry.thresholds import thresholds_to_damage


def _ssim_map(a: np.ndarray, b: np.ndarray, C1: float = 6.5025, C2: float = 58.5225) -> np.ndarray:
    """Local SSIM map (grayscale float32 inputs). Returns per-pixel dissimilarity (1 - SSIM)."""
    import cv2
    a = a.astype(np.float32); b = b.astype(np.float32)
    k = 7
    mu1 = cv2.GaussianBlur(a, (k, k), 1.5)
    mu2 = cv2.GaussianBlur(b, (k, k), 1.5)
    mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu1_mu2 = mu1 * mu2
    s1 = cv2.GaussianBlur(a * a, (k, k), 1.5) - mu1_sq
    s2 = cv2.GaussianBlur(b * b, (k, k), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(a * b, (k, k), 1.5) - mu1_mu2
    ssim = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return np.clip(1.0 - ssim, 0, 1)


def _edge_change(pre_gray: np.ndarray, post_gray: np.ndarray) -> np.ndarray:
    """Absolute edge-response difference (Canny magnitude)."""
    import cv2
    pre_e  = cv2.Canny(pre_gray,  50, 150).astype(np.float32) / 255.0
    post_e = cv2.Canny(post_gray, 50, 150).astype(np.float32) / 255.0
    return np.abs(pre_e - post_e)


def compute_change_score(
    mask: np.ndarray,
    diff_map: np.ndarray,
    ssim_map: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
    diff_weight: float = 0.5,
    ssim_weight: float = 0.3,
    edge_weight: float = 0.2,
) -> float:
    """
    Weighted combination of change signals inside building mask.
    Returns mean change score [0,1].
    """
    building_pixels = int(np.sum(mask > 0))
    if building_pixels == 0:
        return 0.0

    score = diff_map[mask > 0].mean() * diff_weight
    if ssim_map is not None:
        score += ssim_map[mask > 0].mean() * ssim_weight
    if edge_map is not None:
        score += edge_map[mask > 0].mean() * edge_weight

    # Normalise by active weight sum
    total_w = diff_weight + (ssim_weight if ssim_map is not None else 0) + (edge_weight if edge_map is not None else 0)
    return float(score / total_w) if total_w > 0 else 0.0


def run_track2a(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = config or {}
    diff_thresh = float(config.get("diff_threshold", 25.0)) / 255.0

    # Damage thresholds (% changed inside building)
    thresh_minor     = float(config.get("thresh_minor",     8.0))
    thresh_major     = float(config.get("thresh_major",    25.0))
    thresh_destroyed = float(config.get("thresh_destroyed", 55.0))

    rows = read_index_csv(index_csv)
    all_predictions: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for row in rows:
        tile_id    = row["tile_id"]
        pre_path   = row.get("pre_path", "")
        post_path  = row.get("post_path", "")
        label_path = row.get("label_json_path", "")
        if not label_path:
            continue
        try:
            label_data = load_label_json(label_path)
            buildings  = get_buildings_from_label(label_data, use_xy=True)
        except Exception:
            continue

        json_w, json_h = get_label_canvas_size(label_data)
        if json_w <= 0:
            json_w, json_h = 1024, 1024

        try:
            pre_img  = load_image(pre_path)  if pre_path  else None
            post_img = load_image(post_path) if post_path else None
        except Exception:
            pre_img = post_img = None

        if post_img is None:
            for b in buildings:
                all_predictions.append(_row(tile_id, b["uid"], "no-damage", 0.0, "2A", {}))
            continue

        img_h, img_w = post_img.shape[:2]
        sx, sy = scale_factors(img_w, img_h, json_w, json_h)

        # Build change maps once per tile
        post_gray = post_img.mean(axis=2).astype(np.float32) / 255.0
        if pre_img is not None and pre_img.shape[:2] == (img_h, img_w):
            pre_gray = pre_img.mean(axis=2).astype(np.float32) / 255.0
            diff_map = np.abs(post_gray - pre_gray)
            try:
                ssim_map = _ssim_map(pre_gray * 255, post_gray * 255)
                pre_u8   = (pre_gray  * 255).astype(np.uint8)
                post_u8  = (post_gray * 255).astype(np.uint8)
                edge_map = _edge_change(pre_u8, post_u8)
            except Exception:
                ssim_map = edge_map = None
        else:
            diff_map = np.zeros((img_h, img_w), dtype=np.float32)
            ssim_map = edge_map = None

        diff_binary = (diff_map > diff_thresh).astype(np.uint8)

        for b in buildings:
            uid = b["uid"]
            poly, _ = parse_and_scale_building(b["wkt"], sx, sy)
            if poly is None:
                all_predictions.append(_row(tile_id, uid, "no-damage", 0.0, "2A", {}))
                continue

            mask = rasterize_polygon(poly, (img_h, img_w))
            bpx  = mask_area(mask)
            if bpx == 0:
                all_predictions.append(_row(tile_id, uid, "no-damage", 0.0, "2A", {}))
                continue

            # % pixels changed (binary diff)
            pct_binary = 100.0 * float(np.logical_and(mask > 0, diff_binary > 0).sum()) / bpx
            # Weighted multi-signal score
            score = compute_change_score(mask, diff_map, ssim_map, edge_map)

            pred = thresholds_to_damage(
                pct_changed=pct_binary,
                thresh_minor=thresh_minor,
                thresh_major=thresh_major,
                thresh_destroyed=thresh_destroyed,
            )
            evidence = {
                "pct_changed": round(pct_binary, 2),
                "change_score": round(score, 4),
                "building_pixels": bpx,
            }
            all_predictions.append(_row(tile_id, uid, pred, min(score * 2, 1.0), "2A", evidence))

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    n = max(len(all_predictions), 1)
    print(f"  T2A: {len(all_predictions)} predictions in {elapsed_ms/1000:.1f}s "
          f"({elapsed_ms/n:.1f} ms/instance)")
    return all_predictions


def _row(
    tile_id: str, uid: str, pred: str, conf: float, track: str, evidence: dict
) -> dict[str, Any]:
    return {
        "tile_id": tile_id,
        "pred_instance_id": uid,
        "matched_gt_uid": None,
        "iou": None,
        "pred_damage": pred,
        "pred_conf": round(conf, 4),
        "track": track,
        "notes": "",
        **{f"ev_{k}": v for k, v in evidence.items()},
    }


def run_track2a_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    import time
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    t0 = time.perf_counter()
    preds = run_track2a(index_csv, run_dir, config)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    n = max(len(preds), 1)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_predictions_csv(preds, Path(run_dir) / "predictions.csv")
    write_metrics_json({
        "macro_f1": None, "per_class_f1": {}, "coverage": None,
        "avg_latency_ms": round(elapsed_ms / n, 2),
        "total_elapsed_s": round(elapsed_ms / 1000, 2),
        "note": "Run eval-run to compute F1 metrics",
    }, Path(run_dir) / "metrics.json")
