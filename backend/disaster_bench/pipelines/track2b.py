"""
Track 2B — Damage-map aggregation baseline.
Ref §B Track 2B: pixel-wise change/severity -> aggregate % damaged pixels inside footprint
-> threshold rules -> building class.

Uses a Siamese pixel-difference severity map (no GT labels at inference).
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
from disaster_bench.geometry.features import mask_area, pct_damaged_pixels
from disaster_bench.geometry.thresholds import thresholds_to_damage


def _pixel_severity_map(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    thresh_minor: float = 20.0,
    thresh_major: float = 50.0,
    thresh_destroyed: float = 100.0,
) -> np.ndarray:
    """
    Compute per-pixel severity map {0,1,2,3} from pre/post RGB.
    0=no-damage, 1=minor, 2=major, 3=destroyed.
    Uses: abs-diff (RGB channels) + post vs pre brightness drop.
    """
    pre_f  = pre_img.astype(np.float32)
    post_f = post_img.astype(np.float32)

    # Channel-wise absolute difference
    diff = np.abs(post_f - pre_f)           # (H, W, 3)
    diff_mean = diff.mean(axis=2)           # (H, W)

    # Brightness drop: pre was bright (undamaged), post is dark (ash/rubble)
    pre_bright  = pre_f.mean(axis=2)
    post_bright = post_f.mean(axis=2)
    brightness_drop = np.clip(pre_bright - post_bright, 0, 255)

    # Combined severity signal
    signal = (diff_mean * 0.6 + brightness_drop * 0.4)

    severity = np.zeros(signal.shape, dtype=np.uint8)
    severity[signal > thresh_minor]    = 1
    severity[signal > thresh_major]    = 2
    severity[signal > thresh_destroyed] = 3
    return severity


def run_track2b(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = config or {}
    thresh_minor     = float(config.get("pix_minor",    15.0))
    thresh_major     = float(config.get("pix_major",    25.0))
    thresh_destroyed = float(config.get("pix_destroyed", 55.0))

    # Building-level % damaged thresholds
    bld_minor     = float(config.get("thresh_minor",     10.0))
    bld_major     = float(config.get("thresh_major",     30.0))
    bld_destroyed = float(config.get("thresh_destroyed", 60.0))

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
                all_predictions.append(_row(tile_id, b["uid"], "no-damage", 0.0, {}))
            continue

        img_h, img_w = post_img.shape[:2]
        sx, sy = scale_factors(img_w, img_h, json_w, json_h)

        if pre_img is not None and pre_img.shape[:2] == (img_h, img_w):
            severity_map = _pixel_severity_map(
                pre_img, post_img, thresh_minor, thresh_major, thresh_destroyed
            )
        else:
            severity_map = np.zeros((img_h, img_w), dtype=np.uint8)

        for b in buildings:
            uid = b["uid"]
            poly, _ = parse_and_scale_building(b["wkt"], sx, sy)
            if poly is None:
                all_predictions.append(_row(tile_id, uid, "no-damage", 0.0, {}))
                continue

            mask = rasterize_polygon(poly, (img_h, img_w))
            bpx  = mask_area(mask)
            if bpx == 0:
                all_predictions.append(_row(tile_id, uid, "no-damage", 0.0, {}))
                continue

            pct_any    = pct_damaged_pixels(severity_map, mask, severity_min=1)
            pct_major_ = pct_damaged_pixels(severity_map, mask, severity_min=2)
            pct_dest   = pct_damaged_pixels(severity_map, mask, severity_min=3)

            # Use strongest signal
            pred = thresholds_to_damage(
                pct_damaged=pct_any,
                thresh_minor=bld_minor,
                thresh_major=bld_major,
                thresh_destroyed=bld_destroyed,
            )
            evidence = {
                "pct_any_damage": round(pct_any, 2),
                "pct_major_plus": round(pct_major_, 2),
                "pct_destroyed":  round(pct_dest, 2),
                "building_pixels": bpx,
            }
            all_predictions.append(_row(tile_id, uid, pred, min(pct_any / 100.0, 1.0), evidence))

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    n = max(len(all_predictions), 1)
    print(f"  T2B: {len(all_predictions)} predictions in {elapsed_ms/1000:.1f}s "
          f"({elapsed_ms/n:.1f} ms/instance)")
    return all_predictions


def _row(tile_id, uid, pred, conf, evidence):
    return {
        "tile_id": tile_id,
        "pred_instance_id": uid,
        "matched_gt_uid": None,
        "iou": None,
        "pred_damage": pred,
        "pred_conf": round(conf, 4),
        "track": "2B",
        "notes": "",
        **{f"ev_{k}": v for k, v in evidence.items()},
    }


def run_track2b_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    import time
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    t0 = time.perf_counter()
    preds = run_track2b(index_csv, run_dir, config)
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
