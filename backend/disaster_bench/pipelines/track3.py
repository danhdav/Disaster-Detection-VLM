"""
Track 3 — Hybrid: ML/VLM prediction + geometry guardrails.
Ref §B Track 3: Track 1 proposes damage; geometry computes features; policy accepts/adjusts/flags.

Single-pass: per-tile change maps computed ONCE, then per-building features use those maps.
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
from disaster_bench.data.dataset import DAMAGE_CLASSES, load_crop_pair
from disaster_bench.geometry.features import mask_area, area_change_ratio
from disaster_bench.geometry.thresholds import thresholds_to_damage
from disaster_bench.pipelines.track2a import compute_change_score, _ssim_map

_STRONG_CHANGE_THRESH = 0.4
_LOW_CHANGE_THRESH    = 0.05
_MIN_CONF             = 0.35


def _guardrail_policy(
    ml_damage: str,
    ml_conf: float,
    geom: dict[str, float],
    *,
    strong_thresh: float = _STRONG_CHANGE_THRESH,
    low_thresh: float    = _LOW_CHANGE_THRESH,
    min_conf: float      = _MIN_CONF,
) -> tuple[str, bool, str]:
    score      = geom.get("change_score", 0.0)
    geom_dmg   = thresholds_to_damage(pct_changed=geom.get("pct_changed", 0.0))
    low_conf   = ml_conf < min_conf
    geo_dmg    = score >= strong_thresh
    geo_undmg  = score <= low_thresh
    ml_dmg     = ml_damage != "no-damage"
    ml_undmg   = ml_damage == "no-damage"
    conflict   = (ml_dmg and geo_undmg) or (ml_undmg and geo_dmg)
    flagged    = low_conf or conflict

    if conflict:
        final  = geom_dmg
        reason = f"adjusted:geo={geom_dmg},ml={ml_damage},score={score:.3f}"
    elif low_conf:
        final  = ml_damage
        reason = f"flagged_low_conf:{ml_conf:.3f}"
    else:
        final  = ml_damage
        reason = "accepted"
    return final, flagged, reason


def run_track3(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config    = config or {}
    _cuda = False
    try:
        import torch as _t; _cuda = _t.cuda.is_available()
    except ImportError:
        pass
    device    = config.get("device", "cuda" if _cuda else "cpu")
    ckpt      = config.get("model_ckpt", "")
    crop_size = int(config.get("input_size", 128))
    crops_dir = Path(config.get("crops_dir", "data/processed/crops_oracle"))

    model = None
    if ckpt and Path(ckpt).is_file():
        try:
            from disaster_bench.models.damage.six_channel import load_checkpoint
            model = load_checkpoint(ckpt, device=device)
            print(f"  T3: loaded model {ckpt}")
        except Exception as e:
            print(f"  T3: model load failed ({e})")

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

        img_h, img_w = post_img.shape[:2] if post_img is not None else (json_h, json_w)
        sx, sy = scale_factors(img_w, img_h, json_w, json_h)

        # --- Compute tile-level change maps ONCE ---
        if pre_img is not None and post_img is not None and pre_img.shape[:2] == (img_h, img_w):
            pre_gray  = pre_img.mean(axis=2).astype(np.float32) / 255.0
            post_gray = post_img.mean(axis=2).astype(np.float32) / 255.0
            diff_map  = np.abs(post_gray - pre_gray)
            try:
                ssim_map = _ssim_map(pre_gray * 255, post_gray * 255)
            except Exception:
                ssim_map = None
            have_maps = True
        else:
            diff_map = ssim_map = None
            have_maps = False

        # --- Batch ML predictions for this tile ---
        if model is not None:
            import torch
            crop_uids, crop_arrays = [], []
            for b in buildings:
                uid = b["uid"]
                op = crops_dir / tile_id / uid / "pre_bbox.png"
                oo = crops_dir / tile_id / uid / "post_bbox.png"
                if op.is_file() and oo.is_file():
                    crop_uids.append(uid)
                    crop_arrays.append(load_crop_pair(op, oo, crop_size))
            batch_sz = int(config.get("inference_batch", 64))
            uid_to_pred: dict[str, tuple[str, float]] = {}
            for i in range(0, len(crop_arrays), batch_sz):
                batch = np.stack(crop_arrays[i:i + batch_sz])
                t = torch.from_numpy(batch).float().to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(t), dim=1).cpu().numpy()
                for uid_i, p in zip(crop_uids[i:i + batch_sz], probs):
                    idx = int(np.argmax(p))
                    uid_to_pred[uid_i] = (DAMAGE_CLASSES[idx], float(p[idx]))
        else:
            uid_to_pred = {}

        for b in buildings:
            uid = b["uid"]
            poly, _ = parse_and_scale_building(b["wkt"], sx, sy)

            # --- ML prediction (from pre-computed batch) ---
            if uid in uid_to_pred:
                ml_damage, ml_conf = uid_to_pred[uid]
            else:
                ml_damage = config.get("placeholder_damage", "no-damage")
                ml_conf   = 0.5

            if poly is None or not have_maps:
                all_predictions.append({
                    "tile_id": tile_id, "pred_instance_id": uid,
                    "matched_gt_uid": None, "iou": None,
                    "pred_damage": ml_damage, "pred_conf": round(ml_conf, 4),
                    "track": 3, "notes": "",
                })
                continue

            mask         = rasterize_polygon(poly, (img_h, img_w))
            building_area = float(mask_area(mask))
            if building_area == 0:
                all_predictions.append({
                    "tile_id": tile_id, "pred_instance_id": uid,
                    "matched_gt_uid": None, "iou": None,
                    "pred_damage": ml_damage, "pred_conf": round(ml_conf, 4),
                    "track": 3, "notes": "",
                })
                continue

            # Use pre-computed tile maps for building-level features
            score  = compute_change_score(mask, diff_map, ssim_map)
            pct_ch = 100.0 * float(np.logical_and(mask > 0, diff_map > 0.1).sum()) / building_area

            geom = {
                "building_area": building_area,
                "pct_changed": round(pct_ch, 2),
                "change_score": round(score, 4),
            }

            final_damage, flagged, reason = _guardrail_policy(
                ml_damage, ml_conf, geom,
                strong_thresh=float(config.get("strong_thresh", _STRONG_CHANGE_THRESH)),
                low_thresh=float(config.get("low_thresh",    _LOW_CHANGE_THRESH)),
                min_conf=float(config.get("min_confidence", _MIN_CONF)),
            )

            notes = ";".join(filter(None, ["flagged_inconsistent" if flagged else "", reason]))
            all_predictions.append({
                "tile_id": tile_id,
                "pred_instance_id": uid,
                "matched_gt_uid": None,
                "iou": None,
                "pred_damage": final_damage,
                "pred_conf": round(ml_conf, 4),
                "track": 3,
                "notes": notes,
                "ev_building_area": geom["building_area"],
                "ev_pct_changed": geom["pct_changed"],
                "ev_change_score": geom["change_score"],
            })

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    n = max(len(all_predictions), 1)
    print(f"  T3: {len(all_predictions)} predictions in {elapsed_ms/1000:.1f}s "
          f"({elapsed_ms/n:.1f} ms/instance)")
    return all_predictions


def run_track3_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    import time
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    t0 = time.perf_counter()
    preds = run_track3(index_csv, run_dir, config)
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
