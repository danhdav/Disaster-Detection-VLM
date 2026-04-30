"""
Track 1 — REAL DEPLOYMENT variant (no oracle footprints).
Ref §B Track 1 full pipeline:
  1. Run semantic seg footprint model on post image → binary mask
  2. Extract connected-component instances
  3. IoU-match predicted instances → GT instances (for evaluation only)
  4. Extract 6-channel crops from predicted bbox
  5. Damage classifier → per-building damage label

This is the "deployable pipeline" as opposed to the oracle upper-bound in track1.py.
GT labels are NEVER used at inference; only at eval time for IoU matching.
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
from disaster_bench.data.dataset import DAMAGE_CLASSES
from disaster_bench.data.pred_instances import (
    predict_footprints_for_tile,
    match_pred_to_gt,
    save_pred_instances,
)


def _load_damage_model(ckpt: str, device: str) -> "nn.Module | None":
    if not ckpt or not Path(ckpt).is_file():
        return None
    try:
        from disaster_bench.models.damage.six_channel import load_checkpoint
        m = load_checkpoint(ckpt, device=device)
        m.eval()
        return m
    except Exception as e:
        print(f"  T1-deploy: damage model load failed ({e})")
        return None


def _load_footprint_model(ckpt: str, device: str) -> "nn.Module | None":
    if not ckpt or not Path(ckpt).is_file():
        return None
    try:
        from disaster_bench.models.footprints.semantic_seg import load_footprint_checkpoint
        m = load_footprint_checkpoint(ckpt, device=device)
        return m
    except Exception as e:
        print(f"  T1-deploy: footprint model load failed ({e})")
        return None


def _bbox_crop(img: np.ndarray, bbox: tuple, size: int) -> np.ndarray:
    """Crop image to bbox (x0,y0,x1,y1), resize to size×size, return (H,W,3)."""
    from PIL import Image
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), dtype=np.float32)
    patch = img[y0:y1, x0:x1]
    pil   = Image.fromarray(patch).resize((size, size), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def run_track1_deploy(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Full deployment pipeline: footprint prediction + IoU matching + damage classification.
    Returns per-predicted-instance rows with matched_gt_uid when available.
    """
    config     = config or {}
    device     = config.get("device", "cuda" if _cuda_available() else "cpu")
    dmg_ckpt   = config.get("model_ckpt", "")
    foot_ckpt  = config.get("footprint_ckpt", "")
    crop_size  = int(config.get("input_size", 128))
    batch_sz   = int(config.get("inference_batch", 64))
    iou_thr    = float(config.get("match_iou_threshold", 0.5))
    pred_dir   = config.get("pred_instances_dir", "data/processed/pred_instances")
    save_insts = bool(config.get("save_pred_instances", True))

    dmg_model  = _load_damage_model(dmg_ckpt, device)
    foot_model = _load_footprint_model(foot_ckpt, device)

    mode = ("footprint+damage" if foot_model and dmg_model
            else "oracle-bbox+damage" if dmg_model
            else "placeholder")
    print(f"  T1-deploy: mode={mode} device={device}")

    rows = read_index_csv(index_csv)
    all_predictions: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for row in rows:
        tile_id    = row["tile_id"]
        post_path  = row.get("post_path", "")
        pre_path   = row.get("pre_path", "")
        label_path = row.get("label_json_path", "")

        try:
            post_img = load_image(post_path) if post_path else None
        except Exception:
            post_img = None
        try:
            pre_img = load_image(pre_path) if pre_path else None
        except Exception:
            pre_img = None

        # --- Step 1: Get predicted instances ---
        if foot_model is not None and post_img is not None:
            pred_insts = predict_footprints_for_tile(post_img, foot_model, device=device)
        elif label_path and Path(label_path).is_file():
            # Oracle bbox fallback (upper-bound mode)
            try:
                label_data = load_label_json(label_path)
                gt_blds    = get_buildings_from_label(label_data, use_xy=True)
                json_w, json_h = get_label_canvas_size(label_data)
                if json_w <= 0:
                    json_w, json_h = 1024, 1024
                img_h = post_img.shape[0] if post_img is not None else json_h
                img_w = post_img.shape[1] if post_img is not None else json_w
                from disaster_bench.data.polygons import parse_wkt_polygon, scale_factors
                sx, sy = scale_factors(img_w, img_h, json_w, json_h)
                pred_insts = []
                for b in gt_blds:
                    try:
                        poly = parse_wkt_polygon(b["wkt"])
                        from shapely.geometry import box as sbox
                        coords = list(poly.exterior.coords)
                        xs = [c[0] * sx for c in coords]
                        ys = [c[1] * sy for c in coords]
                        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                        cx, cy = float(poly.centroid.x * sx), float(poly.centroid.y * sy)
                        pred_insts.append({
                            "uid": b["uid"],
                            "bbox": (x0, y0, x1, y1),
                            "area": int((x1-x0)*(y1-y0)),
                            "centroid": (cx, cy),
                            "conf": 1.0,
                        })
                    except Exception:
                        continue
            except Exception:
                pred_insts = []
        else:
            pred_insts = []

        # --- Step 2: IoU-match pred → GT (for eval) ---
        match_result: dict[str, Any] = {"matches": [], "coverage": 0.0}
        if label_path and Path(label_path).is_file():
            try:
                label_data = load_label_json(label_path)
                gt_blds    = get_buildings_from_label(label_data, use_xy=True)
                json_w, json_h = get_label_canvas_size(label_data)
                if json_w <= 0:
                    json_w, json_h = 1024, 1024
                img_h = post_img.shape[0] if post_img is not None else json_h
                img_w = post_img.shape[1] if post_img is not None else json_w
                from disaster_bench.data.polygons import parse_wkt_polygon, scale_factors
                sx, sy = scale_factors(img_w, img_h, json_w, json_h)
                gt_insts = []
                for b in gt_blds:
                    try:
                        poly = parse_wkt_polygon(b["wkt"])
                        coords = list(poly.exterior.coords)
                        xs = [c[0] * sx for c in coords]
                        ys = [c[1] * sy for c in coords]
                        gt_insts.append({
                            "uid": b["uid"],
                            "bbox": (min(xs), min(ys), max(xs), max(ys)),
                        })
                    except Exception:
                        continue
                match_result = match_pred_to_gt(pred_insts, gt_insts, iou_threshold=iou_thr)
            except Exception:
                pass

        uid_to_gt = {m["pred_uid"]: m["gt_uid"] for m in match_result.get("matches", [])}

        # --- Step 3: Save predicted instances ---
        if save_insts and pred_insts:
            save_pred_instances(tile_id, pred_insts, out_root=pred_dir)

        # --- Step 4: Damage classification ---
        if not pred_insts:
            continue

        if dmg_model is not None and post_img is not None and pre_img is not None:
            import torch
            crops, uids_order = [], []
            for inst in pred_insts:
                pre_c  = _bbox_crop(pre_img,  inst["bbox"], crop_size)  # (H,W,3) float
                post_c = _bbox_crop(post_img, inst["bbox"], crop_size)
                combined = np.concatenate([pre_c, post_c], axis=2).transpose(2, 0, 1)
                crops.append(combined)
                uids_order.append(inst["uid"])

            preds_dm: list[tuple[str, float]] = []
            for i in range(0, len(crops), batch_sz):
                batch = np.stack(crops[i:i + batch_sz]).astype(np.float32)
                t = torch.from_numpy(batch).to(device)
                with torch.no_grad():
                    probs = torch.softmax(dmg_model(t), dim=1).cpu().numpy()
                for p in probs:
                    idx = int(np.argmax(p))
                    preds_dm.append((DAMAGE_CLASSES[idx], float(p[idx])))
        else:
            uids_order = [inst["uid"] for inst in pred_insts]
            preds_dm   = [("no-damage", 0.5)] * len(uids_order)

        for uid, (pred_dmg, pred_conf) in zip(uids_order, preds_dm):
            all_predictions.append({
                "tile_id":          tile_id,
                "pred_instance_id": uid,
                "matched_gt_uid":   uid_to_gt.get(uid),
                "iou":              None,
                "pred_damage":      pred_dmg,
                "pred_conf":        round(pred_conf, 4),
                "track":            "1-deploy",
                "notes":            f"coverage={match_result.get('coverage', 0):.3f}",
            })

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    n = max(len(all_predictions), 1)
    print(f"  T1-deploy: {len(all_predictions)} predictions in "
          f"{elapsed_ms/1000:.1f}s ({elapsed_ms/n:.1f} ms/instance)")
    return all_predictions


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_track1_deploy_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    import time
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    t0 = time.perf_counter()
    preds = run_track1_deploy(index_csv, run_dir, config)
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
