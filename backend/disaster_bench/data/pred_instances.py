"""
Predicted building footprint instances (not oracle).
Ref §A3: data/processed/pred_instances/<tile_id>/instances.json + mask PNGs.

Workflow:
  1. Run semantic_seg footprint model on a post-event image -> binary mask
  2. Extract connected-component instances from mask
  3. Save each instance: bounding box, area, centroid, polygon, confidence
  4. At evaluation time, IoU-match pred instances to GT instances
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PRED_INSTANCES_DIR = "data/processed/pred_instances"


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_pred_instances(
    tile_id: str,
    instances: list[dict[str, Any]],
    out_root: str | Path = PRED_INSTANCES_DIR,
) -> Path:
    """
    Persist predicted instances to JSON + binary mask PNGs.
    Each instance dict must contain: uid, bbox, area, centroid.
    'mask' ndarray is saved as a separate PNG, excluded from JSON.
    """
    import cv2
    out_dir = Path(out_root) / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_list = []
    for inst in instances:
        uid = inst["uid"]
        mask = inst.get("mask")
        if mask is not None:
            mask_path = out_dir / f"{uid}_mask.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        meta_list.append({
            "uid":      uid,
            "bbox":     list(inst["bbox"]),
            "area":     inst["area"],
            "centroid": list(inst["centroid"]),
            "conf":     float(inst.get("conf", 1.0)),
            "mask_file": f"{uid}_mask.png" if mask is not None else None,
        })

    out_json = out_dir / "instances.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"tile_id": tile_id, "instances": meta_list}, f, indent=2)
    return out_json


def load_pred_instances(
    tile_id: str,
    out_root: str | Path = PRED_INSTANCES_DIR,
    load_masks: bool = False,
) -> list[dict[str, Any]]:
    """Load saved predicted instances for a tile."""
    import cv2
    json_path = Path(out_root) / tile_id / "instances.json"
    if not json_path.is_file():
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    instances = data.get("instances", [])
    if load_masks:
        inst_dir = json_path.parent
        for inst in instances:
            mf = inst.get("mask_file")
            if mf:
                mp = inst_dir / mf
                if mp.is_file():
                    inst["mask"] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    return instances


# ---------------------------------------------------------------------------
# Localization geometry checks (Ref §3.1)
# ---------------------------------------------------------------------------

def instance_count_check(
    pred_instances: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Ref §3.1: Instance count + localization geometry checks.
    Returns summary: count ratio, unmatched preds, unmatched GTs.
    """
    n_pred = len(pred_instances)
    n_gt   = len(gt_instances)
    ratio  = n_pred / n_gt if n_gt > 0 else float("inf")
    return {
        "n_pred":      n_pred,
        "n_gt":        n_gt,
        "count_ratio": round(ratio, 4),
        "over_detect": n_pred > n_gt,
        "under_detect": n_pred < n_gt,
    }


def iou_2d_boxes(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Axis-aligned bounding box IoU."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw  = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_pred_to_gt(
    pred_instances: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    One-to-one greedy IoU matching of predicted to GT building bboxes.
    Returns:
      matches: list of (pred_uid, gt_uid, iou)
      unmatched_preds: list of pred_uids with no GT match
      unmatched_gts:   list of GT uids not covered
      coverage: fraction of GT buildings matched
    Ref §A3 + §3.1: Coverage (% GT buildings found) + localization quality.
    """
    matched_pred: set[str] = set()
    matched_gt:   set[str] = set()
    matches = []

    # Score all pairs
    scored = []
    for p in pred_instances:
        for g in gt_instances:
            iou = iou_2d_boxes(tuple(p["bbox"]), tuple(g["bbox"]))  # type: ignore[arg-type]
            if iou >= iou_threshold:
                scored.append((iou, p["uid"], g["uid"]))
    scored.sort(reverse=True)

    for iou, p_uid, g_uid in scored:
        if p_uid in matched_pred or g_uid in matched_gt:
            continue
        matches.append({"pred_uid": p_uid, "gt_uid": g_uid, "iou": round(iou, 4)})
        matched_pred.add(p_uid)
        matched_gt.add(g_uid)

    gt_uids   = [g["uid"] for g in gt_instances]
    pred_uids = [p["uid"] for p in pred_instances]
    cov = len(matched_gt) / len(gt_uids) if gt_uids else 1.0

    return {
        "matches":         matches,
        "unmatched_preds": [u for u in pred_uids  if u not in matched_pred],
        "unmatched_gts":   [u for u in gt_uids    if u not in matched_gt],
        "n_matches":       len(matches),
        "n_pred":          len(pred_instances),
        "n_gt":            len(gt_instances),
        "coverage":        round(cov, 4),
        "mean_iou":        round(sum(m["iou"] for m in matches) / max(len(matches), 1), 4),
    }


# ---------------------------------------------------------------------------
# Full footprint prediction pipeline for one tile
# ---------------------------------------------------------------------------

def predict_footprints_for_tile(
    post_image: np.ndarray,
    model: "nn.Module | None" = None,
    device: str = "cpu",
    threshold: float = 0.5,
    min_area: int = 50,
) -> list[dict[str, Any]]:
    """
    Run footprint model on a tile's post image, extract instances.
    Falls back to empty list if no model is provided.
    """
    if model is None:
        return []
    from disaster_bench.models.footprints.semantic_seg import predict_mask, mask_to_instances
    binary = predict_mask(model, post_image, device=device, threshold=threshold)
    return mask_to_instances(binary, min_area=min_area)
