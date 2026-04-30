"""
Track 1 — End-to-End ML: oracle crops -> batch damage classifier -> per-building label.
Ref §B Track 1.  Prediction-time inputs: pre_image, post_image (no GT labels at inference).

Uses batch GPU inference (load all crops for a tile, run one forward pass per tile).
Falls back to placeholder when no checkpoint is configured.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from disaster_bench.data.io import (
    get_buildings_from_label,
    get_label_canvas_size,
    load_label_json,
    read_index_csv,
)
from disaster_bench.data.dataset import DAMAGE_CLASSES, load_crop_pair, load_centroid_patch


def _apply_tau_norm(logits: "torch.Tensor", tau: float) -> "torch.Tensor":
    """Scale logits by 1/tau before softmax (temperature normalisation).
    tau > 1 sharpens minority-class boundaries; tau=1.0 is identity."""
    if tau != 1.0:
        logits = logits / tau
    return logits


def _tta4_probs(
    model: Any,
    t: "torch.Tensor",
    use_9ch: bool,
    tau: float,
    device: str,
) -> "np.ndarray":
    """Average softmax over 4 × 90-degree rotations (TTA-4).
    t shape: (B, C, H, W) already on device."""
    import torch
    from disaster_bench.models.damage.classifiers import PrePostDiffCNN
    acc: "torch.Tensor | None" = None
    for k in range(4):
        rot = torch.rot90(t, k=k, dims=[2, 3])
        if use_9ch:
            rot = PrePostDiffCNN.from_six_channel(rot)
        with torch.no_grad():
            logits = model(rot)
            p = torch.softmax(_apply_tau_norm(logits, tau), dim=1)
        acc = p if acc is None else acc + p
    return (acc / 4).cpu().numpy()  # type: ignore[operator]


def run_track1(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config    = config or {}
    device    = config.get("device", "cuda" if _cuda_available() else "cpu")
    ckpt      = config.get("model_ckpt", "")
    crop_size = int(config.get("input_size", 128))
    crops_dir = Path(config.get("crops_dir", "data/processed/crops_oracle"))
    batch_sz  = int(config.get("inference_batch", 64))
    tau       = float(config.get("tau", 1.0))   # temperature-norm (1.0 = off)
    use_tta   = bool(config.get("tta", False))   # TTA-4 rotations

    model = None
    model_type = config.get("model_type", "six_channel")
    if ckpt and Path(ckpt).is_file():
        try:
            if model_type == "six_channel":
                from disaster_bench.models.damage.six_channel import load_checkpoint
                model = load_checkpoint(ckpt, device=device)
            else:
                from disaster_bench.models.damage.classifiers import load_classifier
                model, model_type = load_classifier(ckpt, device=device)
            model.eval()
            print(f"  T1: {model_type} model loaded ({device})")
        except Exception as e:
            print(f"  T1: model load failed ({e}); using placeholder.")

    is_centroid = (model_type == "centroid_patch")

    rows = read_index_csv(index_csv)
    all_predictions: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for row in rows:
        tile_id    = row["tile_id"]
        label_path = row.get("label_json_path", "")
        if not label_path:
            continue
        try:
            label_data = load_label_json(label_path)
            buildings  = get_buildings_from_label(label_data, use_xy=True)
        except Exception:
            continue

        if model is None:
            for b in buildings:
                all_predictions.append(_pred_row(tile_id, b["uid"],
                    config.get("placeholder_damage", "no-damage"), 0.5))
            continue

        # --- Batch inference for this tile ---
        uids, crops = [], []

        if is_centroid:
            # Centroid-patch mode: load full tile images and extract fixed patches
            # around each building centroid — matches the training distribution.
            from PIL import Image as _PIL
            from disaster_bench.data.polygons import parse_wkt_polygon
            try:
                pre_tile  = np.array(_PIL.open(row["pre_path"]).convert("RGB"))
                post_tile = np.array(_PIL.open(row["post_path"]).convert("RGB"))
            except Exception:
                for b in buildings:
                    all_predictions.append(_pred_row(tile_id, b["uid"], "no-damage", 0.5))
                continue
            json_w, json_h = get_label_canvas_size(label_data)
            if json_w <= 0:
                json_w, json_h = 1024, 1024
            img_h, img_w = pre_tile.shape[:2]
            sx = img_w / json_w
            sy = img_h / json_h
            for b in buildings:
                uid = b["uid"]
                wkt = b.get("wkt", "")
                if not wkt:
                    all_predictions.append(_pred_row(tile_id, uid, "no-damage", 0.5))
                    continue
                try:
                    poly = parse_wkt_polygon(wkt)
                    cx = int(poly.centroid.x * sx)
                    cy = int(poly.centroid.y * sy)
                except Exception:
                    all_predictions.append(_pred_row(tile_id, uid, "no-damage", 0.5))
                    continue
                uids.append(uid)
                crops.append(load_centroid_patch(pre_tile, post_tile, cx, cy, crop_size))
        else:
            # Oracle-crop mode: load pre_bbox.png / post_bbox.png for each building.
            for b in buildings:
                uid = b["uid"]
                op = crops_dir / tile_id / uid / "pre_bbox.png"
                oo = crops_dir / tile_id / uid / "post_bbox.png"
                if op.is_file() and oo.is_file():
                    uids.append(uid)
                    crops.append(load_crop_pair(op, oo, crop_size))
                else:
                    all_predictions.append(_pred_row(tile_id, uid, "no-damage", 0.5))

        if not uids:
            continue

        # Batch predict in chunks
        import torch
        from disaster_bench.models.damage.classifiers import PrePostDiffCNN
        use_9ch = isinstance(model, PrePostDiffCNN)
        preds_all: list[tuple[str, float]] = []
        for i in range(0, len(crops), batch_sz):
            batch = np.stack(crops[i:i + batch_sz])
            t = torch.from_numpy(batch).float().to(device)
            if use_tta:
                probs = _tta4_probs(model, t, use_9ch, tau, device)
            else:
                if use_9ch:
                    t = PrePostDiffCNN.from_six_channel(t)
                with torch.no_grad():
                    logits = model(t)
                    probs = torch.softmax(
                        _apply_tau_norm(logits, tau), dim=1
                    ).cpu().numpy()
            for p in probs:
                idx = int(np.argmax(p))
                preds_all.append((DAMAGE_CLASSES[idx], float(p[idx])))

        for uid, (pred_damage, pred_conf) in zip(uids, preds_all):
            all_predictions.append(_pred_row(tile_id, uid, pred_damage, pred_conf))

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    n = max(len(all_predictions), 1)
    print(f"  T1: {len(all_predictions)} predictions in {elapsed_ms/1000:.1f}s "
          f"({elapsed_ms/n:.1f} ms/instance)")
    return all_predictions


def _pred_row(tile_id: str, uid: str, pred_damage: str, pred_conf: float) -> dict[str, Any]:
    return {
        "tile_id": tile_id,
        "pred_instance_id": uid,
        "matched_gt_uid": None,
        "iou": None,
        "pred_damage": pred_damage,
        "pred_conf": round(pred_conf, 4),
        "track": 1,
        "notes": "",
    }


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_track1_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    import time
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    t0 = time.perf_counter()
    preds = run_track1(index_csv, run_dir, config)
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
