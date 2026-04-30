"""
Track 2B (Learned) — Siamese U-Net pixel-wise damage map → polygon aggregation.
Ref §2B learned models: "Siamese U-Net pixel-wise damage severity {0..4}"
                        "Polygon aggregation from a learned damage map"

Each oracle building crop is passed through the trained Siamese U-Net, which
outputs per-pixel severity logits (5 classes: 0=bg, 1-4=damage). The logits are
aggregated over the full crop to produce a building-level damage label.

This is the learned counterpart to Track 2B heuristic (pixel severity map).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from disaster_bench.data.io import (
    get_buildings_from_label,
    load_label_json,
    read_index_csv,
)
from disaster_bench.data.dataset import load_crop_pair
from disaster_bench.models.damage.siamese_unet import (
    load_siamese_unet,
    aggregate_damage_from_map,
)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_track2b_learned(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Config keys:
      model_ckpt     : path to siamese_unet checkpoint (default: models/siamese_unet/best.pt)
      device         : cuda | cpu
      input_size     : crop resize dimension in pixels (default: 256)
      inference_batch: crops per GPU forward pass (default: 16)
      crops_dir      : path to oracle crops (default: data/processed/crops_oracle)
    """
    config    = config or {}
    device    = config.get("device", "cuda" if _cuda_available() else "cpu")
    ckpt      = config.get("model_ckpt", "models/siamese_unet/best.pt")
    crop_size = int(config.get("input_size", 256))
    batch_sz  = int(config.get("inference_batch", 16))
    crops_dir = Path(config.get("crops_dir", "data/processed/crops_oracle"))

    import torch

    model, post_only = load_siamese_unet(ckpt, device=device)
    model.eval()
    print(f"  T2B-L: siamese_unet loaded ({device}) post_only={post_only} "
          f"input_size={crop_size} batch={batch_sz}")

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

        # Collect crops for this tile
        uids:  list[str]        = []
        crops: list[np.ndarray] = []
        for b in buildings:
            uid  = b["uid"]
            pre  = crops_dir / tile_id / uid / "pre_bbox.png"
            post = crops_dir / tile_id / uid / "post_bbox.png"
            if pre.is_file() and post.is_file():
                uids.append(uid)
                crops.append(load_crop_pair(pre, post, crop_size))
            else:
                all_predictions.append(_pred_row(tile_id, uid, "no-damage", 0.0))

        if not uids:
            continue

        # Batch forward pass
        for i in range(0, len(uids), batch_sz):
            batch_crops = crops[i : i + batch_sz]
            batch_uids  = uids[i : i + batch_sz]
            t = torch.from_numpy(np.stack(batch_crops)).float().to(device)
            with torch.no_grad():
                logits = model(t)   # (B, 5, H, W)
            logits_np = logits.cpu().numpy()

            for uid, sev_logits in zip(batch_uids, logits_np):
                # All-ones mask: entire crop is the building (matches training distribution)
                mask = np.ones(sev_logits.shape[1:], dtype=np.uint8)
                label, conf = aggregate_damage_from_map(sev_logits, mask)
                all_predictions.append(_pred_row(tile_id, uid, label, conf))

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    n = max(len(all_predictions), 1)
    print(f"  T2B-L: {len(all_predictions)} predictions in {elapsed_ms/1000:.1f}s "
          f"({elapsed_ms/n:.1f} ms/instance)")
    return all_predictions


def _pred_row(tile_id: str, uid: str, pred_damage: str, pred_conf: float) -> dict[str, Any]:
    return {
        "tile_id":          tile_id,
        "pred_instance_id": uid,
        "matched_gt_uid":   None,
        "iou":              None,
        "pred_damage":      pred_damage,
        "pred_conf":        round(pred_conf, 4),
        "track":            "2b-learned",
        "notes":            "",
    }


def run_track2b_learned_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    t0 = time.perf_counter()
    preds = run_track2b_learned(index_csv, run_dir, config)
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
