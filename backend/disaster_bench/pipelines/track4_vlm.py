"""
Track 4 — VLM baselines (cloud / local / NVIDIA Build).
Ref §3: VLM benchmark, images-only vs tool-grounded.
Ref §4: Hallucination/consistency comparison between grounded and ungrounded.

Each building crop pair is sent to the configured VLM.
Grounded mode adds geometry features as 'tool context' to the prompt.
Latency and cost estimates are recorded per instance.

Image inputs per building (loaded from oracle crops directory):
  pre_bbox.png  / post_bbox.png  : padded context crop with polygon outlined in red
  pre_masked.png/ post_masked.png: same padded region, non-building pixels blacked out

Cloud VLMs receive all 4 images when masked crops exist.
Local VLMs receive a side-by-side composite (pre_masked | post_masked).
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
from disaster_bench.pipelines.track2a import compute_change_score, _ssim_map


def _load_vlm_img(path: Path, size: int) -> np.ndarray | None:
    """
    Load an image for VLM input.
    size > 0: resize to (size, size) with LANCZOS resampling.
    size == 0: return at natural resolution.
    Returns None if the file does not exist.
    """
    if not path.is_file():
        return None
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGB")
    if size > 0:
        img = img.resize((size, size), PILImage.LANCZOS)
    return np.array(img)


def _build_geometry_context(
    mask: np.ndarray,
    diff_map: np.ndarray,
    ssim_map_: np.ndarray | None,
    building_area: float,
) -> dict[str, float]:
    """Assemble geometry feature dict for grounded VLM prompt."""
    pct_ch = 100.0 * float(np.logical_and(mask > 0, diff_map > 0.1).sum()) / max(building_area, 1)
    score  = compute_change_score(mask, diff_map, ssim_map_)
    result = {
        "change_score":  round(score, 4),
        "pct_changed":   round(pct_ch, 2),
        "area_px":       int(building_area),
    }
    if ssim_map_ is not None:
        result["ssim_dissim"] = round(
            float(ssim_map_[mask > 0].mean()) if (mask > 0).any() else 0.0, 4
        )
    return result


def run_track4_vlm(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run VLM damage classification via GitHub Models (ungrounded or grounded).
    Config keys:
      model           : GitHub Models namespaced model ID (e.g. "openai/gpt-4o")
      grounded        : True/False (default False)
      crops_dir       : path to oracle crops
      vlm_input_size  : resize dimension for VLM inputs in pixels (0 = natural, default 512)
      max_crops       : limit for rate-limit-aware testing (default unlimited)
    """
    config    = config or {}
    model     = config.get("model", "openai/gpt-4o")
    grounded  = bool(config.get("grounded", False))
    crops_dir = Path(config.get("crops_dir", "data/processed/crops_oracle"))
    vlm_size  = int(config.get("vlm_input_size", 512))   # 0 = natural resolution
    max_crops = config.get("max_crops", None)

    from disaster_bench.models.damage.vlm_wrapper import get_vlm
    vlm = get_vlm(model=model)
    print(f"  T4-VLM: model={vlm.model_name} "
          f"grounded={grounded} vlm_size={'natural' if vlm_size == 0 else vlm_size}")

    rows = read_index_csv(index_csv)
    all_predictions: list[dict[str, Any]] = []
    total_tokens = 0
    t_start = time.perf_counter()
    n_done  = 0

    for row in rows:
        if max_crops is not None and n_done >= max_crops:
            break
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

        # Build tile-level change maps for grounded mode
        json_w, json_h = get_label_canvas_size(label_data)
        if json_w <= 0:
            json_w, json_h = 1024, 1024

        pre_img = post_img = None
        diff_map = ssim_map_ = None
        if grounded:
            try:
                pre_img  = load_image(pre_path)  if pre_path  else None
                post_img = load_image(post_path) if post_path else None
            except Exception:
                pass
            if (pre_img is not None and post_img is not None
                    and pre_img.shape[:2] == post_img.shape[:2]):
                img_h, img_w = post_img.shape[:2]
                sx, sy = scale_factors(img_w, img_h, json_w, json_h)
                pre_gray  = pre_img.mean(axis=2).astype(np.float32) / 255.0
                post_gray = post_img.mean(axis=2).astype(np.float32) / 255.0
                diff_map  = np.abs(post_gray - pre_gray)
                try:
                    ssim_map_ = _ssim_map(pre_gray * 255, post_gray * 255)
                except Exception:
                    ssim_map_ = None
            else:
                img_h, img_w = json_h, json_w
                sx = sy = 1.0
        else:
            img_h, img_w = json_h, json_w
            sx = sy = 1.0

        for b in buildings:
            if max_crops is not None and n_done >= max_crops:
                break

            uid = b["uid"]
            crop_dir = crops_dir / tile_id / uid

            # Load context crops (padded, polygon outlined in red)
            pre_crop  = _load_vlm_img(crop_dir / "pre_bbox.png",  vlm_size)
            post_crop = _load_vlm_img(crop_dir / "post_bbox.png", vlm_size)
            if pre_crop is None or post_crop is None:
                continue  # crop not generated; skip

            # Load masked crops (building isolated, background black) — optional
            pre_masked  = _load_vlm_img(crop_dir / "pre_masked.png",  vlm_size)
            post_masked = _load_vlm_img(crop_dir / "post_masked.png", vlm_size)

            # Geometry for grounded mode
            geom = None
            if grounded and diff_map is not None:
                poly, _ = parse_and_scale_building(b["wkt"], sx, sy)
                if poly is not None:
                    mask = rasterize_polygon(poly, (img_h, img_w))
                    ba   = float(mask_area(mask))
                    if ba > 0:
                        geom = _build_geometry_context(mask, diff_map, ssim_map_, ba)

            # VLM call — passes all 4 images when masked crops are present
            result = vlm.classify(
                pre_crop, post_crop,
                pre_masked=pre_masked,
                post_masked=post_masked,
                grounded=grounded,
                geometry=geom,
            )
            total_tokens += result.get("tokens_used", 0)
            n_done += 1

            all_predictions.append({
                "tile_id":          tile_id,
                "pred_instance_id": uid,
                "matched_gt_uid":   None,
                "iou":              None,
                "pred_damage":      result.get("damage_level", "no-damage"),
                "pred_conf":        result.get("confidence",   0.0),
                "track":            f"4-vlm-{result.get('mode', 'ungrounded')}",
                "notes":            result.get("reasoning", "")[:120],
                "ev_latency_ms":    result.get("latency_ms", 0),
                "ev_tokens":        result.get("tokens_used", 0),
                "ev_parse_error":   result.get("parse_error", ""),
            })

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    n = max(len(all_predictions), 1)
    print(f"  T4-VLM: {len(all_predictions)} predictions in {elapsed_ms/1000:.1f}s "
          f"({elapsed_ms/n:.1f} ms/inst) | tokens={total_tokens}")
    return all_predictions


def run_track4_vlm_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    preds = run_track4_vlm(index_csv, run_dir, config)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_predictions_csv(preds, Path(run_dir) / "predictions.csv")
    write_metrics_json({
        "macro_f1": None, "per_class_f1": {}, "coverage": None,
        "note": "Run eval-run to compute F1 metrics",
    }, Path(run_dir) / "metrics.json")
