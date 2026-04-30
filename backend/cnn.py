"""
CNN damage classification endpoint using PrePostDiffCNN (PPD5 + τ=1.1 + TTA-4).
Ensemble of 3 seeds → macro_F1 = 0.7738 on the wildfire benchmark.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests as _requests
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from dataparser import (
    fetch_scene_label_documents,
    extract_label_data,
    presigned_scene_image_urls,
)

app = APIRouter(tags=["cnn"])

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
MODEL_DIR = Path(__file__).parent / "models" / "ppd5"
TAU = 1.1
CROP_SIZE = 128


def _error(status: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"status": "error", "error": msg})


# ---------------------------------------------------------------------------
# Model loading — loaded once, cached for lifetime of the process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_models() -> tuple[list[Any], int]:
    from disaster_bench.models.damage.classifiers import load_classifier

    models = []
    input_size = CROP_SIZE
    for seed in ("s1", "s2", "s3"):
        ckpt = MODEL_DIR / seed / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
        model, _, ckpt_size = load_classifier(str(ckpt), device="cpu")
        model.eval()
        models.append(model)
        input_size = ckpt_size  # all seeds share the same size; use last
    return models, input_size


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _download_image(url: str) -> np.ndarray:
    resp = _requests.get(url, timeout=30)
    resp.raise_for_status()
    img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def _crop_building(
    img: np.ndarray,
    wkt: str,
    canvas_w: int = 1024,
    canvas_h: int = 1024,
    pad_fraction: float = 0.25,
    min_pad: int = 8,
) -> np.ndarray | None:
    """Crop a building from a full-tile numpy image using pixel-space WKT."""
    from disaster_bench.data.polygons import parse_wkt_polygon

    img_h, img_w = img.shape[:2]
    sx = img_w / canvas_w
    sy = img_h / canvas_h

    try:
        poly = parse_wkt_polygon(wkt)
        if poly is None:
            return None
        xs = [c[0] * sx for c in poly.exterior.coords]
        ys = [c[1] * sy for c in poly.exterior.coords]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(img_w, int(max(xs))), min(img_h, int(max(ys)))
    except Exception:
        return None

    bbox_w, bbox_h = x2 - x1, y2 - y1
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    pad = max(int(pad_fraction * max(bbox_w, bbox_h)), min_pad)
    crop = img[
        max(0, y1 - pad): min(img_h, y2 + pad),
        max(0, x1 - pad): min(img_w, x2 + pad),
    ]
    return crop.copy() if crop.size > 0 else None


def _to_6ch_tensor(pre: np.ndarray, post: np.ndarray, size: int) -> Any:
    """Resize crops and return a (1, 6, H, W) float32 tensor in [0, 1]."""
    import torch

    def _prep(arr: np.ndarray) -> np.ndarray:
        img = PILImage.fromarray(arr.astype(np.uint8)).resize((size, size), PILImage.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    pre_np  = _prep(pre)
    post_np = _prep(post)
    combined = np.concatenate([pre_np, post_np], axis=2)  # (H, W, 6)
    t = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0)  # (1, 6, H, W)
    return t


# ---------------------------------------------------------------------------
# Inference: 3 seeds × TTA-4 rotations, tau-norm
# ---------------------------------------------------------------------------

def _run_inference(pre_crop: np.ndarray, post_crop: np.ndarray, crop_size: int) -> np.ndarray:
    import torch
    from disaster_bench.models.damage.classifiers import PrePostDiffCNN

    models, _ = _load_models()
    six_ch = _to_6ch_tensor(pre_crop, post_crop, crop_size)  # (1, 6, H, W)

    acc = np.zeros(4, dtype=np.float64)
    for model in models:
        seed_acc = np.zeros(4, dtype=np.float64)
        for k in range(4):                          # TTA-4: 0°, 90°, 180°, 270°
            rot = torch.rot90(six_ch, k=k, dims=[2, 3])
            nine_ch = PrePostDiffCNN.from_six_channel(rot)
            with torch.no_grad():
                logits = model(nine_ch) / TAU       # temperature normalisation
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            seed_acc += probs
        acc += seed_acc / 4                         # average over TTA rotations
    return acc / len(models)                        # average over seeds


# ---------------------------------------------------------------------------
# Helper: find a feature's pixel-space WKT (xy features, not lng_lat)
# ---------------------------------------------------------------------------

def _find_xy_wkt(
    pre_doc: dict | None,
    post_doc: dict | None,
    feature_id: str,
) -> str | None:
    """Search the xy (pixel-coordinate) feature list for a given UID."""
    for doc in (post_doc, pre_doc):
        if not doc:
            continue
        for feat in doc.get("features", {}).get("xy", []):
            uid = feat.get("properties", {}).get("uid")
            if uid == feature_id:
                return feat.get("wkt")
    return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class CnnAnalyzeRequest(BaseModel):
    disasterId: str = Field(min_length=1)
    sceneId:    str = Field(min_length=1)
    featureId:  str
    preDataUrl:  str | None = None
    postDataUrl: str | None = None


@app.post("/cnn-analyze", response_model=None)
def cnn_analyze(body: CnnAnalyzeRequest) -> dict[str, Any] | JSONResponse:
    # Pre-check: models exist on disk before doing any network calls
    try:
        models, crop_size = _load_models()
    except FileNotFoundError as exc:
        return _error(503, f"CNN checkpoints not found: {exc}")

    # Fetch scene label documents
    try:
        pre_doc, post_doc = fetch_scene_label_documents(body.disasterId, body.sceneId)
    except RuntimeError as exc:
        return _error(503, str(exc))

    if pre_doc is None and post_doc is None:
        return _error(404, f"No labels for disaster '{body.disasterId}' scene '{body.sceneId}'.")

    # Get pixel-space WKT from xy features
    wkt = _find_xy_wkt(pre_doc, post_doc, body.featureId)
    if not wkt:
        return _error(404, f"No pixel-space polygon found for feature '{body.featureId}'.")

    # Resolve canvas size from metadata
    meta = (post_doc or pre_doc or {}).get("metadata", {})
    canvas_w = int(meta.get("width",  1024))
    canvas_h = int(meta.get("height", 1024))

    # Resolve image URLs
    pre_url, post_url = body.preDataUrl, body.postDataUrl
    if not pre_url or not post_url:
        try:
            urls = presigned_scene_image_urls(body.sceneId)
            pre_url  = pre_url  or urls.get("pre_image_url")
            post_url = post_url or urls.get("post_image_url")
        except (RuntimeError, FileNotFoundError) as exc:
            return _error(503, f"Image URL resolution failed: {exc}")

    if not pre_url or not post_url:
        return _error(404, "Could not resolve pre/post image URLs.")

    # Download full tile images
    try:
        pre_arr  = _download_image(pre_url)
        post_arr = _download_image(post_url)
    except Exception as exc:
        return _error(502, f"Image download failed: {exc}")

    # Crop building footprint
    pre_crop  = _crop_building(pre_arr,  wkt, canvas_w, canvas_h)
    post_crop = _crop_building(post_arr, wkt, canvas_w, canvas_h)
    if pre_crop is None or post_crop is None:
        return _error(400, "Could not crop building from tile — invalid polygon or empty bbox.")

    # Run CNN inference
    try:
        probs = _run_inference(pre_crop, post_crop, crop_size)
    except Exception as exc:
        return _error(500, f"CNN inference failed: {exc}")

    idx           = int(np.argmax(probs))
    damage_level  = DAMAGE_CLASSES[idx]
    conf_score    = float(probs[idx])
    confidence    = "high" if conf_score >= 0.75 else ("medium" if conf_score >= 0.50 else "low")

    return {
        "status": "ok",
        "result": {
            "damageLevel":   damage_level,
            "confidence":    confidence,
            "probabilities": {
                "no-damage":    round(float(probs[0]), 4),
                "minor-damage": round(float(probs[1]), 4),
                "major-damage": round(float(probs[2]), 4),
                "destroyed":    round(float(probs[3]), 4),
            },
            "model":       "PrePostDiffCNN · PPD5 · τ=1.1 · TTA-4 · 3-seed",
            "sceneId":     body.sceneId,
            "disasterId":  body.disasterId,
            "featureId":   body.featureId,
        },
    }
