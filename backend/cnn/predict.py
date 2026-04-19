"""
Inference helpers for the 6-channel SixChannelCNN damage classifier.
Matches the predict pipeline from Benchmark-Model-xView2.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import SixChannelCNN, DAMAGE_CLASSES, DAMAGE_SEVERITY


# Resize to 128x128 and normalize per-channel to [0, 1] (no ImageNet stats —
# the Benchmark normalises by dividing by 255 only, keeping values in [0,1]).
_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),   # scales uint8 [0,255] -> float [0,1]
])


def load_model(weights_path: str) -> SixChannelCNN:
    """
    Load a SixChannelCNN from a .pt weights file.
    Accepts either a raw state-dict or a Benchmark-style checkpoint dict
    (with 'model_state_dict' key).  If the weights are incompatible (e.g.
    stale ResNet-18 placeholder) the model is returned untrained with a warning.
    """
    model = SixChannelCNN(num_classes=len(DAMAGE_CLASSES))
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"[INFO] Loaded weights from {weights_path}")
    except Exception as e:
        print(f"[WARN] Could not load weights ({e}); using untrained model")
    model.eval()
    return model


def predict_damage(model: SixChannelCNN, pre_bytes: bytes, post_bytes: bytes) -> dict:
    """
    Run damage inference on a pre/post image pair (raw bytes).

    Returns:
        pred_label  : one of DAMAGE_CLASSES
        severity    : one of DAMAGE_SEVERITY
        confidence  : softmax probability of the top class
        margin      : top-1 prob minus top-2 prob
        scores      : {class_name: probability, ...}
    """
    pre  = Image.open(io.BytesIO(pre_bytes)).convert("RGB")
    post = Image.open(io.BytesIO(post_bytes)).convert("RGB")

    pre_t  = _IMG_TRANSFORM(pre)   # (3, 128, 128)
    post_t = _IMG_TRANSFORM(post)  # (3, 128, 128)
    x = torch.cat([pre_t, post_t], dim=0).unsqueeze(0)  # (1, 6, 128, 128)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_idx     = int(torch.argmax(logits, dim=1).item())
    sorted_probs = sorted(probs, reverse=True)

    return {
        "pred_label": DAMAGE_CLASSES[pred_idx],
        "pred_idx":   pred_idx,
        "severity":   DAMAGE_SEVERITY[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "margin":     round(sorted_probs[0] - sorted_probs[1], 4),
        "scores":     {cls: round(p, 4) for cls, p in zip(DAMAGE_CLASSES, probs)},
    }


def compute_geometry_context(pre_bytes: bytes, post_bytes: bytes) -> dict:
    """
    Compute pixel-level change metrics between pre and post images.

    Returns:
        change_score  : mean absolute per-pixel difference (0–1)
        pct_changed   : percentage of pixels with > 10% change
        ssim_dissim   : 1 − SSIM dissimilarity score (if skimage available)
    """
    pre  = np.array(Image.open(io.BytesIO(pre_bytes)).convert("RGB").resize((128, 128))).astype(np.float32) / 255.0
    post = np.array(Image.open(io.BytesIO(post_bytes)).convert("RGB").resize((128, 128))).astype(np.float32) / 255.0

    diff         = np.abs(post - pre)
    change_score = round(float(diff.mean()), 4)
    pct_changed  = round(float((diff.mean(axis=2) > 0.10).mean() * 100), 2)

    result: dict = {"change_score": change_score, "pct_changed": pct_changed}

    try:
        from skimage.metrics import structural_similarity as ssim
        s = ssim(pre, post, data_range=1.0, channel_axis=2)
        result["ssim_dissim"] = round(float(1.0 - s), 4)
    except ImportError:
        pass

    return result


# Legacy single-image entry point (kept for any callers that still use it)
def predict_image(model: SixChannelCNN, image_bytes: bytes) -> dict:
    """
    Single-image fallback: duplicates the image as both pre and post.
    Prefer predict_damage() for proper pre/post comparisons.
    """
    return predict_damage(model, image_bytes, image_bytes)
