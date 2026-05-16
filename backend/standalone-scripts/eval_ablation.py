"""
Ablation test: Pre-only vs Post-only vs Both images
for specific scenes and buildings.

Usage:
    uv run python eval_ablation.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

from dataparser import presigned_scene_image_urls
from disaster_bench.models.damage.vlm_prompts import (
    SYSTEM_PROMPT, ungrounded_prompt, grounded_prompt, parse_vlm_response
)
from disaster_bench.data.polygons import parse_wkt_polygon

# ── Config ────────────────────────────────────────────────────────────────────

LABEL_DIR = Path("D:/Aaron/UTD/Spring 26/Capstone Project/Benchmark-Model-xView2/data/test/labels")

SCENES = {
    "santa-rosa-wildfire_00000257": {
        "8dd3e4f8-b4db-4e3f-9a4d-e2260909737f": "destroyed",
        "bf55d633-004a-4667-a303-fa599b83fd59": "no-damage",
        "1257feaa-1e5e-4daa-b007-041001491192": "no-damage",
        "7aff1bd7-7128-4ced-b4cf-1cd6beaa3cb7": "no-damage",
        "75b72a72-0866-488c-b743-78716798ed20": "no-damage",
        "9273bd9b-30c8-434c-9338-4eecdfd56f64": "no-damage",
        "be7024d9-e5fd-4301-b078-5f104369888c": "destroyed",
        "26424e58-fa9c-4535-93de-c6c7ff555026": "destroyed",
        "71fc7e10-e36a-40b2-8a4e-fa60e2147d99": "no-damage",
        "a808a9f7-5824-4c75-ab89-cc38d66d6224": "no-damage",
        "d3e38c4f-a14a-43b0-926a-fc50dcf36abe": "destroyed",
        "f5b25c90-37c8-4983-a126-e711bc093879": "no-damage",
    },
    "socal-fire_00000217": {
        "0e4a8669-58a7-40ea-a1b5-fc380b78b0fe": "destroyed",
        "b45d2ebc-00c5-4e48-81f3-cda9f64ddaa4": "destroyed",
        "26858490-3ff0-4f62-9efd-a5a4b2f6c10e": "destroyed",
        "550cc571-1890-45d9-81cf-523ad7ec99ee": "destroyed",
    },
}

PAD_FRACTION = 0.25
MIN_PAD = 16


# ── Image helpers ─────────────────────────────────────────────────────────────

def download_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))


def load_wkt_map(scene_id: str) -> dict[str, str]:
    """Returns uid -> WKT from the post-disaster label JSON (xy features)."""
    path = LABEL_DIR / f"{scene_id}_post_disaster.json"
    data = json.loads(path.read_text())
    result = {}
    for feat in data.get("features", {}).get("xy", []):
        uid = feat["properties"]["uid"]
        result[uid] = feat.get("wkt", "")
    return result


MIN_OUTPUT_SIZE = 224


def _crop_region(img_arr: np.ndarray, cx1: int, cy1: int, cx2: int, cy2: int) -> Image.Image:
    crop = Image.fromarray(img_arr[cy1:cy2, cx1:cx2].astype(np.uint8))
    cw, ch = crop.size
    if max(cw, ch) < MIN_OUTPUT_SIZE:
        scale = MIN_OUTPUT_SIZE / max(cw, ch)
        crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.LANCZOS)
    return crop


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def compute_polygon_masked_geometry(
    pre_arr: np.ndarray, post_arr: np.ndarray,
    wkt: str, canvas_w: int = 1024, canvas_h: int = 1024,
) -> dict:
    """
    Compute pixel-change metrics ONLY inside the building polygon.
    Uses PIL for rasterization (no cv2 needed).
    """
    img_h, img_w = pre_arr.shape[:2]
    sx, sy = img_w / canvas_w, img_h / canvas_h

    poly = parse_wkt_polygon(wkt)
    if poly is None:
        return {}

    coords_px = [(c[0] * sx, c[1] * sy) for c in poly.exterior.coords]
    xs = [c[0] for c in coords_px]
    ys = [c[1] for c in coords_px]
    x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
    x2, y2 = min(img_w, int(max(xs))), min(img_h, int(max(ys)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return {}

    pre_crop  = pre_arr[y1:y2, x1:x2].astype(np.float32) / 255.0
    post_crop = post_arr[y1:y2, x1:x2].astype(np.float32) / 255.0
    crop_h, crop_w = pre_crop.shape[:2]

    # Rasterize polygon with PIL (no cv2 dependency)
    local_pts = [(int(round(x - x1)), int(round(y - y1))) for x, y in coords_px]
    mask_img = Image.new("L", (crop_w, crop_h), 0)
    ImageDraw.Draw(mask_img).polygon(local_pts, fill=1)
    mask = np.array(mask_img, dtype=np.uint8)

    n_poly = int(mask.sum())
    if n_poly == 0:
        return {}

    diff = np.abs(post_crop - pre_crop).mean(axis=2)
    poly_diff    = diff[mask > 0]
    change_score = round(float(poly_diff.mean()), 4)
    pct_changed  = round(float((poly_diff > 0.10).mean() * 100), 2)

    return {"change_score": change_score, "pct_changed": pct_changed, "polygon_px": n_poly}


def crop_building(img_arr: np.ndarray, wkt: str,
                  canvas_w: int = 1024, canvas_h: int = 1024) -> str | None:
    """Crop building, draw red outline, upscale if tiny, return base64 PNG data URL."""
    img_h, img_w = img_arr.shape[:2]
    sx = img_w / canvas_w
    sy = img_h / canvas_h

    poly = parse_wkt_polygon(wkt)
    if poly is None:
        return None
    coords_px = [(c[0] * sx, c[1] * sy) for c in poly.exterior.coords]
    xs = [c[0] for c in coords_px]
    ys = [c[1] for c in coords_px]
    x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
    x2, y2 = min(img_w, int(max(xs))), min(img_h, int(max(ys)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None

    pad = max(int(PAD_FRACTION * max(x2 - x1, y2 - y1)), MIN_PAD)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

    crop = _crop_region(img_arr, cx1, cy1, cx2, cy2)
    draw = ImageDraw.Draw(crop)
    # Scale outline coords to match the (possibly upscaled) crop
    scale_x = crop.width  / (cx2 - cx1)
    scale_y = crop.height / (cy2 - cy1)
    outline = [(int((x - cx1) * scale_x), int((y - cy1) * scale_y)) for x, y in coords_px]
    draw.polygon(outline, outline=(255, 0, 0))
    draw.polygon(outline, outline=(255, 80, 80))

    return _to_b64(crop)


def crop_masked_diff(pre_arr: np.ndarray, post_arr: np.ndarray, wkt: str,
                     canvas_w: int = 1024, canvas_h: int = 1024,
                     amplify: int = 6) -> str | None:
    """
    Pixel-difference heatmap cropped to the building region, with everything
    OUTSIDE the polygon blacked out. Shows only the change at the building footprint.
    """
    img_h, img_w = pre_arr.shape[:2]
    sx = img_w / canvas_w
    sy = img_h / canvas_h

    poly = parse_wkt_polygon(wkt)
    if poly is None:
        return None
    coords_px = [(c[0] * sx, c[1] * sy) for c in poly.exterior.coords]
    xs = [c[0] for c in coords_px]
    ys = [c[1] for c in coords_px]
    x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
    x2, y2 = min(img_w, int(max(xs))), min(img_h, int(max(ys)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None

    pad = max(int(PAD_FRACTION * max(x2 - x1, y2 - y1)), MIN_PAD)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

    # Compute amplified diff
    pre_crop  = pre_arr[cy1:cy2, cx1:cx2].astype(np.int32)
    post_crop = post_arr[cy1:cy2, cx1:cx2].astype(np.int32)
    diff = np.clip(np.abs(post_crop - pre_crop) * amplify, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff)

    # Black out everything outside the polygon
    mask = Image.new("L", diff_img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    poly_local = [(x - cx1, y - cy1) for x, y in coords_px]
    mask_draw.polygon(poly_local, fill=255)
    black = Image.new("RGB", diff_img.size, (0, 0, 0))
    diff_masked = Image.composite(diff_img, black, mask)

    # Draw polygon outline in white so the VLM can see the boundary
    draw = ImageDraw.Draw(diff_masked)
    draw.polygon(poly_local, outline=(255, 255, 255))

    # Upscale if tiny
    cw, ch = diff_masked.size
    if max(cw, ch) < MIN_OUTPUT_SIZE:
        scale = MIN_OUTPUT_SIZE / max(cw, ch)
        diff_masked = diff_masked.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.LANCZOS)

    return _to_b64(diff_masked)


# ── VLM call ──────────────────────────────────────────────────────────────────

def _is_claude(model: str) -> bool:
    return model.startswith("claude-") or model.startswith("anthropic/claude")


_DAMAGE_SEVERITY = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def call_vlm(pre_b64: str | None, post_b64: str | None, model: str,
             diff_b64: str | None = None, temperature: float = 0.0,
             geometry_features: dict | None = None) -> dict:
    if geometry_features:
        prompt = grounded_prompt(geometry_features, diff_overlay=bool(diff_b64))
    else:
        prompt = ungrounded_prompt(full_tile=False, diff_overlay=bool(diff_b64))
    t0 = time.perf_counter()

    if _is_claude(model):
        # ── Anthropic API ──────────────────────────────────────────────────
        import anthropic
        api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("claude_api_key not set in .env")

        def _img_block(b64_url: str) -> dict:
            data = b64_url.split(",", 1)[1] if "," in b64_url else b64_url
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

        content: list[dict] = []
        if pre_b64:
            content.append(_img_block(pre_b64))
            content.append({"type": "text", "text": "Pre-disaster image above."})
        if post_b64:
            content.append(_img_block(post_b64))
            content.append({"type": "text", "text": "Post-disaster image above."})
        if diff_b64:
            content.append(_img_block(diff_b64))
            content.append({"type": "text", "text": "Change heatmap above (brighter = more change)."})
        content.append({"type": "text", "text": prompt})

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text if resp.content else ""

    else:
        # ── OpenRouter API ─────────────────────────────────────────────────
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        content_or: list[dict] = [{"type": "text", "text": prompt}]
        if pre_b64:
            content_or.append({"type": "text", "text": "Pre-disaster image:"})
            content_or.append({"type": "image_url", "image_url": {"url": pre_b64}})
        if post_b64:
            content_or.append({"type": "text", "text": "Post-disaster image:"})
            content_or.append({"type": "image_url", "image_url": {"url": post_b64}})
        if diff_b64:
            content_or.append({"type": "text", "text": "Change heatmap (brighter = more change):"})
            content_or.append({"type": "image_url", "image_url": {"url": diff_b64}})

        resp_or = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_or},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp_or.raise_for_status()
        body = resp_or.json()
        raw = body["choices"][0]["message"]["content"]
        if isinstance(raw, list):
            raw = "\n".join(p.get("text", "") for p in raw if isinstance(p, dict))

    latency_ms = (time.perf_counter() - t0) * 1000
    parsed = parse_vlm_response(str(raw))
    parsed["latency_ms"] = round(latency_ms, 0)
    return parsed


def call_vlm_ensemble(pre_b64: str | None, post_b64: str | None, model: str,
                      n_votes: int = 3, diff_b64: str | None = None,
                      temperature: float = 0.5,
                      geometry_features: dict | None = None) -> tuple[str, list[str]]:
    """
    Call VLM n_votes times and return (majority_label, all_vote_labels).
    Tie-break: prefer less severe class (no-damage > minor > major > destroyed).
    """
    votes: list[str] = []
    for _ in range(n_votes):
        try:
            parsed = call_vlm(pre_b64, post_b64, model, diff_b64=diff_b64,
                              temperature=temperature, geometry_features=geometry_features)
            label = parsed.get("damage_level", "?") or "?"
        except Exception:
            label = "ERR"
        votes.append(label)

    # Count votes, tie-break toward less severe
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1

    winner = max(
        counts,
        key=lambda cls: (counts[cls], -_DAMAGE_SEVERITY.index(cls) if cls in _DAMAGE_SEVERITY else -99),
    )
    return winner, votes


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_override: str | None = None, n_votes: int = 1) -> None:
    model = model_override or os.getenv("OPENROUTER_VLM_MODEL", "")
    if not model:
        print("ERROR: pass --model or set OPENROUTER_VLM_MODEL in .env")
        sys.exit(1)

    ensemble = n_votes > 1
    ens_label = f"  votes={n_votes} temp=0.5" if ensemble else "  votes=1 temp=0"
    print(f"\n{'=' * 70}")
    print(f"  Ablation Test: Pre-only vs Post-only vs Both  |  model={model}")
    print(f"  {ens_label}")
    print(f"{'=' * 70}")

    # Per-condition accuracy tracking
    cond_correct = {"pre": 0, "post": 0, "both": 0}
    cond_total   = {"pre": 0, "post": 0, "both": 0}

    for scene_id, buildings in SCENES.items():
        print(f"\n{'─' * 70}")
        print(f"  Scene: {scene_id}")
        print(f"{'─' * 70}")

        # Download full scene images (once per scene)
        print("  Fetching S3 URLs...")
        try:
            urls = presigned_scene_image_urls(scene_id)
            pre_url  = urls.get("pre_image_url")
            post_url = urls.get("post_image_url")
        except Exception as exc:
            print(f"  ERROR fetching URLs: {exc}")
            continue

        print("  Downloading images...")
        try:
            pre_arr  = download_image(pre_url)  if pre_url  else None
            post_arr = download_image(post_url) if post_url else None
        except Exception as exc:
            print(f"  ERROR downloading: {exc}")
            continue

        # Load pixel-space WKTs
        wkt_map = load_wkt_map(scene_id)

        print(f"\n  {'UID':8s}  {'GT':12s}  {'Pre-only':15s}  {'Post-only':15s}  {'Both':15s}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*15}  {'─'*15}  {'─'*15}")

        for uid, gt in buildings.items():
            wkt = wkt_map.get(uid)
            if not wkt:
                print(f"  {uid[:8]}  {gt:12s}  WKT missing — skipped")
                continue

            # Generate crops
            pre_b64  = crop_building(pre_arr,  wkt) if pre_arr  is not None else None
            post_b64 = crop_building(post_arr, wkt) if post_arr is not None else None

            diff_b64: str | None = None  # no diff overlay

            # Polygon-masked change metrics — used for threshold-based grounding
            if pre_arr is not None and post_arr is not None:
                geo = compute_polygon_masked_geometry(pre_arr, post_arr, wkt)
            else:
                geo = {}

            # Inject geometry into "both" condition only when poly_cs is very low
            # (< 0.13 means change inside polygon is minimal → likely no structural damage)
            LOW_CHANGE_THRESHOLD = 0.13
            poly_cs = geo.get("change_score", 1.0)
            geo_for_both = geo if (isinstance(poly_cs, float) and poly_cs < LOW_CHANGE_THRESHOLD) else None

            results: dict[str, str] = {}
            vote_detail: dict[str, list[str]] = {}
            for condition, p, q, d in [
                ("pre",  pre_b64, None,     None),
                ("post", None,    post_b64, None),
                ("both", pre_b64, post_b64, diff_b64),
            ]:
                if p is None and q is None:
                    results[condition] = "no-image"
                    continue
                geo_arg = geo_for_both if condition == "both" else None
                try:
                    if ensemble:
                        pred, votes = call_vlm_ensemble(p, q, model, n_votes=n_votes,
                                                        diff_b64=d, geometry_features=geo_arg)
                        vote_detail[condition] = votes
                    else:
                        parsed = call_vlm(p, q, model, diff_b64=d, geometry_features=geo_arg)
                        pred = parsed.get("damage_level", "?") or "?"
                except Exception as exc:
                    pred = "ERR"
                results[condition] = pred
                cond_total[condition] += 1
                cond_correct[condition] += int(pred == gt)

            def fmt(pred: str, gt: str) -> str:
                ok = "✓" if pred == gt else "✗"
                return f"{ok} {pred[:13]:13s}"

            geo_str = ""
            if geo:
                cs  = geo.get("change_score")
                pct = geo.get("pct_changed")
                npx = geo.get("polygon_px")
                if isinstance(cs, float) and isinstance(pct, float):
                    grounded_marker = " [GROUNDED]" if geo_for_both else ""
                    geo_str = f"  poly_cs={cs:.3f} poly_pct={pct:.1f}% px={npx}{grounded_marker}"

            line = f"  {uid[:8]}  {gt:12s}  {fmt(results['pre'], gt)}  {fmt(results['post'], gt)}  {fmt(results['both'], gt)}"
            if ensemble and vote_detail:
                both_votes = vote_detail.get("both", [])
                abbrev = [v[:3] for v in both_votes]
                line += f"  [{', '.join(abbrev)}]"
            line += geo_str
            print(line)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for cond in ("pre", "post", "both"):
        t = cond_total[cond]
        c = cond_correct[cond]
        pct = f"{c/t:.0%}" if t else "n/a"
        print(f"  {cond:8s}: {c}/{t} = {pct}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="Model to use. Claude: claude-sonnet-4-6, claude-opus-4-6. "
                        "OpenRouter: openai/gpt-4.1. Default: OPENROUTER_VLM_MODEL from .env")
    p.add_argument("--votes", type=int, default=1,
                   help="Ensemble size: call VLM N times per building and take majority vote. "
                        "Use --votes 3 or --votes 5. Default: 1 (no ensemble, temperature=0).")
    args = p.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    run(model_override=args.model, n_votes=args.votes)
