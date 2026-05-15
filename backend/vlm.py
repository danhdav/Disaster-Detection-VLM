"""
This file runs VLM analysis on a disaster using OpenRouter.
In the frontend, the user will press the "VLM analysis" button for a specific disaster location. It will send a request to the /analyze endpoint.
The response will include the VLM's analysis text, the model used, and metadata about the scene and feature analyzed.
"""

from __future__ import annotations

import base64
import io as _io
import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import requests
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from PIL import Image as PILImage, ImageDraw
from pydantic import BaseModel, Field

from dataparser import (
    fetch_scene_label_documents,
    find_feature_by_uid,
    extract_label_data,
    presigned_scene_image_urls,
)
from disaster_bench.models.damage.vlm_prompts import (
    get_system_prompt,
    ungrounded_prompt,
    parse_vlm_response,
    parse_l2_supervisor_response,
)
from eval_vlm import call_l2_hierarchy, load_scene_overrides

# Load Round-6 scene-specific corrective prompts once at startup
_SCENE_OVERRIDES: dict[str, str] = {}
try:
    _SCENE_OVERRIDES = load_scene_overrides()
except Exception as _e:
    print(f"[vlm] Warning: could not load scene overrides: {_e}")

app = APIRouter(tags=["vlm"])


# ---------------------------------------------------------------------------
# Building crop helpers — produce base64 data-URL crops with red outline
# so the VLM receives exactly what the benchmark pipeline produces.
# ---------------------------------------------------------------------------

def _download_arr(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = PILImage.open(_io.BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def _find_xy_wkt(pre_doc: dict | None, post_doc: dict | None, feature_id: str) -> str | None:
    """Find the pixel-space WKT for a feature from the xy feature list."""
    for doc in (post_doc, pre_doc):
        if not doc:
            continue
        for feat in doc.get("features", {}).get("xy", []):
            if feat.get("properties", {}).get("uid") == feature_id:
                return feat.get("wkt")
    return None


def _crop_building_b64(
    img_arr: np.ndarray,
    wkt: str,
    canvas_w: int = 1024,
    canvas_h: int = 1024,
    pad_fraction: float = 0.25,
    min_pad: int = 16,
    min_output_size: int = 224,
) -> str | None:
    """
    Crop a building from a full-tile numpy image, draw a red polygon outline
    around the building footprint, and return a base64 PNG data URL.
    Small crops are upscaled to min_output_size so the VLM can see detail.
    Returns None on any error so callers can fall back gracefully.
    """
    from disaster_bench.data.polygons import parse_wkt_polygon

    img_h, img_w = img_arr.shape[:2]
    sx = img_w / canvas_w
    sy = img_h / canvas_h

    try:
        poly = parse_wkt_polygon(wkt)
        if poly is None:
            return None
        coords_px = [(c[0] * sx, c[1] * sy) for c in poly.exterior.coords]
        xs = [c[0] for c in coords_px]
        ys = [c[1] for c in coords_px]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(img_w, int(max(xs))), min(img_h, int(max(ys)))
    except Exception:
        return None

    bbox_w, bbox_h = x2 - x1, y2 - y1
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    pad = max(int(pad_fraction * max(bbox_w, bbox_h)), min_pad)
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)

    crop_img = PILImage.fromarray(img_arr[cy1:cy2, cx1:cx2].astype(np.uint8))

    # Draw the building polygon outline in red
    draw = ImageDraw.Draw(crop_img)
    outline_coords = [(x - cx1, y - cy1) for x, y in coords_px]
    draw.polygon(outline_coords, outline=(255, 0, 0))
    draw.polygon(outline_coords, outline=(255, 80, 80))  # second pass for visibility

    # Upscale small crops so the VLM can see detail (tiny buildings are ~50px)
    cw, ch = crop_img.size
    if max(cw, ch) < min_output_size:
        scale = min_output_size / max(cw, ch)
        new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
        crop_img = crop_img.resize((new_w, new_h), PILImage.LANCZOS)

    buf = _io.BytesIO()
    crop_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _humanize_evidence(raw: str) -> list[str]:
    """Strip internal pipeline jargon and return short, user-readable bullet points.

    Removes L1/L2 references, indicator names, scores, and boilerplate prefixes
    like "Visual inspection confirms" — keeping only the core visual observation.
    """
    import re

    # Patterns that mark a whole sentence as purely technical (drop entirely)
    _PURELY_TECHNICAL = [
        r"\bL[12]\b.*\bindicators?\b",
        r"\bL[12]\b.*\bscore[s]?\b",
        r"=(?:yes|no|unclear)",
        r"\boverwhelm",
        r"\banti.bias\b",
        r"\bgate\b.*\b(?:pass|fail)",
        r"\bfall[s]?\s+through\b",
        r"\bStep\s+\d\b",
        r"\bdecision\s+tree\b",
        r"\bL[12]\s+path\b",
        r"\bexplicit\s+['\"]?yes['\"]?\s+or\s+['\"]?no['\"]?\b",
        r"\bnot\s+unclear\b",
        r"\ball\s+three\s+are\b",
        r"\bsatisfying\s+the\b",
    ]

    def _clean_sentence(s: str) -> str:
        # Remove "— all three …" and similar dashes with technical content
        s = re.sub(r"\s*[—–-]+\s*(?:all\s+three|and\s+the|the\s+L[12])[^.]*\.?", "", s, flags=re.IGNORECASE)
        # Remove trailing ", and the L2 … ." clause (mixed sentences)
        s = re.sub(r"[,;]?\s*and\s+(?:the\s+)?L[12][^.]*\.?", "", s, flags=re.IGNORECASE)
        # Remove "the L1/L2 path …" clauses
        s = re.sub(r"[,;]?\s*(?:the\s+)?L[12]\s+path[^,\.]*[,\.]?", "", s, flags=re.IGNORECASE)
        # Remove decision tree references
        s = re.sub(r"[,;]?\s*(?:the\s+)?strict\s+\w+\s+decision\s+tree[^,\.]*[,\.]?", "", s, flags=re.IGNORECASE)
        # Remove leading boilerplate prefixes
        s = re.sub(r"^[Vv]isual inspection confirms(?: that)?\s+", "", s)
        s = re.sub(
            r"^(?:The\s+)?(?:post-disaster\s+)?[Ii]mage(?:ry)?\s+(?:shows?|reveals?|confirms?)\s+(?:that\s+)?",
            "", s
        )
        # Remove indicator=value pairs and their parenthesised lists
        s = re.sub(r"\([^)]*=(?:yes|no|unclear)[^)]*\)", "", s)
        s = re.sub(r"\b\w+=(?:yes|no|unclear)\b", "", s)
        # Remove any remaining standalone L2/L1 clauses
        s = re.sub(r",?\s*(?:the\s+)?L[12][^,\.]*[,\.]?", "", s, flags=re.IGNORECASE)
        # Clean up whitespace and punctuation artifacts
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"^\s*[,;\.]+\s*", "", s)
        s = s.strip().rstrip(".,;")
        # Drop if still purely technical after cleaning
        if any(re.search(p, s, re.IGNORECASE) for p in _PURELY_TECHNICAL):
            return ""
        return s

    # Split into sentences, clean each, discard empties
    sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
    results = []
    for sent in sentences:
        cleaned = _clean_sentence(sent.strip())
        if cleaned and len(cleaned) > 10:
            # Further split on semicolons within a sentence
            for part in cleaned.split(";"):
                part = part.strip().rstrip(".,;")
                if len(part) > 8:
                    results.append(part)

    return results if results else [raw.strip()]


def _error_response(status_code: int, error: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"status": "error", "error": error}
    )


def _resolve_scene_image_urls(
    scene_id: str,
    pre_data_url: str | None,
    post_data_url: str | None,
) -> tuple[str | None, str | None]:
    if pre_data_url is not None and post_data_url is not None:
        return pre_data_url, post_data_url

    urls = presigned_scene_image_urls(scene_id)
    resolved_pre = (
        pre_data_url if pre_data_url is not None else urls.get("pre_image_url")
    )
    resolved_post = (
        post_data_url if post_data_url is not None else urls.get("post_image_url")
    )
    return resolved_pre, resolved_post


def _append_image_content(
    content: list[dict[str, Any]], label: str, image_url: str | None
) -> None:
    if not image_url:
        return
    content.append({"type": "text", "text": f"{label} image:"})
    content.append({"type": "image_url", "image_url": {"url": image_url}})


def _make_masked_diff_b64(
    pre_arr: np.ndarray,
    post_arr: np.ndarray,
    wkt: str,
    canvas_w: int = 1024,
    canvas_h: int = 1024,
    amplify: int = 6,
) -> str | None:
    """
    Polygon-masked pixel-difference heatmap cropped to the building region.
    Everything outside the building polygon is blacked out, so the VLM sees
    only structural change at the footprint rather than surrounding vegetation/ash.
    Returns a base64 PNG data URL, or None on any error.
    """
    from disaster_bench.data.polygons import parse_wkt_polygon

    try:
        img_h, img_w = pre_arr.shape[:2]
        sx, sy = img_w / canvas_w, img_h / canvas_h

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

        pad = max(int(0.25 * max(x2 - x1, y2 - y1)), 16)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

        pre_crop  = pre_arr[cy1:cy2, cx1:cx2].astype(np.int32)
        post_crop = post_arr[cy1:cy2, cx1:cx2].astype(np.int32)
        diff = np.clip(np.abs(post_crop - pre_crop) * amplify, 0, 255).astype(np.uint8)
        diff_img = PILImage.fromarray(diff)

        # Black out everything outside the building polygon
        poly_local = [(x - cx1, y - cy1) for x, y in coords_px]
        mask = PILImage.new("L", diff_img.size, 0)
        ImageDraw.Draw(mask).polygon(poly_local, fill=255)
        black = PILImage.new("RGB", diff_img.size, (0, 0, 0))
        diff_masked = PILImage.composite(diff_img, black, mask)

        # White outline so the VLM can see the polygon boundary
        ImageDraw.Draw(diff_masked).polygon(poly_local, outline=(255, 255, 255))

        # Upscale if tiny
        cw, ch = diff_masked.size
        if max(cw, ch) < 224:
            scale = 224 / max(cw, ch)
            diff_masked = diff_masked.resize(
                (max(1, int(cw * scale)), max(1, int(ch * scale))), PILImage.LANCZOS
            )

        buf = _io.BytesIO()
        diff_masked.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# Call OpenRouter with the prompt and return the response
def openrouter_analysis(
    # Required parameters for the VLM
    feature: dict[str, Any] | None,
    pre_data_url: str | None,  # pre-image URL
    post_data_url: str | None,  # post-image URL
    disaster_id: str,
    scene_id: str,
    model_name: str,
    full_tile: bool = False,
    diff_b64: str | None = None,  # polygon-masked diff heatmap, computed by caller
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    # User message: benchmark structured prompt + context metadata + images
    user_prompt = (
        ungrounded_prompt(full_tile=full_tile, diff_overlay=bool(diff_b64))
        + f"\nDisaster: {disaster_id} | Scene: {scene_id}\n"
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

    # Append pre/post image content blocks to the prompt.
    _append_image_content(content, "Pre-disaster", pre_data_url)
    _append_image_content(content, "Post-disaster", post_data_url)
    if diff_b64:
        _append_image_content(content, "Change heatmap (brighter = more change, polygon-masked)", diff_b64)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": get_system_prompt(scene_id)},
            {"role": "user",   "content": content},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    # print the raw prompt content on the backend for debugging
    print("OpenRouter VLM analysis request content:")
    for item in content:
        if item["type"] == "text":
            print(f"TEXT: {item['text']}")
        elif item["type"] == "image_url":
            print(f"IMAGE_URL: {item['image_url']['url']}")
        else:
            print(f"OTHER CONTENT: {item}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()

    # Openrouter has a big response schema, so this parses the content only
    # If the content is an entire string, return, otherwise extract the text parts and concatenate them
    content_value = body["choices"][0]["message"]["content"]
    if isinstance(content_value, list):
        parts: list[str] = []
        for item in content_value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(content_value) if not isinstance(content_value, str) else content_value


class AnalyzeRequest(BaseModel):
    disasterId: str = Field(min_length=1)
    sceneId: str = Field(min_length=1)
    featureId: str | None = None
    preDataUrl: str | None = None
    postDataUrl: str | None = None
    feature: dict[str, Any] | None = None


def persist_analysis_via_fire(
    *,
    disaster_id: str,
    scene_id: str,
    feature_id: str | None,
    analysis_text: str,
    model_name: str | None,
    has_pre_image: bool,
    has_post_image: bool,
) -> str:
    endpoint = (
        f"{os.getenv('INTERNAL_API_BASE', 'http://127.0.0.1:8000').rstrip('/')}/disaster_data"
    )

    # Create the VLM analysis result document structure
    document = {
        "documentType": "analysis_result",
        "createdAt": datetime.now(UTC).isoformat(),
        "disasterId": disaster_id,
        "sceneId": scene_id,
        "featureId": feature_id,
        "analysisText": analysis_text,
        "model": model_name,
        "hasPreImage": has_pre_image,
        "hasPostImage": has_post_image,
        "source": "analyze_endpoint",
    }

    response = requests.post(
        endpoint,
        params={"collection": "analysis"},
        json=document,
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    inserted_id = body.get("_id")

    if not inserted_id:
        raise RuntimeError(
            "POST /fire did not return an inserted VLM analysis document"
        )
    return str(inserted_id)


# Run VLM analysis
@app.post("/analyze", response_model=None)
def analyze_with_openrouter(body: AnalyzeRequest) -> dict[str, Any] | JSONResponse:
    disaster_id = body.disasterId
    scene_id = body.sceneId
    feature_id = body.featureId
    pre_data_url = body.preDataUrl
    post_data_url = body.postDataUrl
    feature = body.feature

    model_name = os.getenv("OPENROUTER_VLM_MODEL")
    if not model_name:
        return _error_response(status_code=500, error="OPENROUTER_VLM_MODEL is not set")

    try:
        pre_doc, post_doc = fetch_scene_label_documents(disaster_id, scene_id)
    except RuntimeError as exc:
        return _error_response(status_code=503, error=str(exc))

    if pre_doc is None and post_doc is None:
        return _error_response(
            status_code=404,
            error=(
                f"No label documents in MongoDB for disaster '{disaster_id}' "
                f"and scene '{scene_id}' (expected metadata.img_name "
                f"{scene_id}_pre_disaster.png and/or {scene_id}_post_disaster.png)."
            ),
        )

    pre_phase = extract_label_data(pre_doc)
    post_phase = extract_label_data(post_doc)

    try:
        pre_data_url, post_data_url = _resolve_scene_image_urls(
            scene_id,
            pre_data_url,
            post_data_url,
        )
    except RuntimeError as exc:
        return _error_response(status_code=503, error=str(exc))
    except FileNotFoundError as exc:
        return _error_response(status_code=404, error=f"S3 imagery: {exc}")

    if feature is None and feature_id:
        feature = find_feature_by_uid(pre_phase, post_phase, feature_id)

    # Crop the specific building and draw a red polygon outline so the VLM
    # receives the same zoomed-in annotated view the benchmark pipeline uses.
    # The xy features carry pixel-space WKT; lng_lat features carry geographic
    # coords and cannot be used for pixel cropping.
    crop_succeeded = False
    if feature_id and pre_data_url and post_data_url:
        xy_wkt = _find_xy_wkt(pre_doc, post_doc, feature_id)
        if xy_wkt:
            try:
                meta     = (pre_doc or post_doc or {}).get("metadata", {})
                canvas_w = int(meta.get("width",  1024))
                canvas_h = int(meta.get("height", 1024))
                pre_arr  = _download_arr(pre_data_url)
                post_arr = _download_arr(post_data_url)
                pre_b64  = _crop_building_b64(pre_arr,  xy_wkt, canvas_w, canvas_h)
                post_b64 = _crop_building_b64(post_arr, xy_wkt, canvas_w, canvas_h)
                if pre_b64 and post_b64:
                    pre_data_url  = pre_b64
                    post_data_url = post_b64
                    crop_succeeded = True
                    print("VLM: sending building crops with red outline.")
                else:
                    print("VLM: crop produced None — falling back to full tile.")
                    pre_arr = post_arr = None
            except Exception as exc:
                print(f"VLM crop failed (full tile fallback): {exc}")
        else:
            print("VLM: no xy WKT found — sending full tile images.")

    # Compute polygon-masked diff heatmap when a crop was successfully made
    diff_b64: str | None = None
    if crop_succeeded and pre_arr is not None and post_arr is not None and xy_wkt:
        diff_b64 = _make_masked_diff_b64(pre_arr, post_arr, xy_wkt, canvas_w, canvas_h)
        if diff_b64:
            print("VLM: polygon-masked diff heatmap computed.")

    try:
        hierarchy_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        scene_description = _SCENE_OVERRIDES.get(scene_id, "")
        raw_result, _latency, _tokens = call_l2_hierarchy(
            pre_data_url,
            post_data_url,
            hierarchy_model,
            tile_id=scene_id,
            scene_description=scene_description,
            use_stage0=True,
        )
        analysis_text = raw_result
    except Exception as exc:
        return _error_response(status_code=500, error=f"Hierarchy analysis failed: {exc}")

    try:
        analysis_document_id = persist_analysis_via_fire(
            disaster_id=disaster_id,
            scene_id=scene_id,
            feature_id=feature_id,
            analysis_text=analysis_text,
            model_name=model_name,
            has_pre_image=bool(pre_data_url),
            has_post_image=bool(post_data_url),
        )
    except requests.RequestException as exc:
        return _error_response(
            status_code=502, error=f"Persisting analysis via /fire failed: {exc}"
        )
    except Exception as exc:
        return _error_response(
            status_code=500, error=f"Persisting analysis via /fire failed: {exc}"
        )

    parsed = parse_l2_supervisor_response(analysis_text)

    # If parse_vlm_response fell back to its error dict, "raw_response" is present.
    # Return a 502 so the caller knows the VLM output was unparseable.
    if "raw_response" in parsed:
        return _error_response(
            status_code=502,
            error=(
                f"VLM returned non-JSON output (parse error: {parsed.get('parse_error')}). "
                f"Raw response: {parsed.get('raw_response')}"
            ),
        )

    # Build a clean, human-readable evidence list
    raw_evidence = parsed.get("key_evidence", "") or ""
    evidence_list = _humanize_evidence(raw_evidence)

    return {
        "status": "ok",
        "result": {
            "damageLevel": parsed.get("damage_level"),
            "confidence": parsed.get("confidence"),
            "keyEvidence": evidence_list,
            "model": model_name,
            "sceneId": scene_id,
            "disasterId": disaster_id,
            "featureId": feature_id,
            "hasPreImage": bool(pre_data_url),
            "hasPostImage": bool(post_data_url),
            "dataSource": "mongodb+s3",
            "analysisDocumentId": analysis_document_id,
            "rawResponse": analysis_text,
            "parseError": parsed.get("parse_error", ""),
        },
    }
