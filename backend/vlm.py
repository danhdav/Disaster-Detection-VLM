'''
This file runs VLM analysis on a disaster using OpenRouter.
In the frontend, the user will press the "VLM analysis" button for a specific disaster location. It will send a request to the /analyze endpoint.
The response will include the VLM's analysis text, the model used, and metadata about the scene and feature analyzed.
'''

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.dataparser import (
    fetch_scene_label_documents,
    find_feature_by_uid,
    extract_label_data,
    presigned_scene_image_urls,
)

app = FastAPI(title="VLM API", version="1.0.0")

# Prompt background information for the VLM
_PROMPT_PREFIX = (
    "You are a weather analyst specializing in disaster damage assessment. Based on the provided before and after satellite imagery, provide the following eight pieces of information:\n"
    "1) A damage classification.\n"
    "2) A confidence score (0-100).\n"
    "3) A very brief justification (less than a paragraph) based on key visual indicators.\n"
    "4) An immediate response recommendation based on the observed damage.\n"
)


def _error_response(status_code: int, error: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "error": error})


def _resolve_scene_image_urls(
    scene_id: str,
    pre_data_url: str | None,
    post_data_url: str | None,
) -> tuple[str | None, str | None]:
    if pre_data_url is not None and post_data_url is not None:
        return pre_data_url, post_data_url

    urls = presigned_scene_image_urls(scene_id)
    resolved_pre = pre_data_url if pre_data_url is not None else urls.get("pre_image_url")
    resolved_post = post_data_url if post_data_url is not None else urls.get("post_image_url")
    return resolved_pre, resolved_post


def _append_image_content(content: list[dict[str, Any]], label: str, image_url: str | None) -> None:
    if not image_url:
        return
    content.append({"type": "text", "text": f"{label} image:"})
    content.append({"type": "image_url", "image_url": {"url": image_url}})

# Call OpenRouter with the prompt and return the response
def openrouter_analysis(
    # Required parameters for the VLM
    feature: dict[str, Any] | None,
    pre_data_url: str | None, # pre-image URL
    post_data_url: str | None, # post-image URL
    disaster_id: str,
    scene_id: str,
    model_name: str,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    feature_properties = (feature or {}).get("properties", {})
    feature_wkt = (feature or {}).get("wkt")

    # Add the requested output format to the prompt
    prompt = (
        _PROMPT_PREFIX
        + f"Disaster: {disaster_id}\n"
        + f"Scene: {scene_id}\n"
        + f"Feature metadata: {json.dumps(feature_properties)}\n"
        + f"Feature geometry (WKT): {feature_wkt}\n"
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    # Append pre/post image content blocks to the prompt.
    _append_image_content(content, "Pre-disaster", pre_data_url)
    _append_image_content(content, "Post-disaster", post_data_url)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
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
    if isinstance(content_value, str):
        return content_value
    if isinstance(content_value, list):
        parts: list[str] = []
        for item in content_value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(body)

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
    endpoint = f"{os.getenv('INTERNAL_API_BASE', 'http://127.0.0.1:5000').rstrip('/')}/fire"

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
        raise RuntimeError("POST /fire did not return an inserted VLM analysis document")
    return str(inserted_id)


# Run VLM analysis
@app.post("/analyze", response_model=None)
def analyze_with_openrouter(
    body: AnalyzeRequest
    ) -> dict[str, Any] | JSONResponse:
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

    try:
        analysis_text = openrouter_analysis(
            feature=feature,
            pre_data_url=pre_data_url,
            post_data_url=post_data_url,
            disaster_id=disaster_id,
            scene_id=scene_id,
            model_name=model_name,
        )
    except requests.RequestException as exc:
        return _error_response(status_code=502, error=f"OpenRouter request failed: {exc}")
    except Exception as exc:  # pragma: no cover - runtime guard
        return _error_response(status_code=500, error=str(exc))

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
        return _error_response(status_code=502, error=f"Persisting analysis via /fire failed: {exc}")
    except Exception as exc:
        return _error_response(status_code=500, error=f"Persisting analysis via /fire failed: {exc}")

    return {
        "status": "ok",
        "result": {
            "text": analysis_text,
            "model": model_name,
            "sceneId": scene_id,
            "disasterId": disaster_id,
            "featureId": feature_id,
            "hasPreImage": bool(pre_data_url),
            "hasPostImage": bool(post_data_url),
            "dataSource": "mongodb+s3",
            "analysisDocumentId": analysis_document_id,
        },
    }