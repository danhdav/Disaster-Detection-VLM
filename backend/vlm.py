'''
This file runs VLM analysis on a disaster using OpenRouter.
In the frontend, the user presses a button to trigger the VLM analysis for a specific disaster location. It will send a request to the /analyze endpoint. The response will include the VLM's analysis text, the model used, and metadata about the scene and feature analyzed.
'''

from __future__ import annotations

import json
import os
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from parse import find_feature_by_uid
from storage import (
    fetch_scene_label_documents,
    label_phase_from_document,
    presigned_scene_image_urls,
)

app = FastAPI(title="VLM API", version="1.0.0")

# Prompt background information for the VLM
_PROMPT_PREFIX = {
    "You are a weather analyst specializing in disaster damage assessment. Based on the provided before and after satellite imagery, provide:\n"
    "1) A damage classification.\n"
    "2) A confidence score (0-100).\n"
    "3) A very brief justification (less than a paragraph) based on key visual indicators.\n"
    "4) An immediate response recommendation based on the observed damage.\n\n"
    "Output your analysis in the following JSON format:\n"
}

# Call OpenRouter with the prompt and return the response
def openrouter_analysis(
    # Required parameters for the VLM
    feature: dict[str, Any] | None,
    pre_data_url: str | None, # pre-image URL
    post_data_url: str | None, # post-image URL
    disaster_id: str,
    scene_id: str,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = os.getenv("OPENROUTER_VLM_MODEL")
    if not model:
        raise RuntimeError("OPENROUTER_VLM_MODEL is not set")

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

    # Append the image URLs to the prompt
    # Ensure this is correct
    if pre_data_url:
        content.append({"type": "text", "text": "Pre-disaster image:"})
        content.append({"type": "image_url", "image_url": {"url": pre_data_url}})
    if post_data_url:
        content.append({"type": "text", "text": "Post-disaster image:"})
        content.append({"type": "image_url", "image_url": {"url": post_data_url}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
    }

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


@app.post("/analyze", response_model=None)
def analyze_with_openrouter(
    body: AnalyzeRequest
    ) -> dict[str, Any] | JSONResponse:
    model_name = os.getenv("OPENROUTER_VLM_MODEL")

    try:
        pre_doc, post_doc = fetch_scene_label_documents(body.disasterId, body.sceneId)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(exc)},
        )

    if pre_doc is None and post_doc is None:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error": (
                    f"No label documents in MongoDB for disaster '{body.disasterId}' "
                    f"and scene '{body.sceneId}' (expected metadata.img_name "
                    f"{body.sceneId}_pre_disaster.png and/or {body.sceneId}_post_disaster.png)."
                ),
            },
        )

    pre_phase = label_phase_from_document(pre_doc)
    post_phase = label_phase_from_document(post_doc)

    pre_data_url = body.preDataUrl
    post_data_url = body.postDataUrl

    if pre_data_url is None or post_data_url is None:
        try:
            urls = presigned_scene_image_urls(body.sceneId)
            if pre_data_url is None:
                pre_data_url = urls.get("pre_image_url")
            if post_data_url is None:
                post_data_url = urls.get("post_image_url")
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": str(exc)},
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "error": f"S3 imagery: {exc}"},
            )
        
    feature = body.feature
    if feature is None and body.featureId:
        feature = find_feature_by_uid(pre_phase, post_phase, body.featureId)

    try:
        analysis_text = openrouter_analysis(
            feature=feature,
            pre_data_url=pre_data_url,
            post_data_url=post_data_url,
            disaster_id=body.disasterId,
            scene_id=body.sceneId,
        )
    except requests.RequestException as exc:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "error": f"OpenRouter request failed: {exc}"},
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(exc)},
        )

    return {
        "status": "ok",
        "result": {
            "text": analysis_text,
            "model": model_name,
            "sceneId": body.sceneId,
            "disasterId": body.disasterId,
            "featureId": body.featureId,
            "hasPreImage": bool(pre_data_url),
            "hasPostImage": bool(post_data_url),
            "dataSource": "mongodb+s3",
        },
    }