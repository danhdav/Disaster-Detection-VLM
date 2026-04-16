'''
This file runs VLM analysis on a disaster using OpenRouter.
In the frontend, the user presses a button to trigger the VLM analysis for a specific disaster location. It will send a request to the /analyze endpoint. The response will include the VLM's analysis text, the model used, and metadata about the scene and feature analyzed.
'''

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
_PROMPT_PREFIX = (
    "You are a weather analyst specializing in disaster damage assessment. Based on the provided before and after satellite imagery, provide:\n"
    "1) A damage classification.\n"
    "2) A confidence score (0-100).\n"
    "3) A very brief justification (less than a paragraph) based on key visual indicators.\n"
    "4) An immediate response recommendation based on the observed damage.\n\n"
    "Output your analysis after the following within the same JSON format:\n"
)

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

    # print the raw content on the backend for debugging
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
    internal_api_base = os.getenv("INTERNAL_API_BASE", "http://127.0.0.1:5000")
    endpoint = f"{internal_api_base.rstrip('/')}/fire"

    # Lines 146-161 below will normalize the jsonified analysisText for label storage
    normalized_analysis: Any = analysis_text.strip()

    markdown_match = re.search(r"```(?:json)?\s*(.*?)\s*```", analysis_text, flags=re.DOTALL | re.IGNORECASE)
    parse_candidates = [analysis_text.strip()]
    if markdown_match:
        parse_candidates.insert(0, markdown_match.group(1).strip())

    for candidate in parse_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                normalized_analysis = parsed
                break
        except json.JSONDecodeError:
            continue

    document = {
        "documentType": "analysis_result",
        "createdAt": datetime.now(UTC).isoformat(),
        "disasterId": disaster_id,
        "sceneId": scene_id,
        "featureId": feature_id,
        "analysisText": normalized_analysis,
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
        raise RuntimeError("POST /fire did not return an inserted _id for analysis document")
    return str(inserted_id)


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

    try:
        analysis_document_id = persist_analysis_via_fire(
            disaster_id=body.disasterId,
            scene_id=body.sceneId,
            feature_id=body.featureId,
            analysis_text=analysis_text,
            model_name=model_name,
            has_pre_image=bool(pre_data_url),
            has_post_image=bool(post_data_url),
        )
    except requests.RequestException as exc:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "error": f"Persisting analysis via /fire failed: {exc}"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": f"Persisting analysis via /fire failed: {exc}"},
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
            "analysisDocumentId": analysis_document_id,
        },
    }