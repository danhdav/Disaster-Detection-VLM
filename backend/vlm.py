import os
from typing import Any
import boto3
from fastapi import FastAPI, HTTPException
from flask import json
from pydantic import BaseModel, Field
from flask import jsonify, request
import requests

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

app = FastAPI()

prompt_text = (
    "You are a disaster damage analyst. Compare pre-disaster and post-disaster satellite images "
    "for one structure and provide:\n"
    "1) damage classification\n"
    "2) confidence (0-100)\n"
    "3) brief justification\n"
    "4) immediate response recommendation\n\n"
)

def openrouter_analyze(
    feature: dict[str, Any] | None,
    pre_data_url: str | None,
    post_data_url: str | None,
    disaster_id: str,
    scene_id: str,
) -> str:
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    if not model_name:
        raise RuntimeError("OPENROUTER_MODEL is not set")

    feature_properties = (feature or {}).get("properties", {})
    feature_wkt = (feature or {}).get("wkt")

    prompt_text_info = (
        f"Disaster: {disaster_id}\n"
        f"Scene: {scene_id}\n"
        f"Feature metadata: {json.dumps(feature_properties)}\n"
        f"Feature geometry (WKT): {feature_wkt}\n"
    )

    prompt = prompt_text + prompt_text_info

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if pre_data_url:
        content.append({"type": "text", "text": "Pre-disaster image:"})
        content.append({"type": "image_url", "image_url": {"url": pre_data_url}})
    if post_data_url:
        content.append({"type": "text", "text": "Post-disaster image:"})
        content.append({"type": "image_url", "image_url": {"url": post_data_url}})

    payload = {
        "model": model_name,
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
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()

    content_value = body["choices"][0]["message"]["content"]
    if isinstance(content_value, str):
        return content_value
    if isinstance(content_value, list):
        parts: list[str] = []
        for item in content_value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(content_value)

@app.post("/analyze")
async def analyze_with_openrouter(payload: dict[str, Any]):
    disaster_id = payload.get("disasterId")
    scene_id = payload.get("sceneId")
    feature_id = payload.get("featureId")

    if not disaster_id or not scene_id:
        raise HTTPException(status_code=400, detail="disasterId and sceneId are required")

    try:
        analysis_text = openrouter_analyze(
            feature=None,
            pre_data_url=None,
            post_data_url=None,
            disaster_id=disaster_id,
            scene_id=scene_id,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "ok",
        "result": {
            "text": analysis_text,
            "model": model_name,
            "sceneId": scene_id,
            "disasterId": disaster_id,
            "featureId": feature_id,
        },
    }