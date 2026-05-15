"""
This file contains API endpoints for the dataset (i.e interacting with MongoDB and AWS S3)
To view the documentation UI, visit /docs
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import redis
import requests
from bson.errors import InvalidId
from bson.objectid import ObjectId
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse, StreamingResponse
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from dataparser import (
    S3_IMAGES_PREFIX,
    bucket_name,
    labels_collection,
    mongo_client,
    mongo_db_name,
    presigned_scene_image_urls,
    s3_client,
    test_mongodb_connection,
    test_s3_connection,
)

app = APIRouter(tags=["data"])
logger = logging.getLogger(__name__)

disaster_labels_collection = labels_collection
analysis_collection_name = os.getenv(
    "MONGO_ANALYSIS_COLLECTION_NAME", "analysis_results"
)
analysis_collection: Collection | None = (
    mongo_client[mongo_db_name][analysis_collection_name]
    if mongo_client is not None
    else None
)
_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
redis_client: redis.Redis | None = redis.from_url(_redis_url) if _redis_url else None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


disaster_labels_cache_ttl_seconds = _env_int("LABELS_CACHE_TTL_SECONDS", 300)
LABELS_CACHE_KEY = "disaster-labels:all:v1"


def _get_target_collection(collection: str) -> Collection | None:
    if collection == "labels":
        return disaster_labels_collection
    if collection == "analysis":
        return analysis_collection
    return None


def _require_mongo():
    if disaster_labels_collection is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured (set MONGO_URI and optional MONGO_DB_NAME / MONGO_COLLECTION_NAME).",
        )


def _require_s3():
    if s3_client is None or not bucket_name:
        raise HTTPException(
            status_code=503,
            detail="S3 is not configured (set S3_BUCKET_NAME).",
        )


def _get_cached_disaster_labels() -> list[dict[str, Any]] | None:
    if redis_client is None:
        return None
    try:
        cached = redis_client.get(LABELS_CACHE_KEY)
    except redis.RedisError as exc:
        logger.warning("Redis get failed for %s: %s", LABELS_CACHE_KEY, exc)
        return None
    if cached is None:
        return None
    try:
        payload = json.loads(cached)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, list) else None


def _set_cached_disaster_labels(labels: list[dict[str, Any]]) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(
            LABELS_CACHE_KEY,
            disaster_labels_cache_ttl_seconds,
            json.dumps(labels),
        )
    except redis.RedisError as exc:
        logger.warning("Redis set failed for %s: %s", LABELS_CACHE_KEY, exc)


def _invalidate_disaster_labels_cache() -> None:
    if redis_client is None:
        return
    try:
        redis_client.delete(LABELS_CACHE_KEY)
    except redis.RedisError as exc:
        logger.warning("Redis delete failed for %s: %s", LABELS_CACHE_KEY, exc)


def _openrouter_credits_status() -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {
            "configured": False,
            "connected": False,
            "remainingCredits": None,
            "error": "OPENROUTER_API_KEY is not set",
        }

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
    except requests.RequestException as exc:
        return {
            "configured": True,
            "connected": False,
            "remainingCredits": None,
            "error": str(exc),
        }

    if not response.ok:
        return {
            "configured": True,
            "connected": False,
            "remainingCredits": None,
            "error": f"OpenRouter credits request failed ({response.status_code})",
        }

    body = response.json()
    data = body.get("data", {})
    total_credits = data.get("total_credits")
    total_usage = data.get("total_usage")
    if isinstance(total_credits, (int, float)) and isinstance(
        total_usage, (int, float)
    ):
        remaining_credits: float | None = float(total_credits) - float(total_usage)
    else:
        remaining_credits = None

    return {
        "configured": True,
        "connected": True,
        "remainingCredits": remaining_credits,
        "totalCredits": total_credits,
        "totalUsage": total_usage,
    }


@app.get("/disaster_data", response_model=list[dict[str, Any]])
@app.get("/fire", response_model=list[dict[str, Any]], include_in_schema=False)
async def get_disaster_data():
    _require_mongo()
    assert disaster_labels_collection is not None
    cached = _get_cached_disaster_labels()
    if cached is not None:
        logger.info(
            "GET /disaster_data cache hit (key=%s, records=%d)",
            LABELS_CACHE_KEY,
            len(cached),
        )
        return cached
    logger.info(
        "GET /disaster_data cache miss (key=%s), querying MongoDB",
        LABELS_CACHE_KEY,
    )
    try:
        labels = list(disaster_labels_collection.find())
        logger.info("GET /disaster_data MongoDB hit, retrieved %d records", len(labels))
        for label in labels:
            if "_id" in label:
                label["_id"] = str(label["_id"])
        _set_cached_disaster_labels(labels)
        return labels
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Get a disaster label by its metadata.img_name field (e.g. "scene00000123_pre_disaster.png")
@app.get("/disaster_data/search/{img_name}")
async def search_disaster_label(img_name: str):
    _require_mongo()
    assert disaster_labels_collection is not None
    key = img_name if img_name.lower().endswith(".png") else f"{img_name}.png"
    try:
        label = disaster_labels_collection.find_one({"metadata.img_name": key})
        if label:
            label["_id"] = str(label["_id"])
            return label
        raise HTTPException(status_code=404, detail="Label not found")
    except HTTPException:
        raise
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Add a disaster label document
@app.post("/disaster_data")
async def add_disaster_label(
    data: dict[str, Any],
    collection: str = Query(default="labels", pattern="^(labels|analysis)$"),
):
    _require_mongo()
    target_collection = _get_target_collection(collection)
    if target_collection is None:
        raise HTTPException(
            status_code=500, detail=f"Collection '{collection}' is not configured"
        )
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
    try:
        result = target_collection.insert_one(data)
        if collection == "labels":
            _invalidate_disaster_labels_cache()
        return {
            "_id": str(result.inserted_id),
            "message": f"Document added successfully to '{collection}'",
            "collection": collection,
        }
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Deletes a disaster label; should be used to remove generated vlm analyses' ground truth labels after server restart
@app.delete("/disaster_data/{label_id}")
async def delete_disaster_label(label_id: str):
    _require_mongo()
    assert disaster_labels_collection is not None
    try:
        oid = ObjectId(label_id)
    except InvalidId as e:
        raise HTTPException(status_code=400, detail="Invalid label id") from e
    try:
        result = disaster_labels_collection.delete_one({"_id": oid})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Label not found")
        _invalidate_disaster_labels_cache()
        return {"message": "Label deleted successfully"}
    except HTTPException:
        raise
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# This endpoint is not currently used
@app.get("/image/{disaster_name}")
async def get_image_urls(disaster_name: str):
    _require_s3()
    try:
        return presigned_scene_image_urls(disaster_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


"""
Gets the disaster image for a given scene and phase (pre or post)
scene_id format follows the following example: santa-rosa-wildfire_00000257
"""


@app.get("/image/{scene_id}/{phase}")
async def get_scene_image(scene_id: str, phase: str):
    _require_s3()
    assert s3_client is not None
    assert bucket_name is not None
    phase_key = phase.strip().lower()
    if phase_key not in {"pre", "post"}:
        raise HTTPException(status_code=400, detail="Phase must be 'pre' or 'post'")

    key = f"{S3_IMAGES_PREFIX}{scene_id}_{phase_key}_disaster.png"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        body = response["Body"]
        media_type = response.get("ContentType") or "image/png"
        return StreamingResponse(body.iter_chunks(), media_type=media_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except s3_client.exceptions.NoSuchKey as e:  # type: ignore[attr-defined]
        raise HTTPException(status_code=404, detail=f"Image not found for key: {key}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/debug/health")
async def check_disasters():
    mongo_ok = test_mongodb_connection()
    s3_ok = test_s3_connection()
    openrouter = _openrouter_credits_status()
    return {
        "status": "ok",
        "mongodb": {"configured": mongo_client is not None, "connected": mongo_ok},
        "s3": {"configured": bool(bucket_name), "connected": s3_ok},
        "openrouter": openrouter,
        "s3ImagesPrefix": S3_IMAGES_PREFIX,
    }
