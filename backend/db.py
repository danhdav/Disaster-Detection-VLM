'''
This file contains API endpoints for the dataset (i.e interacting with MongoDB and AWS S3)
To view the documentation UI, visit /docs
'''

from __future__ import annotations

import os
from typing import Any

import requests
from bson.errors import InvalidId
from bson.objectid import ObjectId
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from dataparser import (
    S3_IMAGES_PREFIX,
    bucket_name,
    labels_collection,
    mongo_client,
    mongo_db_name,
    s3_client,
    presigned_scene_image_urls,
    test_mongodb_connection,
    test_s3_connection,
)


app = FastAPI(title="Database & S3 API", version="1.0.0")

fire_labels_collection = labels_collection
analysis_collection_name = os.getenv("MONGO_ANALYSIS_COLLECTION_NAME", "analysis_results")
analysis_collection: Collection | None = (
    mongo_client[mongo_db_name][analysis_collection_name] if mongo_client is not None else None
)


def _get_target_collection(collection: str) -> Collection | None:
    if collection == "labels":
        return fire_labels_collection
    if collection == "analysis":
        return analysis_collection
    return None


def _require_mongo():
    if fire_labels_collection is None:
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


@app.get("/fire", response_model=list[dict[str, Any]])
async def get_fire_labels():
    _require_mongo()
    assert fire_labels_collection is not None
    try:
        labels = list(fire_labels_collection.find())
        print(f"Total labels retrieved: {len(labels)}")
        for label in labels:
            if "_id" in label:
                label["_id"] = str(label["_id"])
        return labels
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# Get a fire label by its metadata.img_name field (e.g. "scene00000123_pre_disaster.png")
@app.get("/fire/search/{img_name}")
async def search_fire_label(img_name: str):
    _require_mongo()
    assert fire_labels_collection is not None
    key = img_name if img_name.lower().endswith(".png") else f"{img_name}.png"
    try:
        label = fire_labels_collection.find_one({"metadata.img_name": key})
        if label:
            label["_id"] = str(label["_id"])
            return label
        raise HTTPException(status_code=404, detail="Label not found")
    except HTTPException:
        raise
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# Add a fire label document
@app.post("/fire")
async def add_fire_label(
    data: dict[str, Any],
    collection: str = Query(default="labels", pattern="^(labels|analysis)$"),
):
    _require_mongo()
    target_collection = _get_target_collection(collection)
    if target_collection is None:
        raise HTTPException(status_code=500, detail=f"Collection '{collection}' is not configured")
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
    try:
        result = target_collection.insert_one(data)
        return {
            "_id": str(result.inserted_id),
            "message": f"Document added successfully to '{collection}'",
            "collection": collection,
        }
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# Deletes a fire label; should be used to remove generated vlm analyses' ground truth labels after server restart
@app.delete("/fire/{label_id}")
async def delete_fire_label(label_id: str):
    _require_mongo()
    assert fire_labels_collection is not None
    try:
        oid = ObjectId(label_id)
    except InvalidId as e:
        raise HTTPException(status_code=400, detail="Invalid label id") from e
    try:
        result = fire_labels_collection.delete_one({"_id": oid})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Label not found")
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


'''
Gets the disaster image for a given scene and phase (pre or post)
scene_id format follows the following example: santa-rosa-wildfire_00000257
'''
@app.get("/image/{scene_id}/{phase}")
async def get_scene_image(scene_id: str, phase: str):
    _require_s3()
    phase_key = phase.strip().lower()
    if phase_key not in {"pre", "post"}:
        raise HTTPException(status_code=400, detail="Phase must be 'pre' or 'post'")

    try:
        urls = presigned_scene_image_urls(scene_id)
        url_key = "pre_image_url" if phase_key == "pre" else "post_image_url"
        target_url = urls.get(url_key)
        if not target_url:
            raise HTTPException(status_code=404, detail="Image URL not found")

        upstream = requests.get(target_url, stream=True, timeout=30)
        upstream.raise_for_status()

        content_type = upstream.headers.get("Content-Type", "image/png")
        return StreamingResponse(
            upstream.iter_content(chunk_size=1024 * 64),
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=300"},
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"S3 proxy request failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/debug/health")
async def check_disasters():
    mongo_ok = test_mongodb_connection()
    s3_ok = test_s3_connection()
    return {
        "status": "ok",
        "mongodb": {"configured": mongo_client is not None, "connected": mongo_ok},
        "s3": {"configured": bool(bucket_name), "connected": s3_ok},
        "s3ImagesPrefix": S3_IMAGES_PREFIX,
    }
