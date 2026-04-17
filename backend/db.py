"""FastAPI endpoints for MongoDB (labels) and S3 (imagery)."""

from __future__ import annotations

from typing import Any

from bson.errors import InvalidId
from bson.objectid import ObjectId
from fastapi import APIRouter, HTTPException
from pymongo.errors import PyMongoError

from storage import (
    S3_IMAGES_PREFIX,
    bucket_name,
    labels_collection,
    mongo_client,
    presigned_scene_image_urls,
    s3_client,
    test_mongodb_connection,
    test_s3_connection,
)

'''
This file contains API endpoints for the dataset (i.e interacting with MongoDB and AWS S3)
To view the documentation UI, visit /docs
'''

router = APIRouter(tags=["data"])

fire_labels_collection = labels_collection


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


@router.get("/fire", response_model=list[dict[str, Any]])
async def get_fire_labels():
    _require_mongo()
    assert fire_labels_collection is not None
    try:
        labels = list(fire_labels_collection.find())
        for label in labels:
            if "_id" in label:
                label["_id"] = str(label["_id"])
        return labels
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# Query the full image name without the .png extension
@router.get("/fire/search/{img_name}")
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


@router.post("/fire")
async def add_fire_label(data: dict[str, Any]):
    _require_mongo()
    assert fire_labels_collection is not None
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
    try:
        result = fire_labels_collection.insert_one(data)
        return {"_id": str(result.inserted_id), "message": "Label added successfully"}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/fire/{label_id}")
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


@router.get("/image/{disaster_name}")
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


@router.get("/debug/health")
async def check_disasters():
    mongo_ok = test_mongodb_connection()
    s3_ok = test_s3_connection()
    return {
        "status": "ok",
        "mongodb": {"configured": mongo_client is not None, "connected": mongo_ok},
        "s3": {"configured": bool(bucket_name), "connected": s3_ok},
        "s3ImagesPrefix": S3_IMAGES_PREFIX,
    }
