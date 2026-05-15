"""
This file contains API endpoints for the dataset (i.e interacting with MongoDB and AWS S3)
To view the documentation UI, visit /docs
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
from typing import Any

_GEOTRANSFORM_PATH = pathlib.Path(__file__).parent.parent / "xview_geotransforms.json"
_geotransforms: dict = {}
if _GEOTRANSFORM_PATH.is_file():
    with open(_GEOTRANSFORM_PATH) as _f:
        _geotransforms = json.load(_f)


def _geotransform_bounds(
    gt: list[float], width: int = 1024, height: int = 1024
) -> list[float]:
    min_lng = gt[0]
    max_lng = gt[0] + width * gt[1]
    max_lat = gt[3]
    min_lat = gt[3] + height * gt[5]
    return [min_lng, min_lat, max_lng, max_lat]

from bson.errors import InvalidId
from bson.objectid import ObjectId
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
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


app = APIRouter(tags=["data"])

fire_labels_collection = labels_collection
analysis_collection_name = os.getenv(
    "MONGO_ANALYSIS_COLLECTION_NAME", "analysis_results"
)
analysis_collection: Collection | None = (
    mongo_client[mongo_db_name][analysis_collection_name]
    if mongo_client is not None
    else None
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


_allowed_disasters_raw = os.getenv("ALLOWED_DISASTERS", "").strip()
_allowed_disasters: list[str] = (
    [d.strip() for d in _allowed_disasters_raw.split(",") if d.strip()]
    if _allowed_disasters_raw
    else []
)

_eval_scenes_csv = os.getenv("EVAL_SCENES_CSV", "").strip()
_allowed_scenes: list[str] = []
# Maps scene_id -> set of allowed UIDs
_eval_uids_by_scene: dict[str, set[str]] = {}
if _eval_scenes_csv:
    _csv_path = pathlib.Path(__file__).parent / _eval_scenes_csv
    if _csv_path.is_file():
        with open(_csv_path, newline="") as _f:
            for row in csv.DictReader(_f):
                tile = row.get("tile_id", "").strip()
                uid = row.get("uid", "").strip()
                if tile and uid:
                    _eval_uids_by_scene.setdefault(tile, set()).add(uid)
        _allowed_scenes = list(_eval_uids_by_scene.keys())


def _filter_eval_features(doc: dict[str, Any]) -> dict[str, Any]:
    """Keep only building features whose UID is in the eval set for this scene."""
    if not _eval_uids_by_scene:
        return doc
    img_name = doc.get("metadata", {}).get("img_name", "")
    scene_id = img_name.replace("_pre_disaster.png", "").replace("_post_disaster.png", "")
    allowed = _eval_uids_by_scene.get(scene_id)
    if allowed is None:
        return doc
    features = doc.get("features", {})
    lng_lat = [f for f in features.get("lng_lat", []) if f.get("properties", {}).get("uid") in allowed]
    xy = [f for f in features.get("xy", []) if f.get("properties", {}).get("uid") in allowed]
    return {**doc, "features": {**features, "lng_lat": lng_lat, "xy": xy}}


@app.get("/fire", response_model=list[dict[str, Any]])
async def get_fire_labels():
    _require_mongo()
    assert fire_labels_collection is not None
    try:
        query: dict[str, Any] = {}
        if _allowed_disasters:
            query["metadata.disaster"] = {"$in": _allowed_disasters}
        if _allowed_scenes:
            scene_patterns = [f"{s}_pre_disaster.png" for s in _allowed_scenes] + \
                             [f"{s}_post_disaster.png" for s in _allowed_scenes]
            query["metadata.img_name"] = {"$in": scene_patterns}
        labels = list(fire_labels_collection.find(query))
        print(f"Total labels retrieved: {len(labels)}")
        result = []
        for label in labels:
            if "_id" in label:
                label["_id"] = str(label["_id"])
            result.append(_filter_eval_features(label))
        return result
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
        raise HTTPException(
            status_code=500, detail=f"Collection '{collection}' is not configured"
        )
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


"""
Gets the disaster image for a given scene and phase (pre or post)
scene_id format follows the following example: santa-rosa-wildfire_00000257
"""


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

        return RedirectResponse(target_url)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/geotransform/{scene_id}")
def get_geotransform(scene_id: str):
    pre_key  = f"{scene_id}_pre_disaster.png"
    post_key = f"{scene_id}_post_disaster.png"
    pre_gt   = _geotransforms.get(pre_key,  [None])[0]
    post_gt  = _geotransforms.get(post_key, [None])[0]
    return {
        "pre_bounds":  _geotransform_bounds(pre_gt)  if pre_gt  else None,
        "post_bounds": _geotransform_bounds(post_gt) if post_gt else None,
    }


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
