'''
This file contains functions to fetch and parse through MongoDB and AWS S3 data
'''

from __future__ import annotations

import os
from typing import Any
import boto3 # S3 client
from botocore.exceptions import ClientError
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


# Load local environment variables
mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB_NAME", "disaster")
mongo_collection_name = os.getenv("MONGO_COLLECTION_NAME", "fire_labels")

mongo_client: MongoClient | None = MongoClient(mongo_uri) if mongo_uri else None
_db = mongo_client[mongo_db_name] if mongo_client is not None else None
labels_collection: Collection | None = _db[mongo_collection_name] if _db is not None else None

bucket_name = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client("s3") if bucket_name else None
S3_IMAGES_PREFIX = "xview2-test-data/images/"
_SCENE_PHASES = ("pre", "post")

# Test connections at load time
def test_mongodb_connection() -> bool:
    if mongo_client is None:
        return False
    try:
        mongo_client.admin.command("ping")
        return True
    except PyMongoError:
        return False

def test_s3_connection() -> bool:
    if s3_client is None or not bucket_name:
        return False
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except Exception:
        return False


# Get pre and post label documents for a given scene
def fetch_scene_label_documents(
    disaster_id: str,
    scene_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    
    if labels_collection is None:
        raise RuntimeError("MongoDB is not configured (MONGO_URI)")

    docs: dict[str, dict[str, Any] | None] = {}
    for phase in _SCENE_PHASES:
        docs[phase] = labels_collection.find_one(
            {
                "metadata.disaster": disaster_id,
                "metadata.img_name": f"{scene_id}_{phase}_disaster.png",
            }
        )

    return docs["pre"], docs["post"]


def label_phase_from_document(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Shape one Mongo label document like a Flask route phase (metadata, features, imgName)."""
    if doc is None:
        return None
    metadata = doc.get("metadata") or {}
    return {
        "metadata": metadata,
        "features": doc.get("features") or {},
        "imgName": metadata.get("img_name"),
    }


# Get the S3 image URLs for a given scene
def presigned_scene_image_urls(scene_base_name: str) -> dict[str, str]:

    if s3_client is None or not bucket_name:
        raise RuntimeError("S3 is not configured (S3_BUCKET_NAME)")

    keys_by_phase = {
        phase: f"{S3_IMAGES_PREFIX}{scene_base_name}_{phase}_disaster.png" for phase in _SCENE_PHASES
    }

    for key in keys_by_phase.values():
        try:
            s3_client.head_object(Bucket=bucket_name, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"Missing S3 object {key}") from exc
            raise

    return {
        f"{phase}_image_url": s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": key},
            ExpiresIn=3600,
        )
        for phase, key in keys_by_phase.items()
    }


# Parse the wkt node
def parse_polygon_wkt_bounds(wkt: str) -> list[float] | None:
    if not wkt.startswith("POLYGON"):
        return None

    content = wkt.replace("POLYGON", "", 1).strip()
    if not (content.startswith("((") and content.endswith("))")):
        return None

    points = content[2:-2].split(",")
    min_lng = float("inf")
    min_lat = float("inf")
    max_lng = float("-inf")
    max_lat = float("-inf")

    for point in points:
        parts = point.strip().split()
        if len(parts) < 2:
            continue
        lng = float(parts[0])
        lat = float(parts[1])
        min_lng = min(min_lng, lng)
        min_lat = min(min_lat, lat)
        max_lng = max(max_lng, lng)
        max_lat = max(max_lat, lat)

    if min_lng == float("inf"):
        return None
    return [min_lng, min_lat, max_lng, max_lat]


# Merge 2 bounding boxes
def merge_bounds(base: list[float] | None, nxt: list[float] | None) -> list[float] | None:
    if base is None:
        return nxt
    if nxt is None:
        return base
    return [
        min(base[0], nxt[0]),
        min(base[1], nxt[1]),
        max(base[2], nxt[2]),
        max(base[3], nxt[3]),
    ]


# Get the bounding boxes for all features in a document
def extract_label_bounds(label_data: dict[str, Any]) -> list[float] | None:
    bounds: list[float] | None = None
    lng_lat_features = label_data.get("features", {}).get("lng_lat", [])
    for feature in lng_lat_features:
        feature_bounds = parse_polygon_wkt_bounds(feature.get("wkt", ""))
        bounds = merge_bounds(bounds, feature_bounds)
    return bounds


# Fetch a feature (scene) given its uid
def find_feature_by_uid(
    pre_phase: dict[str, Any] | None,
    post_phase: dict[str, Any] | None,
    feature_id: str,
) -> dict[str, Any] | None:
    for phase_data in (pre_phase, post_phase):
        if not phase_data:
            continue
        for feature in phase_data.get("features", {}).get("lng_lat", []):
            uid = feature.get("properties", {}).get("uid")
            if uid == feature_id:
                return feature
    return None