"""Shared MongoDB (label JSON) and S3 (imagery) access for API modules."""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB_NAME", "disaster")
mongo_collection_name = os.getenv("MONGO_COLLECTION_NAME", "fire_labels")

mongo_client: MongoClient | None = MongoClient(mongo_uri) if mongo_uri else None
_db = mongo_client[mongo_db_name] if mongo_client is not None else None
labels_collection: Collection | None = _db[mongo_collection_name] if _db is not None else None

bucket_name = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client("s3") if bucket_name else None

S3_IMAGES_PREFIX = "xview2-test-data/images/"


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


def fetch_scene_label_documents(
    disaster_id: str,
    scene_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load pre/post xView2-style label documents from MongoDB.

    Expects ``metadata.disaster`` and ``metadata.img_name`` like
    ``{scene_id}_pre_disaster.png`` / ``{scene_id}_post_disaster.png``.
    """
    if labels_collection is None:
        raise RuntimeError("MongoDB is not configured (MONGO_URI)")

    pre = labels_collection.find_one(
        {"metadata.disaster": disaster_id, "metadata.img_name": f"{scene_id}_pre_disaster.png"}
    )
    post = labels_collection.find_one(
        {"metadata.disaster": disaster_id, "metadata.img_name": f"{scene_id}_post_disaster.png"}
    )
    return pre, post


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


def presigned_scene_image_urls(scene_base_name: str) -> dict[str, str]:
    """Presigned GET URLs for pre/post PNGs under ``S3_IMAGES_PREFIX``.

    ``scene_base_name`` is the xView2 tile id (e.g. ``guatemala-volcano_00000003``), same as
    ``sceneId`` in the map API.

    Returns keys ``pre_image_url`` and ``post_image_url`` (matches ``/image/{disaster_name}``).
    """
    if s3_client is None or not bucket_name:
        raise RuntimeError("S3 is not configured (S3_BUCKET_NAME)")

    urls: dict[str, str] = {}

    pre_key = f"{S3_IMAGES_PREFIX}{scene_base_name}_pre_disaster.png"
    post_key = f"{S3_IMAGES_PREFIX}{scene_base_name}_post_disaster.png"

    for key in (pre_key, post_key):
        try:
            s3_client.head_object(Bucket=bucket_name, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"Missing S3 object {key}") from exc
            raise
    
    urls["pre_image_url"] = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": pre_key},
        ExpiresIn=3600,
    )
    urls["post_image_url"] = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": post_key},
        ExpiresIn=3600,
    )

    return urls
