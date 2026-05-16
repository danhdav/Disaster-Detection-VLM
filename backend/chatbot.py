"""
This file handles all chatbot-related API endpoints (i.e messages and session handling).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4  # for generating unique session IDs

import chromadb
import redis
import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field  # use for data validation error checking
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

import re as _re

from dataparser import mongo_client, mongo_db_name
from search import route_prompt_to_query

_NEARBY_KEYWORDS = {"street", "road", "avenue", "nearby", "neighborhood", "address", "location", "where", "area"}
_UID_PATTERN = _re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", _re.IGNORECASE)
# Strip frontend-injected context blocks before routing so the UID in [System Context: ...] doesn't
# trigger uid_lookup on every message when the user has a map feature selected.
_SYSTEM_CONTEXT_PATTERN = _re.compile(r"\n\n\[System Context:.*?\]", _re.DOTALL)
# Extracts the scene ID from the injected context, e.g. "inside the image hurricane-matthew_00000010"
_SCENE_ID_FROM_CONTEXT = _re.compile(r"inside the image ([^\s.\]]+)", _re.IGNORECASE)


def _lookup_uid(uid: str) -> str | None:
    """Look up a building UID in MongoDB and return a summary string."""
    try:
        col_name = os.getenv("MONGO_COLLECTION_NAME", "dataset")
        col = mongo_client[mongo_db_name][col_name]
        doc = col.find_one({"features.xy.properties.uid": uid})
        if not doc:
            return None

        img_name = doc["metadata"].get("img_name", "unknown")
        disaster_type = doc["metadata"].get("disaster_type", "unknown")

        subtype = "unknown"
        for b in doc.get("features", {}).get("xy", []):
            if b["properties"].get("uid") == uid:
                subtype = b["properties"].get("subtype", "unknown")
                break

        # Extract centroid from WKT polygon
        lat, lon = 0.0, 0.0
        for entry in doc.get("features", {}).get("lng_lat", []):
            if entry.get("properties", {}).get("uid") == uid:
                coords = _re.findall(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", entry.get("wkt", ""))
                if coords:
                    lon = sum(float(c[0]) for c in coords) / len(coords)
                    lat = sum(float(c[1]) for c in coords) / len(coords)
                break

        address = _reverse_geocode(lat, lon) if lat and lon else None
        lines = [
            f"Building UID: {uid}",
            f"Scene: {img_name}",
            f"Disaster: {disaster_type}",
            f"Damage Level: {subtype}",
            f"Coordinates: Lat {lat:.6f}, Lon {lon:.6f}",
        ]
        if address:
            lines.append(f"Nearest Address: {address}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("UID lookup failed for %s: %s", uid, e)
        return None


def _reverse_geocode(lat: float, lon: float) -> str | None:
    """Convert lat/lon to a human-readable address using Nominatim (free, no API key)."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 18},
            headers={"User-Agent": "SurgeDisasterAssessmentApp/1.0"},
            timeout=5,
        )
        data = r.json()
        return data.get("display_name")
    except Exception as e:
        logger.warning("Reverse geocode failed: %s", e)
        return None

# Setup basic logging to see errors in your terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# RAG: CHROMADB INITIALIZATION
# ==========================================
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./xview2_vector_db")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    disaster_collection = chroma_client.get_or_create_collection(
        name="disaster_assessments"
    )
    articles_collection = chroma_client.get_or_create_collection(
        name="disaster_articles"
    )
    logger.info("ChromaDB initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    disaster_collection = None
    articles_collection = None

chat_sessions_collection_name = "chat_sessions"
chat_sessions_collection: Collection | None = (
    mongo_client[mongo_db_name][chat_sessions_collection_name]
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


chat_sessions_cache_ttl_seconds = _env_int("CHAT_SESSIONS_CACHE_TTL_SECONDS", 300)
CHAT_HISTORY_ALL_CACHE_KEY = "chat-sessions:all:v1"
CHAT_HISTORY_SESSION_CACHE_PREFIX = "chat-sessions:session:"
SUPPORTED_METADATA_FILTER_KEYS = {"disaster_type", "source_type", "lat", "lon"}
chat_query_memory: dict[str, dict[str, Any]] = {}


def _session_cache_key(session_id: str) -> str:
    return f"{CHAT_HISTORY_SESSION_CACHE_PREFIX}{session_id}:v1"


def _cache_set_json(key: str, payload: Any) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(
            key,
            chat_sessions_cache_ttl_seconds,
            json.dumps(payload),
        )
    except redis.RedisError as exc:
        logger.warning("Redis set failed for %s: %s", key, exc)


def _cache_get_json(key: str) -> Any | None:
    if redis_client is None:
        return None
    try:
        cached = redis_client.get(key)
    except redis.RedisError as exc:
        logger.warning("Redis get failed for %s: %s", key, exc)
        return None
    if cached is None:
        return None
    try:
        return json.loads(cached)
    except json.JSONDecodeError:
        logger.warning("Redis cached JSON decode failed for %s", key)
        return None


def _cache_delete_key(key: str) -> None:
    if redis_client is None:
        return
    try:
        redis_client.delete(key)
    except redis.RedisError as exc:
        logger.warning("Redis delete failed for %s: %s", key, exc)


def _invalidate_all_chat_history_cache() -> None:
    _cache_delete_key(CHAT_HISTORY_ALL_CACHE_KEY)


def _invalidate_session_history_cache(session_id: str) -> None:
    _cache_delete_key(_session_cache_key(session_id))


def _invalidate_all_session_history_keys() -> None:
    if redis_client is None:
        return
    pattern = f"{CHAT_HISTORY_SESSION_CACHE_PREFIX}*"
    try:
        keys = list(redis_client.scan_iter(match=pattern))
        if keys:
            redis_client.delete(*keys)
    except redis.RedisError as exc:
        logger.warning("Redis key scan/delete failed for pattern %s: %s", pattern, exc)


def _clear_chat_state_on_startup() -> None:
    cleared_query_memories = len(chat_query_memory)
    chat_query_memory.clear()
    _invalidate_all_chat_history_cache()
    _invalidate_all_session_history_keys()
    if chat_sessions_collection is None:
        logger.warning(
            "Chat startup reset skipped Mongo cleanup (chat_sessions collection unavailable)."
        )
        logger.info(
            "Chat startup reset complete (query_memories=%d)",
            cleared_query_memories,
        )
        return
    try:
        result = chat_sessions_collection.delete_many({})
    except PyMongoError as exc:
        logger.error("Failed to clear chat_sessions on startup: %s", exc)
    else:
        logger.info(
            "Chat startup reset complete (mongo_sessions=%d, query_memories=%d)",
            result.deleted_count,
            cleared_query_memories,
        )


def _auto_sync_chroma() -> None:
    """Populate ChromaDB collections from MongoDB and S3 if they are empty."""
    try:
        if disaster_collection is not None and disaster_collection.count() == 0:
            logger.info("disaster_assessments is empty — running MongoDB sync...")
            import re as _re
            from dataparser import labels_collection as _labels_col, mongo_db_name as _db_name
            from dataparser import mongo_client as _mc
            _col = _mc[_db_name][os.getenv("MONGO_COLLECTION_NAME", "labels")]
            cursor = _col.find({"metadata.img_name": {"$regex": "post", "$options": "i"}})
            documents, metadatas, ids = [], [], []
            for doc in cursor:
                img_name = doc["metadata"].get("img_name")
                if not img_name:
                    continue
                disaster_type = doc["metadata"].get("disaster_type", "unknown")
                damage_counts = {"destroyed": 0, "major-damage": 0, "minor-damage": 0, "no-damage": 0}
                for b in doc.get("features", {}).get("xy", []):
                    subtype = b["properties"].get("subtype", "no-damage")
                    if subtype in damage_counts:
                        damage_counts[subtype] += 1
                center_lat, center_lon = 0.0, 0.0
                try:
                    wkt = doc["features"]["lng_lat"][0]["wkt"]
                    coords = _re.findall(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", wkt)
                    if coords:
                        center_lon = sum(float(c[0]) for c in coords) / len(coords)
                        center_lat = sum(float(c[1]) for c in coords) / len(coords)
                except (KeyError, IndexError):
                    pass
                total = sum(damage_counts.values())
                text = (
                    f"Image Scene ID: {img_name}\n"
                    f"Disaster Event: {disaster_type}\n"
                    f"Location: Lat {center_lat:.4f}, Lon {center_lon:.4f}\n"
                    f"Summary Assessment: {damage_counts['destroyed']} destroyed, "
                    f"{damage_counts['major-damage']} major damage, "
                    f"{damage_counts['minor-damage']} minor damage, "
                    f"{damage_counts['no-damage']} undamaged. Total buildings: {total}."
                )
                documents.append(text)
                ids.append(img_name)
                metadatas.append({"disaster_type": disaster_type, "lat": center_lat, "lon": center_lon})
            if ids:
                disaster_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                logger.info("Synced %d scenes to disaster_assessments.", len(ids))
    except Exception as exc:
        logger.error("Auto-sync disaster_assessments failed: %s", exc)

    try:
        if articles_collection is not None and articles_collection.count() == 0:
            logger.info("disaster_articles is empty — ingesting PDFs from S3...")
            from dataparser import s3_client, bucket_name
            from pdf_ingester import chunk_text as _chunk_text
            from pypdf import PdfReader as _PdfReader
            import io as _io
            prefix = "xview2-test-data/documents/"
            if s3_client and bucket_name:
                resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                for obj in resp.get("Contents", []):
                    key = obj["Key"]
                    if not key.lower().endswith(".pdf"):
                        continue
                    try:
                        body = s3_client.get_object(Bucket=bucket_name, Key=key)["Body"].read()
                        reader = _PdfReader(_io.BytesIO(body))
                        text = "\n".join(p.extract_text() or "" for p in reader.pages)
                        chunks = _chunk_text(text)
                        title = key.split("/")[-1].replace(".pdf", "").replace("_", " ")
                        doc_id = key.replace("/", "_").replace(".pdf", "").replace(" ", "_")
                        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                        metas = [{"source": key, "document_title": title, "disaster_type": "fire", "chunk_index": i, "total_chunks": len(chunks)} for i in range(len(chunks))]
                        articles_collection.upsert(ids=chunk_ids, documents=chunks, metadatas=metas)
                        logger.info("Ingested PDF '%s': %d chunks", title, len(chunks))
                    except Exception as e:
                        logger.warning("Failed to ingest PDF %s: %s", key, e)
    except Exception as exc:
        logger.error("Auto-sync disaster_articles failed: %s", exc)


@asynccontextmanager
async def chatbot_lifespan(_: Any):
    _clear_chat_state_on_startup()
    _auto_sync_chroma()
    yield


app = APIRouter(tags=["chatbot"], lifespan=chatbot_lifespan)


# chat message model for session history
class ChatMessageIn(BaseModel):
    user: str = Field(min_length=1, description="User identifier")
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)


# session history response model
class SessionHistoryResponse(BaseModel):
    session_id: str
    history: dict[str, list[dict[str, str]]]


# chat turn model
class ChatTurn(BaseModel):
    role: str
    content: str


# chatbot request model; includes the conversation history as context
class ChatApiRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_history: list[ChatTurn] = Field(default_factory=list)
    # NEW: Accept an optional dictionary of metadata filters from the frontend
    filters: dict[str, Any] = Field(default_factory=dict)


def _require_chat_sessions_collection() -> Collection:
    if chat_sessions_collection is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured for chat sessions (set MONGO_URI / MONGO_DB_NAME).",
        )
    return chat_sessions_collection


def _merge_where(
    base: dict[str, Any] | None, additional: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not base:
        return additional
    if not additional:
        return base

    clauses: list[dict[str, Any]] = []
    if "$and" in base and isinstance(base["$and"], list):
        clauses.extend(base["$and"])
    else:
        clauses.append(base)
    if "$and" in additional and isinstance(additional["$and"], list):
        clauses.extend(additional["$and"])
    else:
        clauses.append(additional)
    return {"$and": clauses}


def _filename_candidates(filename: str) -> list[str]:
    candidates = [filename]
    if filename.lower().endswith(".json"):
        candidates.append(f"{filename[:-5]}.png")
    elif filename.lower().endswith(".png"):
        candidates.append(f"{filename[:-4]}.json")
    seen: set[str] = set()
    deduped: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _candidate_ids_from_where(where: dict[str, Any] | None) -> list[str]:
    if not where:
        return []
    if "$and" in where and isinstance(where["$and"], list):
        collected: list[str] = []
        for clause in where["$and"]:
            if isinstance(clause, dict):
                collected.extend(_candidate_ids_from_where(clause))
        seen: set[str] = set()
        deduped: list[str] = []
        for item in collected:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped
    filename_filter = where.get("filename")
    if isinstance(filename_filter, str):
        return _filename_candidates(filename_filter)
    if isinstance(filename_filter, dict):
        raw_values = filename_filter.get("$in")
        if isinstance(raw_values, list):
            ids: list[str] = []
            for value in raw_values:
                if isinstance(value, str):
                    ids.extend(_filename_candidates(value))
            seen: set[str] = set()
            deduped: list[str] = []
            for item in ids:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            return deduped
    return []


def _rows_from_query_results(
    results: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    ids_group = results.get("ids", [])
    docs_group = results.get("documents", [])
    metadata_group = results.get("metadatas", [])
    if not ids_group or not isinstance(ids_group, list) or not ids_group[0]:
        return []
    ids = ids_group[0]
    docs = docs_group[0] if docs_group and isinstance(docs_group[0], list) else []
    metadatas = (
        metadata_group[0]
        if metadata_group and isinstance(metadata_group[0], list)
        else []
    )

    rows: list[tuple[str, str, dict[str, Any]]] = []
    for index, item_id in enumerate(ids):
        if not isinstance(item_id, str):
            continue
        doc = docs[index] if index < len(docs) and isinstance(docs[index], str) else ""
        metadata = (
            metadatas[index]
            if index < len(metadatas) and isinstance(metadatas[index], dict)
            else {}
        )
        rows.append((item_id, doc, metadata))
    return rows


def _rows_from_get_results(
    results: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    if not isinstance(ids, list) or not ids:
        return []

    rows: list[tuple[str, str, dict[str, Any]]] = []
    for index, item_id in enumerate(ids):
        if not isinstance(item_id, str):
            continue
        doc = docs[index] if index < len(docs) and isinstance(docs[index], str) else ""
        metadata = (
            metadatas[index]
            if index < len(metadatas) and isinstance(metadatas[index], dict)
            else {}
        )
        rows.append((item_id, doc, metadata))
    return rows


def _extract_filter_value(filters: dict[str, Any], key: str) -> str | None:
    value = filters.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _split_rag_filters(
    filters: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    where_filters: dict[str, Any] = {}
    candidate_ids: list[str] = []

    raw_id = filters.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        candidate_ids.append(raw_id.strip())

    raw_ids = filters.get("ids")
    if isinstance(raw_ids, list):
        for value in raw_ids:
            if isinstance(value, str) and value.strip():
                candidate_ids.append(value.strip())

    for key, value in filters.items():
        if key in SUPPORTED_METADATA_FILTER_KEYS:
            where_filters[key] = value

    seen: set[str] = set()
    deduped_ids: list[str] = []
    for item in candidate_ids:
        if item not in seen:
            seen.add(item)
            deduped_ids.append(item)

    return where_filters, deduped_ids


def _session_memory(session_id: str) -> dict[str, Any]:
    existing = chat_query_memory.get(session_id)
    if existing is not None:
        return existing
    created = {
        "last_queried_filename": None,
        "last_found_files": [],
        "around_center": None,
    }
    chat_query_memory[session_id] = created
    return created


def _session_id_from_request(request: Request) -> str:
    raw = request.headers.get("X-Chat-Session-Id", "").strip()
    return raw or "default"


def _retrieve_rag_context(
    *,
    prompt: str,
    session_id: str,
    filters: dict[str, Any],
) -> tuple[str, str, int]:
    if disaster_collection is None:
        return "", "no_collection", 0

    # Strip frontend-injected [System Context: ...] block before routing so a map-selected feature
    # UID doesn't force uid_lookup on every follow-up message.
    clean_prompt = _SYSTEM_CONTEXT_PATTERN.sub("", prompt).strip()

    # UID direct lookup — bypass ChromaDB and query MongoDB (only when user explicitly typed a UID)
    uid_match = _UID_PATTERN.search(clean_prompt)
    if uid_match:
        uid = uid_match.group(0)
        uid_info = _lookup_uid(uid)
        if uid_info:
            return f"--- BUILDING RECORD (MongoDB lookup) ---\n{uid_info}", "uid_lookup", 1

    memory = _session_memory(session_id)
    safe_filters, id_filters = _split_rag_filters(filters)
    routed = route_prompt_to_query(
        clean_prompt,
        n_results=8,
        last_queried_filename=memory.get("last_queried_filename"),
        last_found_files=memory.get("last_found_files"),
        around_center=memory.get("around_center"),
        disaster_type=_extract_filter_value(safe_filters, "disaster_type"),
        source_type=_extract_filter_value(safe_filters, "source_type"),
    )
    route = routed["route"]
    query_payload = dict(routed["query_payload"])
    merged_where = _merge_where(
        query_payload.get("where"), safe_filters if safe_filters else None
    )
    if merged_where:
        query_payload["where"] = merged_where

    rows: list[tuple[str, str, dict[str, Any]]]
    candidate_ids = list(id_filters)
    candidate_ids.extend(
        _candidate_ids_from_where(
            query_payload.get("where")
            if isinstance(query_payload.get("where"), dict)
            else None
        )
    )

    deduped_candidate_ids: list[str] = []
    seen_ids: set[str] = set()
    for item in candidate_ids:
        if item not in seen_ids:
            seen_ids.add(item)
            deduped_candidate_ids.append(item)

    if deduped_candidate_ids:
        rows = _rows_from_get_results(disaster_collection.get(ids=deduped_candidate_ids))
        if not rows:
            fallback_payload = dict(query_payload)
            fallback_payload.pop("where", None)
            fallback_payload["query_texts"] = [prompt]
            fallback_payload["n_results"] = 15
            rows = _rows_from_query_results(disaster_collection.query(**fallback_payload))
    else:
        rows = _rows_from_query_results(disaster_collection.query(**query_payload))
        if not rows and query_payload.get("where"):
            fallback_payload = dict(query_payload)
            fallback_payload.pop("where", None)
            fallback_payload["query_texts"] = [prompt]
            fallback_payload["n_results"] = 15
            rows = _rows_from_query_results(disaster_collection.query(**fallback_payload))

    # If a specific scene is selected on the map, ensure its record is included at the top.
    # This lets the LLM answer scene-specific chip questions (e.g. "How many destroyed?")
    # even when the semantic search didn't rank that scene highly.
    context_scene_match = _SCENE_ID_FROM_CONTEXT.search(prompt)
    if context_scene_match and disaster_collection:
        scene_filename = f"{context_scene_match.group(1)}_post_disaster.png"
        existing_ids = {r[0] for r in rows}
        if scene_filename not in existing_ids:
            try:
                scene_result = disaster_collection.get(ids=[scene_filename])
                scene_rows = _rows_from_get_results(scene_result)
                rows = scene_rows + rows  # prepend so LLM sees it first
            except Exception as e:
                logger.debug("Scene supplement lookup failed: %s", e)

    matched_ids = [row[0] for row in rows]
    matched_docs = [row[1] for row in rows if row[1]]
    memory_updates = routed["memory_updates"]
    memory["last_queried_filename"] = memory_updates.get("last_queried_filename")
    if matched_ids:
        memory["last_found_files"] = matched_ids
    geo_section: str = ""
    if rows:
        first_metadata = rows[0][2]
        lat = first_metadata.get("lat")
        lon = first_metadata.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            memory["around_center"] = (float(lat), float(lon))
            if any(kw in prompt.lower() for kw in _NEARBY_KEYWORDS):
                addresses: list[str] = []
                seen: set[str] = set()
                for _, _, meta in rows[:5]:
                    r_lat = meta.get("lat")
                    r_lon = meta.get("lon")
                    if not isinstance(r_lat, (int, float)) or not isinstance(r_lon, (int, float)):
                        continue
                    address = _reverse_geocode(float(r_lat), float(r_lon))
                    if address and address not in seen:
                        seen.add(address)
                        addresses.append(address)
                if addresses:
                    geo_section = "--- NEARBY LOCATIONS (reverse geocoded) ---\n" + "\n".join(
                        f"- {a}" for a in addresses
                    )

    # Also query the articles collection for relevant background information
    article_docs: list[str] = []
    if articles_collection is not None:
        try:
            article_count = articles_collection.count()
            if article_count > 0:
                art_results = articles_collection.query(
                    query_texts=[prompt], n_results=min(3, article_count)
                )
                article_docs = [d for d in (art_results.get("documents") or [[]])[0] if d]
        except Exception as e:
            logger.warning(f"Articles collection query failed: {e}")

    sections: list[str] = []
    if matched_docs:
        sections.append("--- DAMAGE ASSESSMENT RECORDS ---\n" + "\n".join(matched_docs))
    if article_docs:
        sections.append("--- REFERENCE ARTICLES (FEMA / Wildfire Guides) ---\n" + "\n".join(article_docs))
    if geo_section:
        sections.append(geo_section)
    return "\n\n".join(sections), route, len(matched_docs)


# Send chat request to OpenRouter and return the response
def openrouter_chat(messages: list[dict[str, Any]]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = os.getenv("OPENROUTER_CHAT_MODEL")
    if not model:
        raise RuntimeError("OPENROUTER_CHAT_MODEL is not set")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
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


@app.get("/")
def index() -> dict[str, str]:
    return {"message": "Chatbot API", "docs": "/docs"}


@app.post("/chat/sessions", status_code=201)
def create_session() -> dict[str, str]:
    session_id = str(uuid4())
    collection = _require_chat_sessions_collection()
    empty_history: dict[str, list[dict[str, str]]] = {}
    try:
        collection.insert_one({"_id": session_id, "history": empty_history})
    except PyMongoError as exc:
        logger.error("Failed to create chat session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=500, detail="Failed to create chat session"
        ) from exc
    _cache_set_json(_session_cache_key(session_id), empty_history)
    _invalidate_all_chat_history_cache()
    return {"session_id": session_id}


@app.delete("/chat/sessions")
def delete_all_sessions() -> dict[str, Any]:
    collection = _require_chat_sessions_collection()
    try:
        result = collection.delete_many({})
    except PyMongoError as exc:
        logger.error("Failed to delete all chat sessions: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to delete chat sessions"
        ) from exc
    chat_query_memory.clear()
    _invalidate_all_chat_history_cache()
    _invalidate_all_session_history_keys()
    return {
        "message": "All chat sessions deleted",
        "deleted_count": result.deleted_count,
    }


@app.delete("/chat/history/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    collection = _require_chat_sessions_collection()
    try:
        result = collection.delete_one({"_id": session_id})
    except PyMongoError as exc:
        logger.error("Failed to delete chat session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=500, detail="Failed to delete chat session"
        ) from exc
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    chat_query_memory.pop(session_id, None)
    _invalidate_session_history_cache(session_id)
    _invalidate_all_chat_history_cache()
    return {"message": f"Session {session_id} deleted"}


@app.get("/chat/history")
def get_all_history() -> dict[str, dict[str, list[dict[str, str]]]]:
    cached = _cache_get_json(CHAT_HISTORY_ALL_CACHE_KEY)
    if isinstance(cached, dict):
        return cached

    collection = _require_chat_sessions_collection()
    try:
        cursor = collection.find({}, {"history": 1})
        payload = {
            str(doc["_id"]): (
                doc["history"] if isinstance(doc.get("history"), dict) else {}
            )
            for doc in cursor
        }
        _cache_set_json(CHAT_HISTORY_ALL_CACHE_KEY, payload)
        return payload
    except PyMongoError as exc:
        logger.error("Failed to fetch all chat histories: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to fetch chat history"
        ) from exc


@app.get("/chat/history/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    cached = _cache_get_json(_session_cache_key(session_id))
    if isinstance(cached, dict):
        return SessionHistoryResponse(session_id=session_id, history=cached)

    collection = _require_chat_sessions_collection()
    try:
        doc = collection.find_one({"_id": session_id}, {"history": 1})
    except PyMongoError as exc:
        logger.error("Failed to fetch chat history for %s: %s", session_id, exc)
        raise HTTPException(
            status_code=500, detail="Failed to fetch chat history"
        ) from exc
    if doc is None:
        raise HTTPException(status_code=404, detail="Session not found")
    history = doc.get("history")
    payload = history if isinstance(history, dict) else {}
    _cache_set_json(_session_cache_key(session_id), payload)
    return SessionHistoryResponse(session_id=session_id, history=payload)


@app.post("/chat/history/{session_id}", response_model=SessionHistoryResponse)
def add_to_history(session_id: str, message: ChatMessageIn) -> SessionHistoryResponse:
    collection = _require_chat_sessions_collection()
    field = f"history.{message.user}"
    try:
        result = collection.update_one(
            {"_id": session_id},
            {
                "$push": {
                    field: {"prompt": message.prompt, "response": message.response}
                }
            },
        )
    except PyMongoError as exc:
        logger.error("Failed to append chat history for %s: %s", session_id, exc)
        raise HTTPException(
            status_code=500, detail="Failed to persist chat history"
        ) from exc
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    _invalidate_session_history_cache(session_id)
    response = get_session_history(session_id)
    _cache_set_json(_session_cache_key(session_id), response.history)
    _invalidate_all_chat_history_cache()
    return response


# Call this endpoint every time a new chat is sent
@app.post("/api/chat")
def api_chat(body: ChatApiRequest, request: Request) -> dict[str, Any]:
    rag_context = ""
    rag_route = "no_collection"
    rag_results = 0

    if disaster_collection:
        try:
            session_id = _session_id_from_request(request)
            rag_context, rag_route, rag_results = _retrieve_rag_context(
                prompt=body.message,
                session_id=session_id,
                filters=body.filters,
            )
            if rag_context:
                logger.info(
                    "RAG retrieval route=%s session=%s results=%d",
                    rag_route,
                    session_id,
                    rag_results,
                )
        except Exception as e:
            logger.error(f"RAG Retrieval Error: {e}")

    system = (
        "You are the 'Surge Disaster Assessment Assistant'. You help users analyze satellite imagery damage assessments and disaster-related documents. "
        "You may respond naturally to greetings. "
        "RULES:\n"
        "1. The 'RELEVANT DATABASE RECORDS' section below is your ONLY authoritative source. Answer from it directly.\n"
        "2. For street/location questions, list ONLY streets that appear in the 'NEARBY LOCATIONS' or 'LOCATION' fields of those records. Never infer or add streets from general knowledge or prior conversation turns.\n"
        "3. For building UID questions, use the 'BUILDING RECORD' section.\n"
        "4. Keep responses short and factual — 2 to 5 bullet points or sentences max.\n"
        "5. A '[System Context: ...]' tag in the user message shows the map feature currently selected. "
        "This tag does NOT restrict what records you may use — answer using ALL relevant records, not just the selected feature. "
        "If the user's question mentions a location or disaster that DIFFERS from the [System Context] tag, answer based on the RELEVANT DATABASE RECORDS, not the tag.\n"
        "6. If the user's question asks about counts or damage levels AND does NOT name a specific disaster, location, or place (e.g. 'how many structures were destroyed?' with no location given), do NOT guess. Ask them to clarify with these options: (a) a specific disaster — list the unique disaster_type values from the RELEVANT DATABASE RECORDS, or (b) total across all records. If they ask for totals, sum the counts from ALL records. If the question already names a location or disaster (e.g. 'in santa rosa', 'hurricane florence'), answer directly from the matching records without asking.\n"
        "7. If the records are empty or do not contain an answer, say so in one sentence.\n"
        "\n\nRELEVANT DATABASE RECORDS:\n"
        f"{rag_context if rag_context else 'No records found in database.'}"
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    # Only send the most recent 6 messages (3 turns) to prevent old UID/address context
    # from polluting answers to new general questions.
    recent_history = body.conversation_history[-6:]
    for turn in recent_history:
        messages.append({"role": turn.role.lower(), "content": turn.content})
    messages.append({"role": "user", "content": body.message})

    try:
        text = openrouter_chat(messages)
        return {
            "response": text,
            "message": text,
            "stats": {"ragRoute": rag_route, "ragResultCount": rag_results},
        }
    except Exception as exc:
        logger.error(f"Chat API Error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")
