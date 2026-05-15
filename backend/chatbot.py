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

from dataparser import mongo_client, mongo_db_name
from search import route_prompt_to_query

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
    logger.info("ChromaDB initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    disaster_collection = None

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


@asynccontextmanager
async def chatbot_lifespan(_: Any):
    _clear_chat_state_on_startup()
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

    memory = _session_memory(session_id)
    safe_filters, id_filters = _split_rag_filters(filters)
    routed = route_prompt_to_query(
        prompt,
        n_results=500,
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
            fallback_payload["n_results"] = 5
            rows = _rows_from_query_results(disaster_collection.query(**fallback_payload))
    else:
        rows = _rows_from_query_results(disaster_collection.query(**query_payload))
        if not rows and query_payload.get("where"):
            fallback_payload = dict(query_payload)
            fallback_payload.pop("where", None)
            fallback_payload["query_texts"] = [prompt]
            fallback_payload["n_results"] = 5
            rows = _rows_from_query_results(disaster_collection.query(**fallback_payload))

    matched_ids = [row[0] for row in rows]
    matched_docs = [row[1] for row in rows if row[1]]
    memory_updates = routed["memory_updates"]
    memory["last_queried_filename"] = memory_updates.get("last_queried_filename")
    if matched_ids:
        memory["last_found_files"] = matched_ids
    if rows:
        first_metadata = rows[0][2]
        lat = first_metadata.get("lat")
        lon = first_metadata.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            memory["around_center"] = (float(lat), float(lon))

    # Format context with source filenames from metadata
    formatted_chunks: list[str] = []
    for row in rows:
        chunk_text = row[1]
        metadata = row[2]
        if chunk_text:  # Only include non-empty chunks
            source_info = metadata.get("filename", "Unknown Source")
            document_title = metadata.get("document_title", "")
            if document_title:
                source_info = f"{source_info} ({document_title})"
            formatted_chunks.append(f"[Source: {source_info}]\n{chunk_text}")
            logger.debug(f"Formatted chunk from {source_info}: {len(chunk_text)} chars")

    formatted_context = "\n\n".join(formatted_chunks)
    logger.info(
        f"RAG context formatted: {len(formatted_chunks)} chunks, {len(formatted_context)} total chars"
    )
    return formatted_context, route, len(formatted_chunks)


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
            logger.info(
                f"RAG context retrieved: route={rag_route}, results={rag_results}, context_len={len(rag_context)}"
            )
            if rag_context and rag_context.strip():
                logger.info(
                    "RAG retrieval route=%s session=%s results=%d",
                    rag_route,
                    session_id,
                    rag_results,
                )
            else:
                logger.warning(
                    "RAG context is empty despite retrieval attempt: route=%s session=%s",
                    rag_route,
                    session_id,
                )
        except Exception as e:
            logger.error(f"RAG Retrieval Error: {e}", exc_info=True)

    # ROBUST SYSTEM PROMPT SUPPORTING MULTIPLE QUERY TYPES
    # FORCED-ANSWER SYSTEM PROMPT
    system = (
        "You are the 'Disaster Assessment Assistant'. You help users analyze satellite imagery damage labels and disaster-related documents. "
        "You may respond naturally and politely to general greetings. "
        
        "CRITICAL KNOWLEDGE HIERARCHY:\n"
        "1. PRIORITIZE DATA: When answering specific questions, you must first rely entirely on the 'RELEVANT DATABASE RECORDS' provided below. "
        "2. DOCUMENT USAGE: If the records contain text from PDFs, you MUST use that text to form your response. Do not refuse to answer if text is provided. "
        "3. FALLBACK TO GENERAL KNOWLEDGE: If the 'RELEVANT DATABASE RECORDS' do not contain specific information about a question, or if the records are empty, you should answer the question using your general internal knowledge (as a standard Large Language Model would). "
        
        "When using your general knowledge because the database is empty, politely mention that you are providing general information rather than specific record-based data. "

        "\n\nRELEVANT DATABASE RECORDS:\n"
        f"{rag_context if rag_context else 'No records found in database.'}"
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    # Add history and user message
    for turn in body.conversation_history:
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
