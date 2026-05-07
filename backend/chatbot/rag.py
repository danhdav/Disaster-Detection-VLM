"""
RAG (Retrieval-Augmented Generation) module for the chatbot.
Manages document storage and retrieval using ChromaDB with persistent storage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import chromadb
from .search import route_prompt_to_query

# Persistent ChromaDB client
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = None


@dataclass
class ShortTermMemory:
    last_queried_filename: str | None = None
    last_found_files: list[str] = field(default_factory=list)


_memory_store: dict[str, ShortTermMemory] = {}


def _memory_for(session_id: str) -> ShortTermMemory:
    if session_id not in _memory_store:
        _memory_store[session_id] = ShortTermMemory()
    return _memory_store[session_id]


def init_rag_collection(collection_name: str = "disaster_docs"):
    """Initialize or get the RAG collection"""
    global collection
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_documents(
    documents: list[str], ids: list[str], metadata: list[dict] | None = None
):
    """Add documents to the RAG collection"""
    if collection is None:
        init_rag_collection()
    collection.add(documents=documents, ids=ids, metadatas=metadata or [])


def retrieve_context(query: str, n_results: int = 3) -> list[str]:
    """Retrieve relevant documents for a query"""
    if collection is None:
        init_rag_collection()

    # Return empty list if no documents
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


def reset_short_term_memory(session_id: str = "default") -> None:
    """Clear short-term memory for a session."""
    _memory_store[session_id] = ShortTermMemory()


def get_short_term_memory(session_id: str = "default") -> dict:
    """Expose short-term memory for inspection/debugging."""
    memory = _memory_for(session_id)
    return {
        "last_queried_filename": memory.last_queried_filename,
        "last_found_files": list(memory.last_found_files),
    }


def _set_last_found_files(files: list[str], session_id: str = "default") -> None:
    """Store most recent retrieved file group for plural references."""
    memory = _memory_for(session_id)
    memory.last_found_files = list(files)


def retrieve_context_with_memory(
    query: str, session_id: str = "default", n_results: int = 3
) -> list[str]:
    """Retrieve context with prompt routing and short-term memory."""
    if collection is None:
        init_rag_collection()

    if collection.count() == 0:
        return []

    memory = _memory_for(session_id)
    routing = route_prompt_to_query(
        query,
        n_results=n_results,
        last_queried_filename=memory.last_queried_filename,
        last_found_files=memory.last_found_files,
    )
    query_params = routing["query_payload"]
    results = collection.query(**query_params)

    memory.last_queried_filename = routing["memory_updates"]["last_queried_filename"]

    metadatas = results.get("metadatas", [])
    if metadatas and metadatas[0]:
        files = [meta["filename"] for meta in metadatas[0] if "filename" in meta]
        if files:
            _set_last_found_files(files, session_id=session_id)

    return results["documents"][0] if results["documents"] else []


def retrieve_context_routed(
    query: str, session_id: str = "default", n_results: int = 3
) -> dict[str, Any]:
    """Return routed retrieval details and documents for chat orchestration."""
    if collection is None:
        init_rag_collection()
    if collection.count() == 0:
        return {"route": "empty", "query_payload": {}, "documents": [], "metadatas": []}

    memory = _memory_for(session_id)
    routing = route_prompt_to_query(
        query,
        n_results=n_results,
        last_queried_filename=memory.last_queried_filename,
        last_found_files=memory.last_found_files,
    )
    results = collection.query(**routing["query_payload"])
    memory.last_queried_filename = routing["memory_updates"]["last_queried_filename"]

    metadatas = results.get("metadatas", [])
    if metadatas and metadatas[0]:
        files = [meta["filename"] for meta in metadatas[0] if "filename" in meta]
        if files:
            _set_last_found_files(files, session_id=session_id)

    return {
        "route": routing["route"],
        "query_payload": routing["query_payload"],
        "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
        "metadatas": metadatas[0] if metadatas else [],
    }


def get_collection_stats() -> dict:
    """Get stats about the current collection"""
    if collection is None:
        init_rag_collection()
    return {"document_count": collection.count()}
