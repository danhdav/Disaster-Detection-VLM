"""
Reusable ChromaDB query patterns for disaster-search use cases.
"""

from __future__ import annotations

import re
from typing import Any

FILENAME_PATTERN = re.compile(r"([a-zA-Z0-9_-]+\.(?:json|png))", re.IGNORECASE)
COORD_PATTERN = re.compile(
    r"lat(?:itude)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)\D+lon(?:gitude)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PLURAL_TERMS = ("these", "those", "them", "all", "they")
SINGULAR_TERMS = ("this", "it", "that", "the file", "the image")


def semantic_query(query_text: str, n_results: int = 5) -> dict[str, Any]:
    """Basic semantic similarity query."""
    return {"query_texts": [query_text], "n_results": n_results}


def disaster_type_query(
    query_text: str, disaster_type: str, n_results: int = 5
) -> dict[str, Any]:
    """Semantic query constrained to one disaster type."""
    return {
        "query_texts": [query_text],
        "n_results": n_results,
        "where": {"disaster_type": disaster_type},
    }


def radius_query(
    query_text: str,
    lat: float,
    lon: float,
    radius: float = 0.05,
    n_results: int = 5,
) -> dict[str, Any]:
    """Semantic query constrained to a lat/lon bounding box."""
    return {
        "query_texts": [query_text],
        "n_results": n_results,
        "where": {
            "$and": [
                {"lat": {"$gte": lat - radius}},
                {"lat": {"$lte": lat + radius}},
                {"lon": {"$gte": lon - radius}},
                {"lon": {"$lte": lon + radius}},
            ]
        },
    }


def group_files_query(
    query_text: str, filenames: list[str], n_results: int | None = None
) -> dict[str, Any]:
    """Semantic query restricted to a known set of filenames."""
    if not filenames:
        raise ValueError("filenames must contain at least one value")
    return {
        "query_texts": [query_text],
        "n_results": n_results or len(filenames),
        "where": {"filename": {"$in": filenames}},
    }


def single_file_query(query_text: str, filename: str) -> dict[str, Any]:
    """Semantic query restricted to one filename."""
    return {
        "query_texts": [query_text],
        "n_results": 1,
        "where": {"filename": filename},
    }


def keyword_document_query(
    query_text: str, keyword: str, n_results: int = 5
) -> dict[str, Any]:
    """Semantic query with document-content keyword filtering."""
    return {
        "query_texts": [query_text],
        "n_results": n_results,
        "where_document": {"$contains": keyword},
    }


def ids_get_pattern(ids: list[str]) -> dict[str, Any]:
    """Direct lookup payload for collection.get(ids=...)."""
    if not ids:
        raise ValueError("ids must contain at least one value")
    return {"ids": ids}


def _to_json_filename(filename: str) -> str:
    if filename.lower().endswith(".png"):
        return f"{filename[:-4]}.json"
    return filename


def _merge_where(
    where: dict[str, Any] | None, additional_filters: list[dict[str, Any]]
) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = []

    if where:
        if "$and" in where and isinstance(where["$and"], list):
            clauses.extend(where["$and"])
        else:
            clauses.append(where)

    clauses.extend(additional_filters)
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _apply_metadata_filters(
    payload: dict[str, Any],
    *,
    disaster_type: str | None,
    source_type: str | None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if disaster_type:
        filters.append({"disaster_type": disaster_type})
    if source_type:
        filters.append({"source_type": source_type})

    if not filters:
        return payload

    updated = dict(payload)
    updated["where"] = _merge_where(payload.get("where"), filters)
    return updated


def route_prompt_to_query(
    prompt: str,
    *,
    n_results: int = 5,
    last_queried_filename: str | None = None,
    last_found_files: list[str] | None = None,
    around_center: tuple[float, float] | None = None,
    around_radius: float = 0.02,
    coord_radius: float = 0.05,
    disaster_type: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    """
    Route user prompt to the most efficient Chroma query pattern.

    Returns:
      {
        "route": "...",
        "query_payload": {...},
        "memory_updates": {
            "last_queried_filename": str | None,
            "last_found_files": list[str]
        }
      }
    """
    user_lower = prompt.lower()
    memory_group = list(last_found_files or [])
    memory_updates = {
        "last_queried_filename": last_queried_filename,
        "last_found_files": memory_group,
    }

    exact_match = FILENAME_PATTERN.search(prompt)
    coord_match = COORD_PATTERN.search(prompt)
    is_around_query = "around" in user_lower
    is_plural_ref = any(word in user_lower for word in PLURAL_TERMS)
    is_singular_ref = any(word in user_lower for word in SINGULAR_TERMS)

    if exact_match:
        filename = _to_json_filename(exact_match.group(1))
        payload = single_file_query(prompt, filename=filename)
        memory_updates["last_queried_filename"] = filename
        route = "single_file"
    elif is_around_query and around_center is not None:
        lat, lon = around_center
        payload = radius_query(
            prompt,
            lat=lat,
            lon=lon,
            radius=around_radius,
            n_results=max(n_results, 5),
        )
        route = "around_radius"
    elif coord_match:
        lat = float(coord_match.group(1))
        lon = float(coord_match.group(2))
        payload = radius_query(
            prompt, lat=lat, lon=lon, radius=coord_radius, n_results=n_results
        )
        route = "coordinate_radius"
    elif is_plural_ref and memory_group:
        payload = group_files_query(prompt, filenames=memory_group, n_results=n_results)
        route = "memory_group"
    elif is_singular_ref and last_queried_filename:
        payload = single_file_query(prompt, filename=last_queried_filename)
        route = "memory_single"
    else:
        payload = semantic_query(prompt, n_results=n_results)
        route = "semantic"

    payload = _apply_metadata_filters(
        payload, disaster_type=disaster_type, source_type=source_type
    )
    return {
        "route": route,
        "query_payload": payload,
        "memory_updates": memory_updates,
    }
