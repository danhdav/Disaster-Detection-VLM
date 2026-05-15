"""
PDF ingestion and chunking for disaster article RAG.
Handles extraction, chunking, and embedding storage in ChromaDB.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Any

import chromadb
from pypdf import PdfReader

from dataparser import s3_client

logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # words of overlap between chunks
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_PATH = os.getenv(
    "CHROMA_DB_PATH",
    os.path.join(BASE_DIR, "xview2_vector_db"),
)

# PDF ingestion is disabled by deployment configuration.
articles_collection = None
logger.info("PDF ingestion is disabled. document ingest/list/delete endpoints are inactive.")


def _word_count(text: str) -> int:
    """Count words in text by splitting on whitespace."""
    return len(text.split())


def _get_words(text: str) -> list[str]:
    """Split text into word list."""
    return text.split()


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Chunk text into overlapping segments.

    Parameters:
      text: Full text to chunk
      chunk_size: Target words per chunk
      overlap: Words of overlap between consecutive chunks

    Returns:
      List of text chunks with overlap preserved

    Example:
      text = "Word1 Word2 Word3 Word4 Word5"
      chunks = chunk_text(text, chunk_size=2, overlap=1)
      # chunks = ["Word1 Word2", "Word2 Word3 Word4", "Word4 Word5"]
      #           (2 words)    (3 words: 2 new + 1 overlap)
    """
    words = _get_words(text)
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move start forward by (chunk_size - overlap) for next iteration
        start += chunk_size - overlap
        if start >= len(words):
            break

    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file (local or S3).

    Parameters:
      pdf_path: Path to PDF (local file path or S3 key for remote fetch)

    Returns:
      Concatenated text from all PDF pages
    """
    try:
        if pdf_path.startswith("s3://"):
            # Remote S3 path
            bucket, key = pdf_path.replace("s3://", "").split("/", 1)
            if s3_client is None:
                raise RuntimeError("S3 client not configured")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            pdf_bytes = response["Body"].read()
            pdf_file = io.BytesIO(pdf_bytes)
        else:
            # Local file path
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pdf_file = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_file)
        text_parts: list[str] = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            except Exception as e:
                logger.warning(
                    f"Failed to extract text from page {page_num}: {e}"
                )

        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        raise


def ingest_pdf(
    pdf_path: str,
    document_title: str | None = None,
    disaster_type: str | None = None,
    source_url: str | None = None,
    filename: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    source_type: str | None = "document",
) -> dict[str, Any]:
    """
    Download or open a PDF, chunk it, embed, and store in ChromaDB.

    Parameters:
        pdf_path: Local path or S3 URI (e.g., "s3://bucket/key.pdf" or "/tmp/article.pdf")
        document_title: Human-readable title for the document
        disaster_type: Optional disaster type for filtering (e.g., "hurricane")
        source_url: Optional source URL for attribution
        filename: Optional filename to store in metadata (e.g. 'doc123.json').
            If omitted a filename will be derived from the source path.
        lat, lon: Optional spatial coordinates to store on the document metadata
        source_type: Optional source type (defaults to "document") used by search filters

    Returns:
      {
        "success": bool,
        "chunks_ingested": int,
        "document_id": str,
        "error": str | None
      }
    """
    if articles_collection is None:
        return {
            "success": False,
            "chunks_ingested": 0,
            "error": "ChromaDB articles collection not initialized",
        }

    source_key = pdf_path
    if pdf_path.startswith("s3://"):
        source_key = pdf_path

    try:
        # Extract PDF text from a local file or S3 URI.
        text = extract_text_from_pdf(source_key)

        if not text.strip():
            return {
                "success": False,
                "chunks_ingested": 0,
                "error": "PDF contains no extractable text",
            }

        # Chunk the text with overlap
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        if not chunks:
            return {
                "success": False,
                "chunks_ingested": 0,
                "error": "No chunks produced from text",
            }

        # Normalize a source key for PDF documents
        source_key = pdf_path
        if pdf_path.startswith("s3://"):
            source_key = pdf_path

        # Generate document ID from source path/key (remove extension, sanitize)
        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", source_key.replace(".pdf", ""))

        # Derive a metadata filename if not explicitly provided
        filename_value = filename or f"{doc_id}.json"

        # Prepare metadata for all chunks (include fields expected by search queries)
        metadata_base: dict[str, Any] = {
            "source": source_key,
            "document_title": document_title or source_key.split("/")[-1],
            "filename": filename_value,
            "source_type": source_type,
        }
        if lat is not None and lon is not None:
            metadata_base["lat"] = lat
            metadata_base["lon"] = lon
        if disaster_type:
            metadata_base["disaster_type"] = disaster_type
        if source_url:
            metadata_base["source_url"] = source_url

        # Add chunks to ChromaDB
        chunk_ids: list[str] = []
        chunk_texts: list[str] = []
        chunk_metadatas: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)

            metadata = dict(metadata_base)
            metadata["chunk_index"] = idx
            metadata["total_chunks"] = len(chunks)
            chunk_metadatas.append(metadata)

        # Bulk insert into ChromaDB (embeddings generated automatically)
        articles_collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas,
        )

        logger.info(
            f"Ingested PDF {source_key}: {len(chunks)} chunks added to ChromaDB"
        )
        return {
            "success": True,
            "chunks_ingested": len(chunks),
            "document_id": doc_id,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error ingesting PDF {source_key}: {e}")
        return {
            "success": False,
            "chunks_ingested": 0,
            "error": str(e),
        }


def list_ingested_articles() -> list[dict[str, Any]]:
    """
    List all unique articles currently in the articles collection.

    Returns:
      List of article metadata dicts (deduplicated by document source)
    """
    if articles_collection is None:
        return []

    try:
        # Get all items (without filtering)
        all_items = articles_collection.get()
        metadatas = all_items.get("metadatas", [])

        # Deduplicate by source
        seen_sources: set[str] = set()
        articles: list[dict[str, Any]] = []
        for meta in metadatas:
            source = meta.get("source")
            if source and source not in seen_sources:
                seen_sources.add(source)
                articles.append(
                    {
                        "source": source,
                        "title": meta.get("document_title", "Unknown"),
                        "disaster_type": meta.get("disaster_type"),
                        "total_chunks": meta.get("total_chunks"),
                    }
                )

        return articles
    except Exception as e:
        logger.error(f"Error listing articles: {e}")
        return []


def delete_article(source: str) -> dict[str, Any]:
    """
    Delete all chunks of an article from ChromaDB.

    Parameters:
      source: Local path or S3 URI of the article to delete

    Returns:
      {"success": bool, "chunks_deleted": int, "error": str | None}
    """
    if articles_collection is None:
        return {
            "success": False,
            "chunks_deleted": 0,
            "error": "ChromaDB articles collection not initialized",
        }

    try:
        source_key = source
        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", source_key.replace(".pdf", ""))

        # Get all chunk IDs for this document
        all_items = articles_collection.get()
        ids_to_delete = [
            id_val
            for id_val, meta in zip(
                all_items.get("ids", []), all_items.get("metadatas", [])
            )
            if meta.get("source") == source_key
        ]

        if ids_to_delete:
            articles_collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for {source_key}")

        return {
            "success": True,
            "chunks_deleted": len(ids_to_delete),
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error deleting article {source_key}: {e}")
        return {
            "success": False,
            "chunks_deleted": 0,
            "error": str(e),
        }