"""
Unified Sync Script for Disaster Assessments.
Handles syncing MongoDB image records and ingesting PDFs into a single ChromaDB collection.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Any

import chromadb
from dotenv import load_dotenv
from pymongo import MongoClient
from pypdf import PdfReader

# Attempt to import s3_client, fallback to None if not available
try:
    from dataparser import s3_client
except ImportError:
    s3_client = None

# ---------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB Setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = os.getenv("MONGO_DB_NAME")
col = os.getenv("MONGO_COLLECTION_NAME")
database = mongo_client[db]
labels_collection = database[col]

# FIX: Absolute Pathing
# Since script is now in 'backend/', BASE_DIR is 'backend/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "xview2_vector_db"))

logger.info(f"Connecting to ChromaDB at: {CHROMA_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="disaster_assessments")

# PDF Chunking Parameters
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # words of overlap between chunks


# ---------------------------------------------------------------------------
# 2. MONGODB SYNC FUNCTIONS
# ---------------------------------------------------------------------------
def sync_mongo_to_chroma():
    """Extracts post-disaster image damage assessments from Mongo and pushes to Chroma."""
    query = {"metadata.img_name": {"$regex": "post", "$options": "i"}}
    cursor = labels_collection.find(query)

    logger.info("Syncing MongoDB Image Records...")

    documents = []
    metadatas = []
    ids = []

    for doc in cursor:
        img_name = doc["metadata"].get("img_name")
        disaster_type = doc["metadata"].get("disaster_type", "unknown")

        damage_counts = {"destroyed": 0, "major-damage": 0, "minor-damage": 0, "no-damage": 0}
        buildings = doc.get("features", {}).get("xy", [])
        for b in buildings:
            subtype = b["properties"].get("subtype", "no-damage")
            if subtype in damage_counts:
                damage_counts[subtype] += 1

        center_lat, center_lon = 0.0, 0.0
        try:
            wkt_string = doc["features"]["lng_lat"][0]["wkt"]
            coords = re.findall(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", wkt_string)
            if coords:
                center_lon = sum(float(c[0]) for c in coords) / len(coords)
                center_lat = sum(float(c[1]) for c in coords) / len(coords)
        except (KeyError, IndexError):
            pass

        rag_text_chunk = (
            f"Image Scene ID: {img_name}\n"
            f"Disaster Event: {disaster_type}\n"
            f"Location: Lat {center_lat:.4f}, Lon {center_lon:.4f}\n"
            f"Summary Assessment: {damage_counts['destroyed']} destroyed, "
            f"{damage_counts['major-damage']} major, {damage_counts['minor-damage']} minor."
        )

        documents.append(rag_text_chunk)
        ids.append(img_name)
        metadatas.append(
            {"disaster_type": disaster_type, "lat": center_lat, "lon": center_lon, "source_type": "image_assessment"}
        )

    if ids:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"✅ Mongo Sync Complete! {len(ids)} images synced.")
    else:
        logger.info("No matching Mongo records found.")


# ---------------------------------------------------------------------------
# 3. PDF INGESTION FUNCTIONS
# ---------------------------------------------------------------------------
def _get_words(text: str) -> list[str]:
    return text.split()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = _get_words(text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
        if start >= len(words): break
    return chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        if pdf_path.startswith("s3://"):
            bucket, key = pdf_path.replace("s3://", "").split("/", 1)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            pdf_file = io.BytesIO(response["Body"].read())
        else:
            with open(pdf_path, "rb") as f:
                pdf_file = io.BytesIO(f.read())

        reader = PdfReader(pdf_file)
        text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        raise

def ingest_pdf(pdf_path: str, document_title: str | None = None, disaster_type: str | None = None):
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip(): return
        chunks = chunk_text(text)
        
        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(pdf_path).replace(".pdf", ""))
        
        chunk_ids, chunk_texts, chunk_metadatas = [], [], []
        for idx, chunk in enumerate(chunks):
            chunk_ids.append(f"{doc_id}_chunk_{idx}")
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                "source": os.path.basename(pdf_path),
                "document_title": document_title or os.path.basename(pdf_path),
                "source_type": "document",
                "disaster_type": disaster_type or "general",
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })

        collection.add(ids=chunk_ids, documents=chunk_texts, metadatas=chunk_metadatas)
        logger.info(f"✅ Ingested PDF: {pdf_path} ({len(chunks)} chunks)")
    except Exception as e:
        logger.error(f"Error ingesting PDF {pdf_path}: {e}")

def sync_all_pdfs_in_directory(directory_path: str):
    if not os.path.exists(directory_path):
        logger.warning(f"Directory {directory_path} not found.")
        return
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            ingest_pdf(os.path.join(directory_path, filename))

# ---------------------------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting combined synchronization process...")
    
    # 1. Sync MongoDB
    sync_mongo_to_chroma()
    
    # 2. Sync PDFs
    # Make sure this folder exists inside your backend folder!
    pdf_dir = os.path.join(BASE_DIR, "standalone-scripts/pdfs") 
    sync_all_pdfs_in_directory(pdf_dir)
    
    # 3. Final Verification Log
    all_data = collection.get(include=["metadatas"])
    unique_sources = set(m.get("source") for m in all_data["metadatas"] if m.get("source_type") == "document")
    
    logger.info("--- SYNC SUMMARY ---")
    logger.info(f"Total Chunks in DB: {collection.count()}")
    logger.info(f"Unique PDFs Found: {len(unique_sources)}")
    for src in unique_sources:
        logger.info(f" -> {src}")
    logger.info("--------------------")