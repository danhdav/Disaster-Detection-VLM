import os
import re

import chromadb
from dotenv import load_dotenv
from pymongo import MongoClient

# 1. SETUP CONNECTIONS
load_dotenv()
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = os.getenv("MONGO_DB_NAME")
col = os.getenv("MONGO_COLLECTION_NAME")
database = mongo_client[db]
labels_collection = database[col]

# PersistentClient needs to change for deployment
chroma_client = chromadb.PersistentClient(path="../xview2_vector_db")
collection = chroma_client.get_or_create_collection(name="disaster_assessments")


def sync_mongo_to_chroma():
    # Only grab post-disaster images for the assessment summary
    query = {"metadata.img_name": {"$regex": "post", "$options": "i"}}
    cursor = labels_collection.find(query)

    print("Reverting to Image-Level Sync (One record per image)...")

    documents = []
    metadatas = []
    ids = []

    for doc in cursor:
        img_name = doc["metadata"].get("img_name")
        disaster_type = doc["metadata"].get("disaster_type", "unknown")

        # 1. Extract Aggregate Damage Counts
        damage_counts = {
            "destroyed": 0,
            "major-damage": 0,
            "minor-damage": 0,
            "no-damage": 0,
        }
        buildings = doc.get("features", {}).get("xy", [])
        for b in buildings:
            subtype = b["properties"].get("subtype", "no-damage")
            if subtype in damage_counts:
                damage_counts[subtype] += 1

        # 2. Extract Center Coordinates from the first building
        center_lat, center_lon = 0.0, 0.0
        try:
            wkt_string = doc["features"]["lng_lat"][0]["wkt"]
            coords = re.findall(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", wkt_string)
            if coords:
                center_lon = sum(float(c[0]) for c in coords) / len(coords)
                center_lat = sum(float(c[1]) for c in coords) / len(coords)
        except (KeyError, IndexError):
            pass

        # 3. Format the Summary Text Chunk
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
            {"disaster_type": disaster_type, "lat": center_lat, "lon": center_lon}
        )

    # 4. Push to ChromaDB
    if ids:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Sync complete! {len(ids)} image summaries are now searchable.")
    else:
        print("No matching records found.")


if __name__ == "__main__":
    sync_mongo_to_chroma()
