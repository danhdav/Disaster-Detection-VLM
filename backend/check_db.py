import chromadb
import os

# Use the EXACT same path your chatbot uses
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "xview2_vector_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("disaster_assessments")

print(f"Total items in DB: {collection.count()}")

# Peek at the metadata of the first 5 items
results = collection.get(limit=5, include=["metadatas", "documents"])
for i, meta in enumerate(results["metadatas"]):
    source_type = meta.get("source_type", "unknown")
    source_file = meta.get("source", "N/A")
    print(f"Item {i}: Type={source_type}, Source={source_file}")