"""
RAG (Retrieval-Augmented Generation) module for the chatbot.
Manages document storage and retrieval using ChromaDB with persistent storage.
"""

import os
import chromadb

# Persistent ChromaDB client
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = None


def init_rag_collection(collection_name: str = "disaster_docs"):
    """Initialize or get the RAG collection"""
    global collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_documents(documents: list[str], ids: list[str], metadata: list[dict] | None = None):
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


def get_collection_stats() -> dict:
    """Get stats about the current collection"""
    if collection is None:
        init_rag_collection()
    return {"document_count": collection.count()}
