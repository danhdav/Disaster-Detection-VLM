'''
This file handles all chatbot-related API endpoints (i.e messages and session handling).
'''

from __future__ import annotations

import os
import logging
from typing import Any
from uuid import uuid4 # for generating unique session IDs

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field # use for data validation error checking
import chromadb # <-- NEW: Imported ChromaDB

# Setup basic logging to see errors in your terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = APIRouter(tags=["chatbot"])

# ==========================================
# RAG: CHROMADB INITIALIZATION
# ==========================================
CHROMA_PATH = "./xview2_vector_db"
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    disaster_collection = chroma_client.get_or_create_collection(name="disaster_assessments")
    logger.info("ChromaDB initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    disaster_collection = None

# In-memory session storage:
# { session_id: { user_id: [ {"prompt": ..., "response": ...}, ... ] } }
chat_sessions: dict[str, dict[str, list[dict[str, str]]]] = {}

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
    chat_sessions[session_id] = {}
    return {"session_id": session_id}

@app.delete("/chat/sessions")
def delete_all_sessions() -> dict[str, Any]:
    deleted_count = len(chat_sessions)
    chat_sessions.clear()
    return {"message": "All chat sessions deleted", "deleted_count": deleted_count}

@app.delete("/chat/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del chat_sessions[session_id]
    return {"message": f"Session {session_id} deleted"}

@app.get("/chat/history")
def get_all_history() -> dict[str, dict[str, list[dict[str, str]]]]:
    return chat_sessions

@app.get("/chat/history/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    history = chat_sessions.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionHistoryResponse(session_id=session_id, history=history)

@app.post("/chat/history/{session_id}", response_model=SessionHistoryResponse)
def add_to_history(session_id: str, message: ChatMessageIn) -> SessionHistoryResponse:
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session_history = chat_sessions[session_id]
    session_history.setdefault(message.user, []).append(
        {"prompt": message.prompt, "response": message.response}
    )
    return SessionHistoryResponse(session_id=session_id, history=session_history)

# Call this endpoint every time a new chat is sent
# Call this endpoint every time a new chat is sent
@app.post("/api/chat")
def api_chat(body: ChatApiRequest) -> dict[str, Any]:
    rag_context = ""
    
    if disaster_collection:
        try:
            # We search specifically for the filename or keywords in the message
            query_kwargs = {
                "query_texts": [body.message],
                "n_results": 5 # Increased to 5 for better coverage
            }

            if body.filters and len(body.filters) > 0:
                query_kwargs["where"] = body.filters

            results = disaster_collection.query(**query_kwargs)

            if results and results.get('documents') and results['documents'][0]:
                rag_context = "\n".join(results['documents'][0])
                # DEBUG: Print this to your terminal to see what the LLM sees
                print(f"--- RAG CONTEXT FOUND ---\n{rag_context}\n-------------------------")
        except Exception as e:
            logger.error(f"RAG Retrieval Error: {e}")

    # STRENGTHENED SYSTEM PROMPT
    system = (
        "You are the 'Disaster Assessment Assistant'. You have access to a database of satellite imagery labels. "
        "Strictly use the 'RELEVANT DATABASE RECORDS' provided below to answer. "
        "If the records contain information about a filename the user mentioned, summarize the building damage counts and location. "
        "If no records are provided below, or they don't match the user's request, say: 'I couldn't find that specific record in my database.' "
        "\n\nRELEVANT DATABASE RECORDS:\n"
        f"{rag_context if rag_context else 'No records found for this query.'}"
    )
    
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    
    # Add history and user message
    for turn in body.conversation_history:
        messages.append({"role": turn.role.lower(), "content": turn.content})
    messages.append({"role": "user", "content": body.message})

    try:
        text = openrouter_chat(messages)
        return {"response": text, "message": text, "stats": None}
    except Exception as exc:
        logger.error(f"Chat API Error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")