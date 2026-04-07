"""FastAPI chatbot: session/history APIs and UI chat via OpenRouter (`/api/chat`)."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Chatbot API", version="1.0.0")

# In-memory session storage:
# { session_id: { user_id: [ {"prompt": ..., "response": ...}, ... ] } }
chat_sessions: dict[str, dict[str, list[dict[str, str]]]] = {}


class ChatMessageIn(BaseModel):
    user: str = Field(min_length=1, description="User identifier")
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: dict[str, list[dict[str, str]]]


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatApiRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_history: list[ChatTurn] = Field(default_factory=list)


def _openrouter_chat_completion(messages: list[dict[str, Any]]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = os.getenv("OPENROUTER_CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
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


@app.post("/api/chat")
def api_chat(body: ChatApiRequest) -> dict[str, Any]:
    """Endpoint used by the frontend chat page (`conversation_history` + `message`)."""
    system = (
        "You are an assistant for disaster damage assessment from satellite imagery. "
        "Answer clearly and concisely. If you cite numbers, note they are illustrative "
        "unless the user provided real data."
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in body.conversation_history:
        role = turn.role.lower().strip()
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": turn.content})
    messages.append({"role": "user", "content": body.message})

    try:
        text = _openrouter_chat_completion(messages)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"response": text, "message": text, "stats": None}
