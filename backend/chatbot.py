"""FastAPI chatbot session and history APIs.

Implements:
- Create a new chat session
- Delete a chat session
- Read chat history (all sessions or one session)
- Add prompt/response messages to a session history

Chat history shape:
{
	"<session_id>": {
		"<user>": [
			{"prompt": "...", "response": "..."}
		]
	}
}
"""

from __future__ import annotations

from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Chatbot API", version="1.0.0")

# In-memory session storage
chat_sessions: Dict[str, Dict[str, List[dict[str, str]]]] = {}

class ChatMessageIn(BaseModel):
	user: str = Field(min_length=1, description="User identifier")
	prompt: str = Field(min_length=1)
	response: str = Field(min_length=1)

class SessionHistoryResponse(BaseModel):
	session_id: str
	history: Dict[str, List[dict[str, str]]]

# Index
@app.get("/")
def index() -> dict[str, str]:
	# Assign a session id and initialize empty history if in new browser session
	session_id = str(uuid4())
	chat_sessions[session_id] = {}
	return {"message": "Welcome to the Chatbot API!", "session_id": session_id}

# Delete a session and its chat history
@app.delete("/chat/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
	"""Delete a session and all of its chat history."""
	if session_id not in chat_sessions:
		raise HTTPException(status_code=404, detail="Session not found")

	del chat_sessions[session_id]
	return {"message": f"Session {session_id} deleted"}

# Fetch chat history for all sessions
@app.get("/chat/history")
def get_all_history() -> dict[str, Dict[str, List[dict[str, str]]]]:
	"""Return history for all sessions."""
	return chat_sessions

# Fetch a session's chat history
@app.get("/chat/history/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
	"""Return history for one session."""
	history = chat_sessions.get(session_id)
	if history is None:
		raise HTTPException(status_code=404, detail="Session not found")

	return SessionHistoryResponse(session_id=session_id, history=history)

# Add to a session's chat history
@app.post("/chat/history/{session_id}", response_model=SessionHistoryResponse)
def add_to_history(session_id: str, message: ChatMessageIn) -> SessionHistoryResponse:
	"""Append one prompt/response exchange to a user's session history."""
	if session_id not in chat_sessions:
		raise HTTPException(status_code=404, detail="Session not found")

	session_history = chat_sessions[session_id]
	session_history.setdefault(message.user, []).append(
		{"prompt": message.prompt, "response": message.response}
	)

	return SessionHistoryResponse(session_id=session_id, history=session_history)