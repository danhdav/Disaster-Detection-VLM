"""Main backend entrypoint with one FastAPI app and module sub-routers."""

from __future__ import annotations

import os
from pathlib import Path


from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load local environment variables before importing API modules.
load_dotenv(Path(__file__).with_name(".env"))

from chatbot import app as chatbot_router  # noqa: E402
from cnn import app as cnn_router  # noqa: E402
from db import app as db_router  # noqa: E402
from vlm import app as vlm_router  # noqa: E402





def _cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if raw_origins == "*":
        return ["*"]
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def create_app() -> FastAPI:
    app = FastAPI(title="Disaster Detection API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chatbot_router)
    app.include_router(cnn_router)
    app.include_router(db_router)
    app.include_router(vlm_router)

    @app.get("/")
    def index() -> dict[str, str]:
        return {"message": "Disaster Detection API", "docs": "/docs"}

    return app


app = create_app()


