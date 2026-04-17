"""Main backend entrypoint with one FastAPI app and module sub-routers."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load local environment variables before importing API modules.
load_dotenv(Path(__file__).with_name(".env"))

from chatbot import router as chatbot_router
from db import router as db_router
from vlm import router as vlm_router


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


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
    app.include_router(db_router)
    app.include_router(vlm_router)

    @app.get("/")
    def index() -> dict[str, str]:
        return {"message": "Disaster Detection API", "docs": "/docs"}

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = _env_int("PORT", 8000)
    workers = _env_int("UVICORN_WORKERS", 2)
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")
    access_log = _env_bool("UVICORN_ACCESS_LOG", True)
    timeout_keep_alive = _env_int("UVICORN_TIMEOUT_KEEP_ALIVE", 5)
    timeout_graceful_shutdown = _env_int("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", 30)
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level=log_level,
        access_log=access_log,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
    )