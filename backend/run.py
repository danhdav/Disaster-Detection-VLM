from server import app
import os
import uvicorn


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


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = _env_int("PORT", 8000)
    workers = _env_int("UVICORN_WORKERS", 2)
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")
    access_log = _env_bool("UVICORN_ACCESS_LOG", True)
    timeout_keep_alive = _env_int("UVICORN_TIMEOUT_KEEP_ALIVE", 5)
    timeout_graceful_shutdown = _env_int("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", 30)
    uvicorn.run(
        app,
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
