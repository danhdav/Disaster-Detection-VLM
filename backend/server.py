# This file will start the flask server

from pathlib import Path
from collections.abc import Callable

from dotenv import load_dotenv
from a2wsgi import ASGIMiddleware
from flask import Flask
from flask_cors import CORS

# Load local environment variables before importing API modules.
load_dotenv(Path(__file__).with_name(".env"))

# import FastAPI apps
from chatbot import app as chatbot_api
from db import app as db_api
from vlm import app as vlm_api

app = Flask(__name__)
CORS(app)

# Wrap APIs in middleware
chatbot_wsgi = ASGIMiddleware(chatbot_api)
db_wsgi = ASGIMiddleware(db_api)
vlm_wsgi = ASGIMiddleware(vlm_api)


def _is_chatbot_path(path: str) -> bool:
    return path.startswith("/chat") or path == "/api/chat"


def _is_db_path(path: str) -> bool:
    return (
        path == "/fire"
        or path.startswith("/fire/")
        or path.startswith("/image/")
        or path == "/debug/health"
    )


def _is_vlm_path(path: str) -> bool:
    return path == "/analyze"


def _dispatch_apis(default_wsgi: Callable):
    def _wsgi(environ: dict, start_response: Callable):
        path = environ.get("PATH_INFO", "")
        if _is_chatbot_path(path):
            return chatbot_wsgi(environ, start_response)
        if _is_db_path(path):
            return db_wsgi(environ, start_response)
        if _is_vlm_path(path):
            return vlm_wsgi(environ, start_response)
        return default_wsgi(environ, start_response)

    return _wsgi


app.wsgi_app = _dispatch_apis(app.wsgi_app)  # type: ignore[assignment]

@app.route("/")
def index():
    return "Running Flask server"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)