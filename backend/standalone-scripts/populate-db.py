from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

# Run with `uv run python populate-db.py`


def upload_json_files(
    folder: Path,
    collection: Collection[Any],
    limit: int | None = None,
) -> tuple[int, int]:
    files_uploaded = 0
    documents_inserted = 0

    for file_path in sorted(folder.rglob("*.json")):
        # Stop once limit is reached
        if limit is not None and files_uploaded >= limit:
            break

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            if data:
                collection.insert_many(data)
                documents_inserted += len(data)
        else:
            collection.insert_one(data)
            documents_inserted += 1

        print(f"Uploaded: {file_path.name}")
        files_uploaded += 1

    return files_uploaded, documents_inserted


def main() -> None:
    load_dotenv()

    # Read JSON files directly from the user's Downloads folder
    folder = Path(r"C:\Users\Proshun Saha\Downloads\test images tar")

    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    collection_name = os.getenv("MONGO_COLLECTION_NAME")

    if not mongo_uri:
        raise RuntimeError("Missing MONGO_URI environment variable.")

    if not db_name:
        raise RuntimeError("Missing MONGO_DB_NAME environment variable.")

    if not collection_name:
        raise RuntimeError("Missing MONGO_COLLECTION_NAME environment variable.")

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    client = MongoClient(mongo_uri)
    collection: Collection[Any] = client[db_name][collection_name]

    files_uploaded, documents_inserted = upload_json_files(
        folder, collection, limit=None
    )

    print(
        f"Done! Uploaded {files_uploaded} file(s) "
        f"and inserted {documents_inserted} document(s) "
        f"into {db_name}.{collection_name}."
    )


if __name__ == "__main__":
    main()
