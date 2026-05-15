from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

DEFAULT_PREFIX = "xview2-test-data/documents/"


def _normalize_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _collect_pdfs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".pdf" else []
    if not input_path.is_dir():
        return []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in input_path.glob(pattern) if path.is_file())


def _build_object_key(prefix: str, file_path: Path, root_dir: Path) -> str:
    try:
        relative = file_path.relative_to(root_dir)
        return f"{prefix}{relative.as_posix()}"
    except ValueError:
        return f"{prefix}{file_path.name}"


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload one PDF or a directory of PDFs to S3."
    )
    parser.add_argument("path", help="PDF file path or directory containing PDFs")
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET_NAME"),
        help="Target S3 bucket (defaults to S3_BUCKET_NAME from .env)",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 key prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when path is a directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    args = parser.parse_args()

    if not args.bucket:
        print("Error: missing bucket. Set S3_BUCKET_NAME or pass --bucket.")
        return 1

    input_path = Path(args.path).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: path does not exist: {input_path}")
        return 1

    files = _collect_pdfs(input_path, recursive=args.recursive)
    if not files:
        print("No PDF files found to upload.")
        return 1

    root_dir = input_path if input_path.is_dir() else input_path.parent
    prefix = _normalize_prefix(args.prefix)
    s3_client = boto3.client("s3")

    uploaded = 0
    for file_path in files:
        key = _build_object_key(prefix, file_path, root_dir)
        print(f"{file_path} -> s3://{args.bucket}/{key}")
        if args.dry_run:
            continue
        try:
            s3_client.upload_file(str(file_path), args.bucket, key)
            uploaded += 1
        except (BotoCoreError, ClientError) as exc:
            print(f"Upload failed for {file_path}: {exc}")

    if args.dry_run:
        print(f"Dry run complete. {len(files)} file(s) matched.")
        return 0

    print(f"Upload complete. {uploaded}/{len(files)} file(s) uploaded.")
    return 0 if uploaded == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
