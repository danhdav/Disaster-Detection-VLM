"""
Scan dataset root -> index.csv; read images and label JSON.
Layout-agnostic: finds *_post_disaster.json and pairs with pre, derives paths.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Benchmark scope: California wildfires (SoCal + Santa Rosa)
SCOPE_DISASTERS: set[str] = {"socal-fire", "santa-rosa-wildfire"}


def scan_dataset(
    dataset_root: str | Path,
    filter_disasters: set[str] | None = SCOPE_DISASTERS,
    split: str = "test",
) -> list[dict[str, str]]:
    """
    Scan dataset_root for label JSONs (e.g. *_post_disaster.json).
    filter_disasters: set of allowed disaster names read from JSON metadata.
        Defaults to SCOPE_DISASTERS (socal-fire only). Pass None to disable filter.
    split: value written to the 'official_split' column for all rows from this root.
    Return list of rows: tile_id, official_split, pre_path, post_path, label_json_path, target_path, disaster.
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        return []

    rows: list[dict[str, str]] = []
    skipped: dict[str, int] = {}

    for label_path in root.rglob("*_post_disaster.json"):
        stem = label_path.stem  # e.g. socal-fire_00001400_post_disaster
        if not stem.endswith("_post_disaster"):
            continue

        # Read metadata to check disaster name before committing
        try:
            with open(label_path, encoding="utf-8") as f:
                meta = json.load(f).get("metadata", {})
            disaster = meta.get("disaster", "")
        except Exception:
            disaster = ""

        if filter_disasters is not None and disaster not in filter_disasters:
            skipped[disaster] = skipped.get(disaster, 0) + 1
            continue

        tile_id = stem.replace("_post_disaster", "")
        pre_stem = f"{tile_id}_pre_disaster"
        post_stem = f"{tile_id}_post_disaster"

        label_dir = label_path.parent
        candidates_pre = [
            label_dir / f"{pre_stem}.png",
            label_dir / f"{pre_stem}.jpg",
            label_dir.parent / "images" / f"{pre_stem}.png",
            label_dir.parent / "images" / f"{pre_stem}.jpg",
        ]
        candidates_post = [
            label_dir / f"{post_stem}.png",
            label_dir / f"{post_stem}.jpg",
            label_dir.parent / "images" / f"{post_stem}.png",
            label_dir.parent / "images" / f"{post_stem}.jpg",
        ]
        pre_path = _first_exists(candidates_pre)
        post_path = _first_exists(candidates_post)

        rows.append({
            "tile_id": tile_id,
            "disaster": disaster,
            "official_split": split,
            "pre_path": str(pre_path) if pre_path else "",
            "post_path": str(post_path) if post_path else "",
            "label_json_path": str(label_path.resolve()),
            "target_path": "",
        })

    if skipped:
        skipped_summary = ", ".join(f"{k}({v})" for k, v in sorted(skipped.items()))
        print(f"  Skipped (out of scope): {skipped_summary}")

    return sorted(rows, key=lambda r: r["tile_id"])


def _first_exists(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def write_index_csv(rows: list[dict[str, str]], out_csv: str | Path) -> None:
    """Write index rows to CSV."""
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_index_csv(index_csv: str | Path) -> list[dict[str, str]]:
    """Read index CSV into list of row dicts."""
    with open(index_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def load_label_json(path: str | Path) -> dict[str, Any]:
    """Load a single label JSON (xView2-style with features.xy, features.lng_lat, metadata)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_buildings_from_label(label_data: dict[str, Any], use_xy: bool = True) -> list[dict[str, Any]]:
    """
    Extract building list from label JSON.
    Each building: { "uid", "subtype" (if present), "wkt" } in pixel (xy) or lng_lat coords.
    """
    key = "xy" if use_xy else "lng_lat"
    features = label_data.get("features", {}).get(key, [])
    out = []
    for feat in features:
        props = feat.get("properties", {})
        uid = props.get("uid", "")
        subtype = props["subtype"]
        wkt = feat.get("wkt", "")
        if uid and wkt:
            out.append({"uid": uid, "subtype": subtype, "wkt": wkt})
    return out


def get_label_canvas_size(label_data: dict[str, Any]) -> tuple[int, int]:
    """Return (width, height) of the label canvas from metadata."""
    meta = label_data.get("metadata", {})
    w = int(meta.get("width", 0))
    h = int(meta.get("height", 0))
    return w, h


def load_image(path: str | Path) -> np.ndarray:
    """Load image as RGB numpy array (H, W, 3) uint8."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    return np.array(im)


def load_image_grayscale(path: str | Path) -> np.ndarray:
    """Load image as grayscale (H, W) uint8."""
    with Image.open(path) as im:
        im = im.convert("L")
    return np.array(im)
