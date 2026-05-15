"""
Regenerate vlm_eval_sample.csv with full-resolution masked crop paths and WKT.

Adds columns:
  pre_masked_path  — pre_masked.png (building only, background black)
  post_masked_path — post_masked.png (building only, background black)
  wkt              — pixel-space polygon WKT from post_disaster label JSON

Usage:
    uv run python build_eval_csv.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

BENCH      = Path("D:/Aaron/UTD/Spring 26/Capstone Project/Benchmark-Model-xView2")
CROPS      = BENCH / "data/processed/crops_oracle"
LABELS     = BENCH / "data/test/labels"
SRC_CSV    = Path("data/vlm_eval_sample.csv")
DST_CSV    = Path("data/vlm_eval_full.csv")


def load_wkt_map(tile_id: str) -> dict[str, str]:
    """Return uid → pixel-space WKT from the post-disaster label JSON."""
    path = LABELS / f"{tile_id}_post_disaster.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for feat in data.get("features", {}).get("xy", []):
        uid = feat.get("properties", {}).get("uid", "")
        wkt = feat.get("wkt", "")
        if uid and wkt:
            result[uid] = wkt
    return result


def main() -> None:
    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))

    # Cache WKT maps per tile (many buildings share a tile)
    wkt_cache: dict[str, dict[str, str]] = {}

    updated: list[dict] = []
    missing_masked = missing_wkt = 0

    for row in rows:
        tile = row["tile_id"]
        uid  = row["uid"]

        pre_masked  = CROPS / tile / uid / "pre_masked.png"
        post_masked = CROPS / tile / uid / "post_masked.png"

        if tile not in wkt_cache:
            wkt_cache[tile] = load_wkt_map(tile)
        wkt = wkt_cache[tile].get(uid, "")

        if not pre_masked.exists() or not post_masked.exists():
            missing_masked += 1
        if not wkt:
            missing_wkt += 1

        updated.append({
            **row,
            "pre_masked_path":  str(pre_masked),
            "post_masked_path": str(post_masked),
            "wkt":              wkt,
        })

    fieldnames = list(updated[0].keys())
    with open(DST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(updated)

    print(f"Written : {DST_CSV}  ({len(updated)} buildings)")
    print(f"Missing masked crops : {missing_masked}")
    print(f"Missing WKT          : {missing_wkt}")


if __name__ == "__main__":
    main()
