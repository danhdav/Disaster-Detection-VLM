"""
Copy the 191 vlm_eval_sample building crops into backend/data/
so the VLM repo is self-contained for testing.

Run once:
    python copy_test_data.py
"""
import csv
import shutil
from pathlib import Path

BENCH   = Path("D:/Aaron/UTD/Spring 26/Capstone Project/Benchmark-Model-xView2")
SRC_CSV = BENCH / "data/processed/vlm_eval_sample.csv"

HERE    = Path(__file__).parent          # backend/
DST_DIR = HERE / "data/crops"           # backend/data/crops/
DST_CSV = HERE / "data/vlm_eval_sample.csv"

DST_DIR.mkdir(parents=True, exist_ok=True)

updated_rows = []
copied = skipped = 0

with open(SRC_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        tile = row["tile_id"]
        uid  = row["uid"]
        new_row = dict(row)

        for prefix in ["pre", "post"]:
            src = BENCH / f"data/processed/crops_oracle/{tile}/{uid}/{prefix}_bbox.png"
            dst = DST_DIR / tile / uid / f"{prefix}_bbox.png"
            dst.parent.mkdir(parents=True, exist_ok=True)

            key = f"{prefix}_path"
            new_row[key] = str(dst)

            if dst.exists():
                skipped += 1
            elif src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"  MISSING: {src}")

        updated_rows.append(new_row)

# Write local CSV with updated (absolute local) paths
fieldnames = list(updated_rows[0].keys())
with open(DST_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(updated_rows)

total_mb = sum(p.stat().st_size for p in DST_DIR.rglob("*.png")) / 1024 / 1024
print(f"Copied   : {copied} files")
print(f"Skipped  : {skipped} already existed")
print(f"Total    : {len(updated_rows)} buildings  |  {total_mb:.1f} MB in {DST_DIR}")
print(f"CSV      : {DST_CSV}")
