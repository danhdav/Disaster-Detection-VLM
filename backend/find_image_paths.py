"""Resolve pre/post image paths for 4 test tiles."""
import os
from pathlib import Path

IMAGES_BASE = Path("../../../Benchmark-Model-xView2/train/images")

CASES = [
    ("no-damage",    "santa-rosa-wildfire_00000045", "ae28960c"),
    ("minor-damage", "santa-rosa-wildfire_00000207", "21c3f233"),
    ("major-damage", "socal-fire_00000431",          "852f29c6"),
    ("destroyed",    "santa-rosa-wildfire_00000012",  "95625a84"),
]

for label, tile_id, uid_prefix in CASES:
    pre  = IMAGES_BASE / f"{tile_id}_pre_disaster.png"
    post = IMAGES_BASE / f"{tile_id}_post_disaster.png"
    print(f"{label:20s}")
    print(f"  pre  exists={pre.exists()}   {pre}")
    print(f"  post exists={post.exists()}  {post}")
