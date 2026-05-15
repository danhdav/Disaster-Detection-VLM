"""
CNN-only test — no MongoDB, no VLM, no API keys needed.

Usage:
    # With synthetic images (just checks the pipeline works):
    uv run python test_cnn_only.py

    # With real pre/post image files:
    uv run python test_cnn_only.py --pre path/to/pre.png --post path/to/post.png

    # With a folder of image pairs (expects *_pre_disaster.png / *_post_disaster.png):
    uv run python test_cnn_only.py --folder "D:/path/to/images"
"""

import argparse
import io
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

WEIGHTS_PATH = os.getenv("CNN_WEIGHTS_PATH", "cnn/weights/model.pt")
DAMAGE_CLASSES  = ["no-damage", "minor-damage", "major-damage", "destroyed"]
DAMAGE_SEVERITY = ["LOW",       "MODERATE",     "HIGH",         "SEVERE"]


# ── helpers ───────────────────────────────────────────────────────────────────

def ensure_weights():
    if os.path.exists(WEIGHTS_PATH):
        print(f"[INFO] Weights found: {WEIGHTS_PATH}")
        return
    print(f"[INFO] No weights found — saving untrained placeholder to {WEIGHTS_PATH}")
    from cnn.model import DisasterCNN
    model = DisasterCNN(num_classes=len(DAMAGE_CLASSES), pretrained=False)
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("[INFO] Placeholder saved (untrained — predictions are random until you train)")


def make_synthetic_pair():
    """Generate a synthetic pre/post image pair for quick smoke testing."""
    def _img(color, rect_color):
        img = Image.new("RGB", (224, 224), color=color)
        ImageDraw.Draw(img).rectangle([40, 40, 184, 184], fill=rect_color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    pre  = _img((70, 130, 80),  (50, 110, 60))   # green — undamaged
    post = _img((140, 70, 30),  (100, 50, 20))    # burnt orange — fire damage
    return pre, post, "synthetic"


def load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def print_bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def run_prediction(pre_bytes: bytes, post_bytes: bytes, label: str = ""):
    from cnn.predict import load_model, predict_damage, compute_geometry_context

    model  = load_model(WEIGHTS_PATH)
    result = predict_damage(model, pre_bytes, post_bytes)
    geom   = compute_geometry_context(pre_bytes, post_bytes)

    tag = f"  [{label}]" if label else ""
    print(f"\n{'─'*56}{tag}")
    print(f"  Predicted  : {result['pred_label'].upper()}")
    print(f"  Severity   : {result['severity']}")
    print(f"  Confidence : {result['confidence']:.2%}  |  Margin: {result['margin']:.3f}")
    print()
    print("  Class Probabilities")
    print("  " + "─" * 44)
    for cls, prob in result["scores"].items():
        marker = "  ◄" if cls == result["pred_label"] else ""
        print(f"  {cls:<15}  {print_bar(prob)}  {prob*100:5.1f}%{marker}")
    print()
    print(f"  Geometry (pixel-level change)")
    print(f"  Change score  : {geom['change_score']}")
    print(f"  Pct changed   : {geom['pct_changed']}%")
    if "ssim_dissim" in geom:
        print(f"  SSIM dissim   : {geom['ssim_dissim']}")
    print()
    return result


def find_image_pairs(folder: str):
    """Find pre/post image pairs in a folder (xBD naming convention)."""
    folder = Path(folder)
    pre_files = sorted(folder.rglob("*pre_disaster*"))
    pairs = []
    for pre in pre_files:
        post = Path(str(pre).replace("pre_disaster", "post_disaster"))
        if post.exists():
            pairs.append((str(pre), str(post)))
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNN-only disaster damage test")
    parser.add_argument("--pre",    default=None, help="Path to pre-disaster image")
    parser.add_argument("--post",   default=None, help="Path to post-disaster image")
    parser.add_argument("--folder", default=None, help="Folder with pre/post image pairs")
    parser.add_argument("--limit",  type=int, default=5, help="Max pairs to test from folder (default 5)")
    parser.add_argument("--filter", default=None, help="Comma-separated disaster names to include e.g. 'santa-rosa,socal'")
    args = parser.parse_args()

    print("=" * 56)
    print("  CNN Damage Prediction Test (no MongoDB / no VLM)")
    print("=" * 56)

    ensure_weights()

    if args.folder:
        pairs = find_image_pairs(args.folder)
        if not pairs:
            print(f"\n[ERROR] No pre/post image pairs found in: {args.folder}")
            print("        Expected files named like: *pre_disaster*.png / *post_disaster*.png")
            sys.exit(1)

        if args.filter:
            keywords = [k.strip().lower() for k in args.filter.split(",")]
            pairs = [(pre, post) for pre, post in pairs
                     if any(k in Path(pre).name.lower() for k in keywords)]
            print(f"\n[INFO] Filter '{args.filter}' matched {len(pairs)} pair(s)")

        print(f"\n[INFO] Found {len(pairs)} image pair(s) — testing up to {args.limit}")
        passed = 0
        for i, (pre_path, post_path) in enumerate(pairs[:args.limit]):
            try:
                pre_bytes  = load_image_bytes(pre_path)
                post_bytes = load_image_bytes(post_path)
                name = Path(pre_path).stem.replace("_pre_disaster", "")
                run_prediction(pre_bytes, post_bytes, label=name)
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {pre_path}: {e}")

        print(f"\n[RESULT] {passed}/{min(len(pairs), args.limit)} pairs processed successfully")

    elif args.pre and args.post:
        if not os.path.exists(args.pre):
            print(f"[ERROR] Pre-image not found: {args.pre}")
            sys.exit(1)
        if not os.path.exists(args.post):
            print(f"[ERROR] Post-image not found: {args.post}")
            sys.exit(1)
        pre_bytes  = load_image_bytes(args.pre)
        post_bytes = load_image_bytes(args.post)
        run_prediction(pre_bytes, post_bytes, label=Path(args.pre).stem)

    else:
        print("\n[INFO] No images provided — running with synthetic test images")
        print("[INFO] Use --pre / --post or --folder to test with real images\n")
        pre_bytes, post_bytes, label = make_synthetic_pair()
        result = run_prediction(pre_bytes, post_bytes, label=label)
        print(f"[RESULT] Pipeline works. Predicted: {result['pred_label']} ({result['severity']})")
        print("[NOTE]   Predictions are RANDOM until the model is trained on real data.")

    print("=" * 56)


if __name__ == "__main__":
    main()
