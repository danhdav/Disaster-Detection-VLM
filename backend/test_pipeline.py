"""
End-to-end test script for the CNN + VLM pipeline.

Run from the backend/ folder:
    uv run python test_pipeline.py

Tests:
    1. MongoDB connection
    2. CNN damage prediction (pre + post images, 6-channel input)
    3. Full CNN + GPT-4o grounded VLM assessment
"""

import io
import os
import sys
import torch
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

WEIGHTS_PATH = os.getenv("CNN_WEIGHTS_PATH", "cnn/weights/model.pt")

# ── helpers ───────────────────────────────────────────────────────────────────

def print_result(label: str, passed: bool, detail: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {label}")
    if detail:
        print(f"           {detail}")

def make_test_image(color: tuple) -> bytes:
    """Generate a small solid-color image to simulate a satellite crop."""
    img  = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill=(color[0] // 2, color[1] // 2, color[2] // 2))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def ensure_weights():
    """Save untrained model weights so the pipeline can load something for testing."""
    if os.path.exists(WEIGHTS_PATH):
        print(f"  [INFO] Weights found at {WEIGHTS_PATH}")
        return
    print(f"  [INFO] No weights found — saving untrained model to {WEIGHTS_PATH}")
    from cnn.model import DisasterCNN, DAMAGE_CLASSES
    model = DisasterCNN(num_classes=len(DAMAGE_CLASSES), pretrained=False)
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("  [INFO] Placeholder weights saved (untrained — predictions are random)")

# ── test 1: mongodb ───────────────────────────────────────────────────────────

def test_mongodb():
    print("\n[1] MongoDB Connection")
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGO_URI")
        if not uri:
            print_result("MONGO_URI env var set", False, "Not found in .env")
            return False
        c = MongoClient(uri, serverSelectionTimeoutMS=5000)
        c.admin.command("ping")
        print_result("Connect and ping", True)
        return True
    except Exception as e:
        print_result("Connect and ping", False, str(e))
        return False

# ── test 2: cnn ───────────────────────────────────────────────────────────────

def test_cnn():
    print("\n[2] CNN Damage Prediction (pre + post 6-channel)")
    try:
        ensure_weights()
        from cnn.predict import load_model, predict_damage
        model = load_model(WEIGHTS_PATH)
        print_result("Load model weights", True)

        # Simulate pre (green field) and post (burnt orange) images
        pre_bytes  = make_test_image((80, 120, 60))
        post_bytes = make_test_image((160, 80, 40))

        result = predict_damage(model, pre_bytes, post_bytes)
        expected_keys = ["pred_label", "pred_idx", "severity", "confidence", "margin", "scores"]
        has_fields = all(k in result for k in expected_keys)
        valid_label = result.get("pred_label") in ["no-damage", "minor-damage", "major-damage", "destroyed"]

        print_result("predict_damage returns expected fields", has_fields)
        print_result("pred_label is valid damage class",       valid_label, f"got: {result.get('pred_label')} ({result.get('severity')})")
        print_result("confidence in range [0,1]",              0.0 <= result.get("confidence", -1) <= 1.0)

        from cnn.predict import compute_geometry_context
        geom = compute_geometry_context(pre_bytes, post_bytes)
        has_geom = "change_score" in geom and "pct_changed" in geom
        print_result("Geometry context computed", has_geom, str(geom))

        return has_fields and valid_label
    except Exception as e:
        print_result("CNN test", False, str(e))
        return False

# ── test 3: vlm pipeline ──────────────────────────────────────────────────────

def test_vlm_pipeline():
    print("\n[3] CNN + VLM Grounded Pipeline (GPT-4o)")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print_result("OPENAI_API_KEY env var set", False, "Not found or still placeholder in .env")
            return False
        print_result("OPENAI_API_KEY env var set", True)

        from cnn.vlm_pipeline import DisasterPipeline
        pipeline = DisasterPipeline(weights_path=WEIGHTS_PATH)
        print_result("DisasterPipeline initialized", True)

        pre_bytes  = make_test_image((80, 120, 60))
        post_bytes = make_test_image((160, 80, 40))

        result = pipeline.assess(pre_bytes, post_bytes, lat=38.495, lon=-122.772)

        has_cnn  = "cnn"  in result and "pred_label"  in result["cnn"]
        has_geom = "geometry" in result and "change_score" in result["geometry"]
        has_vlm  = "vlm"  in result and "damage_level" in result["vlm"]

        print_result("CNN result in response",      has_cnn,  str(result.get("cnn", {})))
        print_result("Geometry context in response", has_geom, str(result.get("geometry", {})))
        print_result("VLM assessment in response",  has_vlm)

        if has_vlm:
            vlm = result["vlm"]
            print(f"\n  --- VLM Assessment ---")
            print(f"  Damage level : {vlm.get('damage_level', 'N/A')}")
            print(f"  Severity     : {vlm.get('severity',     'N/A')}")
            print(f"  Reasoning    : {vlm.get('reasoning',    'N/A')}")
            print(f"  Summary      : {vlm.get('assessment_summary', 'N/A')}")
            print(f"  Latency      : {result.get('latency_ms')} ms  |  Tokens: {result.get('tokens_used')}")

        return has_cnn and has_vlm
    except Exception as e:
        print_result("VLM pipeline test", False, str(e))
        return False

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 57)
    print("  Disaster Detection VLM — Pipeline Test")
    print("=" * 57)

    results = {
        "MongoDB":    test_mongodb(),
        "CNN":        test_cnn(),
        "CNN + VLM":  test_vlm_pipeline(),
    }

    print("\n" + "=" * 57)
    print("  Summary")
    print("=" * 57)
    for name, passed in results.items():
        print(f"  {name:<20} {'PASS' if passed else 'FAIL'}")

    print("=" * 57)
    sys.exit(0 if all(results.values()) else 1)
