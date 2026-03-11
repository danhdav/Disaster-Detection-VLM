"""
CNN + VLM performance test — fully self-contained.
Uses crops copied into backend/data/crops/ and backend/data/vlm_eval_sample.csv.

Usage:
    python run_perf_test.py
    python run_perf_test.py --cnn_only     (skip VLM, no API needed)
    python run_perf_test.py --limit 8      (test more cases)
    python run_perf_test.py --all          (run all 191 buildings)

Data setup (run once if data/crops/ is missing):
    python copy_test_data.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

HERE         = Path(__file__).parent     # backend/
LOCAL_CSV    = HERE / "data/vlm_eval_sample.csv"

WEIGHTS_PATH = os.getenv("CNN_WEIGHTS_PATH", "cnn/weights/model.pt")
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SEV            = {"no-damage": "LOW", "minor-damage": "MODERATE",
                  "major-damage": "HIGH", "destroyed": "SEVERE"}


def load_cases(limit: int) -> list[dict]:
    """
    Pick up to `limit` cases from the local vlm_eval_sample.csv.
    One per class first, then extras in file order.
    """
    if not LOCAL_CSV.exists():
        print(f"ERROR: {LOCAL_CSV} not found. Run: python copy_test_data.py")
        sys.exit(1)

    rows = []
    with open(LOCAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pre  = Path(row["pre_path"])
            post = Path(row["post_path"])
            if not pre.exists() or not post.exists():
                continue
            row["pre_abs"]  = str(pre)
            row["post_abs"] = str(post)
            rows.append(row)

    # First: one per class
    selected, seen = [], set()
    for row in rows:
        lbl = row["gt_label"]
        if lbl not in seen:
            selected.append(row)
            seen.add(lbl)
        if len(seen) == 4:
            break

    # Fill remaining slots
    extra = [r for r in rows if r not in selected]
    selected += extra[: max(0, limit - len(selected))]
    return selected[:limit]


def bar(p: float, w: int = 18) -> str:
    return "█" * round(p * w) + "░" * (w - round(p * w))


def run(cnn_only: bool, limit: int) -> None:
    from cnn.predict import load_model, predict_damage, compute_geometry_context

    model = load_model(WEIGHTS_PATH)
    cases = load_cases(limit)

    if not cases:
        print("ERROR: no test cases found (check vlm_eval_sample.csv paths)")
        sys.exit(1)

    pipeline = None
    if not cnn_only:
        try:
            from cnn.vlm_pipeline import DisasterPipeline
            pipeline = DisasterPipeline(weights_path=WEIGHTS_PATH)
        except Exception as e:
            print(f"[WARN] VLM disabled: {e}\n")

    correct_cnn = 0
    correct_vlm = 0
    total = len(cases)

    print("=" * 70)
    print(f"  CNN + VLM Performance Test   ({total} buildings)")
    print("=" * 70)

    for i, row in enumerate(cases, 1):
        gt      = row["gt_label"]
        tile_id = row["tile_id"]
        uid     = row["uid"][:8]

        pre_bytes  = Path(row["pre_abs"]).read_bytes()
        post_bytes = Path(row["post_abs"]).read_bytes()

        cnn    = predict_damage(model, pre_bytes, post_bytes)
        geom   = compute_geometry_context(pre_bytes, post_bytes)
        c_ok   = cnn["pred_label"] == gt
        correct_cnn += int(c_ok)

        vlm_label = None
        if pipeline:
            try:
                out       = pipeline.assess(pre_bytes, post_bytes)
                vlm_label = out["vlm"].get("damage_level", "?")
                v_ok      = vlm_label == gt
                correct_vlm += int(v_ok)
            except Exception as e:
                vlm_label = f"ERROR({e})"
                v_ok      = False
        else:
            v_ok = False

        # ── print ──────────────────────────────────────────────────────────
        print(f"\n[{i}/{total}] tile={tile_id}  uid={uid}")
        print(f"  Ground truth : {gt.upper()}  ({SEV[gt]})")
        print()
        print(f"  CNN prediction")
        for cls in DAMAGE_CLASSES:
            p      = cnn["scores"][cls]
            marker = "  ◄" if cls == cnn["pred_label"] else ""
            short  = cls.replace("-damage","").replace("no-","none").replace("destroyed","destr")
            print(f"    {short:8s}  {bar(p)}  {p*100:5.1f}%{marker}")
        cnn_tag = "✓ CORRECT" if c_ok else f"✗ wrong (pred={cnn['pred_label']})"
        print(f"  Conf={cnn['confidence']:.2%}  Margin={cnn['margin']:.3f}  → {cnn_tag}")

        print(f"\n  Geometry  change={geom['change_score']}  "
              f"pct_changed={geom['pct_changed']}%  "
              f"ssim_dissim={geom.get('ssim_dissim','n/a')}")

        if vlm_label and not vlm_label.startswith("ERROR"):
            vlm_tag = "✓ CORRECT" if v_ok else f"✗ wrong (pred={vlm_label})"
            print(f"\n  VLM prediction : {vlm_label.upper()}  → {vlm_tag}")
            if "vlm" in out:
                v = out["vlm"]
                print(f"  Reasoning      : {v.get('reasoning','')[:120]}")
                indicators = v.get("damage_indicators", [])
                if indicators:
                    print(f"  Indicators     : {', '.join(indicators[:3])}")
                print(f"  Latency        : {out.get('latency_ms')} ms  |  Tokens: {out.get('tokens_used')}")
        elif vlm_label:
            print(f"\n  VLM : {vlm_label}")

        print("  " + "─" * 66)

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  CNN accuracy : {correct_cnn}/{total}  ({correct_cnn/total:.0%})")
    if pipeline:
        print(f"  VLM accuracy : {correct_vlm}/{total}  ({correct_vlm/total:.0%})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cnn_only", action="store_true")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--all", action="store_true", help="Run all 191 buildings")
    args = p.parse_args()
    if args.all:
        args.limit = 9999

    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    run(args.cnn_only, args.limit)
