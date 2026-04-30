"""
Write runs/<run_id>/predictions.csv and metrics.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_predictions_csv(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    columns: list[str] | None = None,
) -> None:
    """Write predictions table to CSV."""
    if not rows:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("tile_id,pred_instance_id,matched_gt_uid,iou,pred_damage,pred_conf,track,notes\n")
        return
    if columns is None:
        columns = [
            "tile_id", "pred_instance_id", "matched_gt_uid", "iou",
            "pred_damage", "pred_conf", "track", "notes",
            "gt_damage",  # optional, added at eval time
        ]
    # Only include columns that appear in rows
    seen = set()
    for r in rows:
        seen.update(r.keys())
    cols = [c for c in columns if c in seen]
    if not cols:
        cols = list(rows[0].keys())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_metrics_json(metrics: dict[str, Any], out_path: str | Path) -> None:
    """Write metrics dict to JSON."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def print_confusion_matrix(cm: dict[str, dict[str, int]], classes: list[str] | None = None) -> None:
    """Pretty-print confusion matrix to stdout."""
    if classes is None:
        from disaster_bench.eval.metrics import DAMAGE_CLASSES
        classes = DAMAGE_CLASSES
    abbr = {"no-damage": "no_dmg", "minor-damage": "minor", "major-damage": "major", "destroyed": "destr"}
    labels = [abbr.get(c, c[:6]) for c in classes]
    col_w = max(8, max(len(l) for l in labels) + 2)
    header = f"{'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))
    for c, label in zip(classes, labels):
        row_str = f"{label:>{col_w}}"
        for c2 in classes:
            row_str += f"{cm.get(c, {}).get(c2, 0):>{col_w}}"
        print(row_str)


def print_metrics_summary(metrics: dict[str, Any], title: str = "") -> None:
    """Print human-readable metrics summary."""
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
    print(f"  Macro F1:        {metrics.get('macro_f1', 0):.4f}")
    pf = metrics.get("per_class_f1", {})
    for k, v in pf.items():
        print(f"    {k:<18} F1={v:.4f}")
    if metrics.get("fema_macro_f1") is not None:
        print(f"  FEMA macro-F1:   {metrics.get('fema_macro_f1', 0):.4f}")
    if metrics.get("avg_latency_ms") is not None:
        print(f"  Avg latency:     {metrics['avg_latency_ms']:.1f} ms/instance")
    cov = metrics.get("coverage")
    if cov is not None:
        print(f"  Coverage:        {cov:.4f}")
    if "confusion_matrix" in metrics:
        print("\n  Confusion Matrix (rows=GT, cols=Pred):")
        print_confusion_matrix(metrics["confusion_matrix"])
    excl = metrics.get("excluded_unclassified")
    if excl is not None:
        pct = metrics.get("excluded_unclassified_pct", 0.0)
        print(f"  Excluded (un-classified): {excl} buildings ({pct:.1f}% of GT -- not scored)")
    is_dmg = metrics.get("is_damaged_f1")
    if is_dmg is not None:
        print(f"\n  Multi-label:")
        print(f"    is_damaged F1:   {is_dmg:.4f}")
        print(f"    is_destroyed F1: {metrics.get('is_destroyed_f1', 0):.4f}")
        print(f"    severity MAE:    {metrics.get('severity_mae', 0):.4f}")
