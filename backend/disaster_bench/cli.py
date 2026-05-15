"""
CLI: build-index, make-oracle-crops, run, eval-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(config_path: str | Path | None) -> dict:
    if not config_path or not Path(config_path).is_file():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}
    except Exception:
        return {}


def cmd_build_index(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import scan_dataset, write_index_csv, SCOPE_DISASTERS
    if args.filter_disasters:
        filt: set[str] | None = set(d.strip() for d in args.filter_disasters.split(","))
    elif args.no_filter:
        filt = None
    else:
        filt = SCOPE_DISASTERS  # default: socal-fire only
    print(f"  Scope filter: {filt if filt else 'ALL (no filter)'}")
    rows = scan_dataset(args.dataset_root, filter_disasters=filt)
    write_index_csv(rows, args.out_csv)
    disasters = sorted(set(r.get("disaster", "") for r in rows))
    print(f"Disasters in index: {disasters}")
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    return 0


def cmd_make_oracle_crops(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import read_index_csv
    from disaster_bench.data.crops import make_oracle_crops_for_tile
    rows = read_index_csv(args.index_csv)
    total = 0
    for row in rows:
        if not row.get("label_json_path"):
            continue
        n = make_oracle_crops_for_tile(
            row["tile_id"],
            row["pre_path"],
            row["post_path"],
            row["label_json_path"],
            args.out_dir,
            pad_fraction=args.pad_fraction,
        )
        total += n
    print(f"Wrote {total} oracle crops to {args.out_dir}")
    return 0


def cmd_overlays(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import read_index_csv
    from disaster_bench.data.overlays import make_overlay_for_tile
    rows = read_index_csv(args.index_csv)
    out_dir = Path(args.out_dir)
    count = 0
    for row in rows[: args.limit]:
        if not row.get("label_json_path"):
            continue
        out_path = out_dir / f"{row['tile_id']}_{args.which}.png"
        if make_overlay_for_tile(
            row["tile_id"],
            row["pre_path"],
            row["post_path"],
            row["label_json_path"],
            out_path,
            which=args.which,
        ):
            count += 1
    print(f"Wrote {count} overlays to {args.out_dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    index_csv = args.index_csv or config.get("index_csv", "data/processed/index.csv")
    if not Path(index_csv).is_file():
        print(f"Index not found: {index_csv}", file=sys.stderr)
        return 1
    track = args.track.lower()
    if track == "track1" or track == "1":
        from disaster_bench.pipelines.track1 import run_track1_and_save
        run_track1_and_save(index_csv, run_dir, config)
    elif track == "track2a" or track == "2a":
        from disaster_bench.pipelines.track2a import run_track2a_and_save
        run_track2a_and_save(index_csv, run_dir, config)
    elif track == "track2b" or track == "2b":
        from disaster_bench.pipelines.track2b import run_track2b_and_save
        run_track2b_and_save(index_csv, run_dir, config)
    elif track == "track3" or track == "3":
        from disaster_bench.pipelines.track3 import run_track3_and_save
        run_track3_and_save(index_csv, run_dir, config)
    elif track in ("track4", "4", "vlm", "track4_vlm"):
        from disaster_bench.pipelines.track4_vlm import run_track4_vlm_and_save
        run_track4_vlm_and_save(index_csv, run_dir, config)
    elif track in ("track1_deploy", "1-deploy", "deploy"):
        from disaster_bench.pipelines.track1_deploy import run_track1_deploy_and_save
        run_track1_deploy_and_save(index_csv, run_dir, config)
    else:
        print(f"Unknown track: {args.track}", file=sys.stderr)
        return 1
    print(f"Run written to {run_dir}")
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import read_index_csv, load_label_json, get_buildings_from_label
    from disaster_bench.eval.metrics import compute_metrics, coverage
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    run_dir = Path(args.run_dir)
    pred_path = run_dir / "predictions.csv"
    if not pred_path.is_file():
        print(f"Predictions not found: {pred_path}", file=sys.stderr)
        return 1
    with open(pred_path, encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        pred_rows = list(reader)
    index = read_index_csv(args.index_csv)
    # Build GT damage by (tile_id, uid) from label JSONs.
    # "un-classified" buildings are excluded from both GT lookup and coverage
    # denominator — they are still predicted but not scored.
    gt_damage: dict[tuple[str, str], str] = {}
    total_gt = 0
    excluded_unclassified = 0
    for row in index:
        label_path = row.get("label_json_path")
        if not label_path:
            continue
        try:
            label_data = load_label_json(label_path)
            for b in get_buildings_from_label(label_data, use_xy=True):
                uid     = b["uid"]
                subtype = b.get("subtype") or ""
                if subtype == "un-classified":
                    excluded_unclassified += 1
                    continue  # predicted but not scored
                if not subtype:
                    subtype = "no-damage"
                gt_damage[(row["tile_id"], uid)] = subtype
                total_gt += 1
        except Exception:
            pass
    # For deploy tracks: re-run bbox IoU matching from saved pred_instances on disk.
    # This is more reliable than relying on matched_gt_uid written by the pipeline.
    tracks_in_run = {r.get("track", "") for r in pred_rows}
    is_deploy = "1-deploy" in tracks_in_run
    if is_deploy:
        from disaster_bench.data.pred_instances import load_pred_instances, match_pred_to_gt
        from disaster_bench.data.io import get_label_canvas_size
        from disaster_bench.data.polygons import parse_wkt_polygon, scale_factors
        pred_instances_dir = run_dir.parent.parent / "data" / "processed" / "pred_instances"
        # Build per-tile uid_to_gt mapping via bbox IoU matching
        deploy_uid_to_gt: dict[tuple[str, str], tuple[str, float]] = {}
        for idx_row in index:
            tile_id    = idx_row["tile_id"]
            label_path = idx_row.get("label_json_path", "")
            if not label_path or not Path(label_path).is_file():
                continue
            pred_insts = load_pred_instances(tile_id, out_root=pred_instances_dir)
            if not pred_insts:
                continue
            try:
                label_data = load_label_json(label_path)
                json_w, json_h = get_label_canvas_size(label_data)
                if json_w <= 0:
                    json_w, json_h = 1024, 1024
                gt_blds = get_buildings_from_label(label_data, use_xy=True)
                sx = 1024 / json_w if json_w else 1.0
                sy = 1024 / json_h if json_h else 1.0
                gt_insts = []
                for b in gt_blds:
                    poly = parse_wkt_polygon(b["wkt"])
                    coords = list(poly.exterior.coords)
                    xs = [c[0] * sx for c in coords]
                    ys = [c[1] * sy for c in coords]
                    gt_insts.append({"uid": b["uid"], "bbox": (min(xs), min(ys), max(xs), max(ys))})
                match_result = match_pred_to_gt(pred_insts, gt_insts, iou_threshold=0.2)
                for m in match_result.get("matches", []):
                    deploy_uid_to_gt[(tile_id, m["pred_uid"])] = (m["gt_uid"], m["iou"])
            except Exception:
                pass
        for r in pred_rows:
            key = (r["tile_id"], r["pred_instance_id"])
            if key in deploy_uid_to_gt:
                gt_uid, iou_val = deploy_uid_to_gt[key]
                gt_key = (r["tile_id"], gt_uid)
                r["gt_damage"] = gt_damage.get(gt_key, "")
                if gt_key in gt_damage:
                    r["matched_gt_uid"] = gt_uid
                    r["iou"] = round(iou_val, 4)
                else:
                    r["matched_gt_uid"] = ""
                    r["iou"] = ""
            else:
                r["matched_gt_uid"] = ""
                r["iou"] = ""
    else:
        # Oracle-style: match by pred_instance_id == GT UID.
        for r in pred_rows:
            key = (r["tile_id"], r["pred_instance_id"])
            r["gt_damage"] = gt_damage.get(key, "")
            if key in gt_damage:
                r["matched_gt_uid"] = r["pred_instance_id"]
                r["iou"] = 1.0
            else:
                # Clear stale match (un-classified or unmatched row from a previous eval)
                r["matched_gt_uid"] = ""
                r["iou"] = ""
    matched_uids = set()
    for r in pred_rows:
        if r.get("matched_gt_uid"):
            matched_uids.add((r["tile_id"], r["matched_gt_uid"]))
    # Preserve latency from existing metrics.json (set during pipeline run)
    prev_latency = None
    prev_metrics_path = run_dir / "metrics.json"
    if prev_metrics_path.is_file():
        try:
            import json as _json
            with open(prev_metrics_path, encoding="utf-8") as _f:
                _prev = _json.load(_f)
            prev_latency     = _prev.get("avg_latency_ms")
            prev_elapsed     = _prev.get("total_elapsed_s")
        except Exception:
            prev_latency = prev_elapsed = None
    else:
        prev_elapsed = None

    metrics = compute_metrics(pred_rows, latency_ms=prev_latency)
    metrics["coverage"] = round(coverage(len(matched_uids), total_gt), 4) if total_gt else None
    if prev_elapsed is not None:
        metrics["total_elapsed_s"] = prev_elapsed
    total_with_excl = total_gt + excluded_unclassified
    metrics["excluded_unclassified"] = excluded_unclassified
    metrics["excluded_unclassified_pct"] = round(
        100.0 * excluded_unclassified / total_with_excl, 2
    ) if total_with_excl else 0.0
    write_predictions_csv(pred_rows, pred_path)
    write_metrics_json(metrics, run_dir / "metrics.json")

    from disaster_bench.eval.report import print_metrics_summary
    print_metrics_summary(metrics, title=str(run_dir))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="disaster-bench", description="SoCal wildfire damage benchmark")
    sub = p.add_subparsers(dest="command", required=True)
    # build-index
    b = sub.add_parser("build-index", help="Scan dataset and write index.csv")
    b.add_argument("--dataset_root", required=True, help="Path to test_images_labels_targets")
    b.add_argument("--out_csv", required=True, help="Output index CSV path")
    b.add_argument("--filter_disasters", default="", help="Comma-separated disaster names to include (default: socal-fire)")
    b.add_argument("--no_filter", action="store_true", help="Include all disaster types (override scope)")
    b.set_defaults(func=cmd_build_index)
    # make-oracle-crops
    c = sub.add_parser("make-oracle-crops", help="Generate oracle pre/post crops from GT polygons")
    c.add_argument("--index_csv", required=True, help="Index CSV from build-index")
    c.add_argument("--out_dir", required=True, help="Output directory (crops_oracle)")
    c.add_argument("--pad_fraction", type=float, default=0.25,
                   help="Bbox-relative padding on each side: pad = max(pad_fraction * max(w,h), 8) px (default 0.25)")
    c.set_defaults(func=cmd_make_oracle_crops)
    # overlays
    o = sub.add_parser("overlays", help="Debug overlay images (masks on pre/post)")
    o.add_argument("--index_csv", required=True, help="Index CSV")
    o.add_argument("--out_dir", required=True, help="Output directory")
    o.add_argument("--which", choices=("pre", "post"), default="post")
    o.add_argument("--limit", type=int, default=10, help="Max tiles to process")
    o.set_defaults(func=cmd_overlays)
    # run
    r = sub.add_parser("run", help="Run a track (1, 2a, 2b, 3)")
    r.add_argument("--track", required=True, help="track1, track2a, track2b, track3")
    r.add_argument("--config", help="YAML config (optional)")
    r.add_argument("--run_dir", required=True, help="Output run directory (e.g. runs/run1)")
    r.add_argument("--index_csv", help="Index CSV (default from config or data/processed/index.csv)")
    r.set_defaults(func=cmd_run)
    # eval-run
    e = sub.add_parser("eval-run", help="Add GT damage and compute metrics for a run")
    e.add_argument("--run_dir", required=True, help="Run directory with predictions.csv")
    e.add_argument("--index_csv", required=True, help="Index CSV (to load GT from labels)")
    e.set_defaults(func=cmd_eval_run)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
