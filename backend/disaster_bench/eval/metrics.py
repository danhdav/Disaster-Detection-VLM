"""
Evaluation metrics — macro-F1, per-class F1, coverage, confusion matrix.
Ref §B: macro-F1 (primary), per-class F1, coverage.
Ref §06 §2.2: FEMA-style 3-class mapping.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Ref §2.2 — FEMA-style 3-class remapping
FEMA_REMAP = {
    "no-damage":    "intact",
    "minor-damage": "damaged",
    "major-damage": "damaged",
    "destroyed":    "destroyed",
}
FEMA_CLASSES = ["intact", "damaged", "destroyed"]


def fema_label(damage: str) -> str:
    """Map 4-class damage label to 3-class FEMA label."""
    return FEMA_REMAP.get(damage, damage)


def f1_per_class(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str] | None = None,
) -> dict[str, float]:
    if classes is None:
        classes = DAMAGE_CLASSES
    scores: dict[str, dict[str, float]] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        pred_pos = sum(1 for p in y_pred if p == c)
        true_pos = sum(1 for t in y_true if t == c)
        prec = tp / pred_pos if pred_pos else 0.0
        rec  = tp / true_pos if true_pos else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        scores[c] = {"precision": prec, "recall": rec, "f1": f1}
    return {c: scores[c]["f1"] for c in classes}


def precision_recall_per_class(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    if classes is None:
        classes = DAMAGE_CLASSES
    result: dict[str, dict[str, float]] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        pp = sum(1 for p in y_pred if p == c)
        tp_ = sum(1 for t in y_true if t == c)
        prec = tp / pp  if pp  else 0.0
        rec  = tp / tp_ if tp_ else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                     "support": tp_}
    return result


def macro_f1(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str] | None = None,
) -> float:
    per_class = f1_per_class(y_true, y_pred, classes=classes)
    return sum(per_class.values()) / len(per_class) if per_class else 0.0


def coverage(num_matched_gt: int, num_total_gt: int) -> float:
    if num_total_gt == 0:
        return 1.0
    return num_matched_gt / num_total_gt


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Return confusion matrix as nested dict: cm[true_class][pred_class] = count.
    Ref §05: Confusion matrix (minor vs no-damage vs major etc.).
    """
    if classes is None:
        classes = DAMAGE_CLASSES
    cm: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm.get(t, {}):
            cm[t][p] += 1
    return cm


def fema_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """
    Ref §2.2: FEMA-style 3-class (intact/damaged/destroyed) metrics.
    Remaps 4-class predictions before computing F1.
    """
    y_true_f = [fema_label(y) for y in y_true]
    y_pred_f = [fema_label(y) for y in y_pred]
    per_class = f1_per_class(y_true_f, y_pred_f, classes=FEMA_CLASSES)
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "fema_macro_f1": round(macro, 4),
        "fema_per_class_f1": {k: round(v, 4) for k, v in per_class.items()},
    }


def multilabel_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """
    Ref §2.3: Multi-label derived targets.
    is_damaged = minor OR major OR destroyed
    is_destroyed = destroyed
    severity_level = ordinal {0,1,2,3}
    """
    SEVERITY = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}

    def _binary_f1(yt_bin: list[int], yp_bin: list[int]) -> float:
        tp = sum(1 for t, p in zip(yt_bin, yp_bin) if t == 1 and p == 1)
        pp = sum(yp_bin)
        tp_ = sum(yt_bin)
        prec = tp / pp  if pp  else 0.0
        rec  = tp / tp_ if tp_ else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    yt_dmg  = [int(y != "no-damage") for y in y_true]
    yp_dmg  = [int(y != "no-damage") for y in y_pred]
    yt_dest = [int(y == "destroyed") for y in y_true]
    yp_dest = [int(y == "destroyed") for y in y_pred]
    yt_sev  = [SEVERITY.get(y, 0) for y in y_true]
    yp_sev  = [SEVERITY.get(y, 0) for y in y_pred]

    mae = sum(abs(t - p) for t, p in zip(yt_sev, yp_sev)) / max(len(yt_sev), 1)
    exact_match = sum(1 for t, p in zip(yt_sev, yp_sev) if t == p) / max(len(yt_sev), 1)

    return {
        "is_damaged_f1":   round(_binary_f1(yt_dmg,  yp_dmg),  4),
        "is_destroyed_f1": round(_binary_f1(yt_dest, yp_dest), 4),
        "severity_mae":    round(mae, 4),
        "severity_exact":  round(exact_match, 4),
    }


def compute_metrics(
    rows: list[dict[str, Any]],
    *,
    gt_uid_col: str      = "matched_gt_uid",
    gt_damage_col: str   = "gt_damage",
    pred_damage_col: str = "pred_damage",
    latency_ms: float | None = None,
) -> dict[str, Any]:
    """
    Full metrics from prediction rows (with GT damage filled in).
    Computes 4-class F1, FEMA 3-class F1, multi-label metrics, and confusion matrix.
    """
    eval_rows = [r for r in rows if r.get(gt_uid_col) and r.get(gt_damage_col)]
    y_true = [r[gt_damage_col]   for r in eval_rows]
    y_pred = [r[pred_damage_col] for r in eval_rows]

    per_class   = f1_per_class(y_true, y_pred)
    pr_detail   = precision_recall_per_class(y_true, y_pred)
    macro       = macro_f1(y_true, y_pred)
    cm          = confusion_matrix(y_true, y_pred)
    fema        = fema_metrics(y_true, y_pred)
    multilabel  = multilabel_metrics(y_true, y_pred)
    num_matched = len(set(r[gt_uid_col] for r in eval_rows))

    result: dict[str, Any] = {
        "macro_f1":             round(macro, 4),
        "per_class_f1":         {k: round(v, 4) for k, v in per_class.items()},
        "per_class_detail":     pr_detail,
        "confusion_matrix":     cm,
        "coverage":             None,
        "num_matched_gt":       num_matched,
        "num_eval_rows":        len(eval_rows),
        **fema,
        **multilabel,
    }
    if latency_ms is not None:
        result["avg_latency_ms"] = round(latency_ms, 2)
    return result
