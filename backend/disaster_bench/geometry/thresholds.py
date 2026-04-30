"""
Track 2 severity mapping: numeric features -> damage class.
Threshold rules for %changed, %damaged_pixels, area_change -> no-damage | minor | major | destroyed.
"""
from __future__ import annotations

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def thresholds_to_damage(
    pct_changed: float | None = None,
    pct_damaged: float | None = None,
    area_change_ratio: float | None = None,
    *,
    thresh_minor: float = 5.0,
    thresh_major: float = 25.0,
    thresh_destroyed: float = 70.0,
    area_loss_destroyed: float = -0.8,
) -> str:
    """
    Map numeric features to one of DAMAGE_CLASSES.
    Uses pct_damaged or pct_changed (priority pct_damaged if provided).
    area_change_ratio can override to 'destroyed' if very negative (e.g. < -0.8).
    """
    if area_change_ratio is not None and area_change_ratio <= area_loss_destroyed:
        return "destroyed"
    pct = pct_damaged if pct_damaged is not None else pct_changed
    if pct is None:
        return "no-damage"
    if pct >= thresh_destroyed:
        return "destroyed"
    if pct >= thresh_major:
        return "major-damage"
    if pct >= thresh_minor:
        return "minor-damage"
    return "no-damage"
