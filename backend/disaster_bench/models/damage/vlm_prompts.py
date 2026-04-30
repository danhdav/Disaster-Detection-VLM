"""
VLM prompt templates for building damage classification.
Ref §04 §3 VLM benchmarks: cloud (OpenAI/Gemini/Claude), NVIDIA Build, local (Llama/Qwen2-VL).
Ref §06 §4: VLM images-only vs VLM + geometry tools.

Two prompt modes:
  ungrounded — VLM sees only the image crops; no geometry context.
  grounded   — VLM also receives a JSON payload with geometry features
               (change score, area, SSIM, etc.) as "tool" context.

Image layouts sent to cloud VLMs:
  Without masked crops (2 images):
    Image 1: BEFORE — padded context crop, target building outlined in red
    Image 2: AFTER  — padded context crop, target building outlined in red

  With masked crops (4 images):
    Image 1: BEFORE — padded context crop, target building outlined in red
    Image 2: AFTER  — padded context crop, target building outlined in red
    Image 3: BEFORE — target building only, background blacked out
    Image 4: AFTER  — target building only, background blacked out

Local VLMs receive a single composite image (side-by-side panels).
"""
from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# JSON output schema (shared across all prompt variants)
# ---------------------------------------------------------------------------

DAMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "considered_classes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The two most plausible classes you considered (e.g. [\"major-damage\",\"destroyed\"])",
        },
        "destroyed_gate_passed": {
            "type": "boolean",
            "description": "True only if at least 2 destroyed criteria are clearly visible",
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 short visual observations from the AFTER image",
        },
        "why_not_destroyed": {
            "type": "string",
            "description": "Required when damage_level != destroyed. Why the destroyed gate failed.",
        },
        "why_destroyed": {
            "type": "string",
            "description": "Required when damage_level == destroyed. Which 2+ criteria passed.",
        },
        "damage_level": {
            "type": "string",
            "enum": ["no-damage", "minor-damage", "major-damage", "destroyed"],
            "description": "Final xView2 damage classification",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Your confidence in the final label",
        },
    },
    "required": ["considered_classes", "destroyed_gate_passed", "key_evidence",
                 "damage_level", "confidence"],
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a remote sensing damage analyst classifying wildfire building damage from before/after satellite imagery.

Use the xView2 damage scale:
- no-damage:    No visible structural change between BEFORE and AFTER images.
- minor-damage: Superficial or partial damage. Roof partially affected, scorch marks, small debris nearby. Structure clearly still standing and mostly intact.
- major-damage: Significant structural damage. Large portions of roof gone, partial wall collapse, heavy burn/charring, but building footprint still recognizable as a standing structure.
- destroyed:    Complete structural loss. Building is rubble, ash, or missing. Only foundation or a debris field remains.

DESTROYED GATE — You may only label a building "destroyed" if AT LEAST 3 of these are clearly visible in the AFTER image:
  1. Building footprint is mostly rubble, ash, or debris field
  2. Roof completely gone and interior exposed across most of the footprint
  3. Walls largely collapsed (structure no longer standing as a building)
  4. Building appears missing or flattened compared to BEFORE

If the building footprint is still recognizable as a standing structure (even if heavily damaged), choose major-damage, not destroyed.
If the destroyed gate is NOT passed, choose major-damage, minor-damage, or no-damage.
If uncertain between two adjacent classes, pick the LESS SEVERE option.
Do NOT default to worst-case. Err toward major or minor when in doubt.

Always respond with valid JSON matching the provided schema. Think step by step."""

_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(DAMAGE_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)

_IMAGE_LAYOUT_2 = (
    "You will receive TWO satellite images of the same building:\n"
    "  Image 1: BEFORE the disaster — padded context crop "
    "(target building outlined in red)\n"
    "  Image 2: AFTER  the disaster — same padded context crop "
    "(target building outlined in red)\n\n"
    "Assess ONLY the outlined target building — ignore neighbouring structures.\n\n"
)

_IMAGE_LAYOUT_4 = (
    "You will receive FOUR satellite images of the same building:\n"
    "  Image 1: BEFORE the disaster — padded context crop "
    "(target building outlined in red)\n"
    "  Image 2: AFTER  the disaster — same padded context crop "
    "(target building outlined in red)\n"
    "  Image 3: BEFORE the disaster — target building ONLY "
    "(background blacked out)\n"
    "  Image 4: AFTER  the disaster — target building ONLY "
    "(background blacked out)\n\n"
    "Assess ONLY the outlined / isolated target building. "
    "Use Images 3-4 to inspect the building structure in detail "
    "and Images 1-2 for surrounding context.\n\n"
)

_IMAGE_LAYOUT_2_FULL_TILE = (
    "You will receive TWO full satellite tile images of a disaster area:\n"
    "  Image 1: BEFORE the disaster\n"
    "  Image 2: AFTER  the disaster\n\n"
    "No specific building is highlighted. Assess the overall structural damage "
    "visible across the scene and classify the most representative damage level "
    "for buildings in the affected area.\n\n"
)

_IMAGE_LAYOUT_2_DIFF = (
    "You will receive THREE satellite images of the same building:\n"
    "  Image 1: BEFORE the disaster — padded context crop "
    "(target building outlined in red)\n"
    "  Image 2: AFTER  the disaster — same padded context crop "
    "(target building outlined in red)\n"
    "  Image 3: CHANGE HEATMAP — pixel-difference map between BEFORE and AFTER. "
    "Brighter pixels (white/yellow) = more change; dark pixels = no change. "
    "Use this to localize exactly where on the building footprint structural change occurred.\n\n"
    "Assess ONLY the outlined target building — ignore neighbouring structures.\n"
    "Use Image 3 to identify the location and extent of structural change before "
    "making your damage classification.\n\n"
)


def _cnn_block(cnn_probs: dict) -> str:
    """Format CNN softmax probabilities as a reference hint for the VLM prompt."""
    return (
        "A reference CNN model has already analyzed this building and produced the following "
        "damage probability estimates:\n"
        f"  no-damage: {cnn_probs.get('p_nodmg', 0):.3f}  "
        f"minor-damage: {cnn_probs.get('p_minor', 0):.3f}  "
        f"major-damage: {cnn_probs.get('p_major', 0):.3f}  "
        f"destroyed: {cnn_probs.get('p_dest', 0):.3f}\n"
        "Use these as a calibration prior — they reflect a trained model's assessment — "
        "but weigh the visual evidence in the images above all else. "
        "If the images clearly contradict the CNN, trust the images.\n\n"
    )


def ungrounded_prompt(
    use_masked: bool = False,
    diff_overlay: bool = False,
    cnn_probs: dict | None = None,
    full_tile: bool = False,
) -> str:
    """
    Prompt for VLM with images only (no geometry features).
    use_masked=True  → 4-image layout (context + isolated building crops).
    diff_overlay=True → 3-image layout (context + pixel-difference heatmap).
    full_tile=True   → 2-image full scene layout (no building highlighted).
    use_masked takes priority; full_tile is used when no crop is available.
    cnn_probs: optional dict with keys p_nodmg/p_minor/p_major/p_dest to inject as a reference prior.
    """
    if use_masked:
        layout = _IMAGE_LAYOUT_4
    elif diff_overlay:
        layout = _IMAGE_LAYOUT_2_DIFF
    elif full_tile:
        layout = _IMAGE_LAYOUT_2_FULL_TILE
    else:
        layout = _IMAGE_LAYOUT_2
    cnn_hint = _cnn_block(cnn_probs) if cnn_probs else ""
    return layout + cnn_hint + _SCHEMA_SUFFIX


def grounded_prompt(
    geometry_features: dict[str, Any],
    use_masked: bool = False,
    diff_overlay: bool = False,
    cnn_probs: dict | None = None,
) -> str:
    """
    Prompt for VLM with geometry tool context (grounded mode).
    Ref §4: 'VLM + geometry tools (hallucination/consistency)'.
    The geometry features act as a tool-call result to reduce hallucination.
    cnn_probs: optional dict with keys p_nodmg/p_minor/p_major/p_dest to inject as a reference prior.
    """
    if use_masked:
        layout = _IMAGE_LAYOUT_4
    elif diff_overlay:
        layout = _IMAGE_LAYOUT_2_DIFF
    else:
        layout = _IMAGE_LAYOUT_2
    feat_str = json.dumps(geometry_features, indent=2)
    geo_block = (
        "Additionally, a geometry analysis tool has computed the following measurements:\n"
        f"```json\n{feat_str}\n```\n\n"
        "Definitions:\n"
        "  change_score: weighted multi-signal change intensity [0,1]\n"
        "  pct_changed: % building pixels with significant pixel diff\n"
        "  ssim_dissim: 1-SSIM, higher = more structural change\n"
        "  area_px: building footprint area in pixels\n\n"
        "Use BOTH the visual evidence AND the geometry measurements to classify damage.\n"
        "If visual evidence and geometry conflict, explain which you trust more and why.\n\n"
    )
    cnn_hint = _cnn_block(cnn_probs) if cnn_probs else ""
    return layout + geo_block + cnn_hint + _SCHEMA_SUFFIX


def local_composite_prompt(
    grounded: bool = False,
    geometry_features: dict[str, Any] | None = None,
) -> str:
    """
    Prompt for local VLMs that receive a single side-by-side composite image.
    LEFT panel = BEFORE the disaster; RIGHT panel = AFTER the disaster.
    The target building is isolated (black background) in both panels.
    """
    layout = (
        "A single composite satellite image is provided with TWO side-by-side panels:\n"
        "  LEFT panel:  BEFORE the disaster — target building isolated "
        "(background blacked out)\n"
        "  RIGHT panel: AFTER  the disaster — target building isolated "
        "(background blacked out)\n\n"
        "Assess the structural damage to the building between the two panels.\n\n"
    )
    geo_block = ""
    if grounded and geometry_features:
        feat_str = json.dumps(geometry_features, indent=2)
        geo_block = (
            "Geometry measurements from an analysis tool:\n"
            f"```json\n{feat_str}\n```\n\n"
            "Use both visual evidence and geometry to classify damage.\n\n"
        )
    return layout + geo_block + _SCHEMA_SUFFIX


def few_shot_examples() -> list[dict[str, str]]:
    """
    Static few-shot examples (text only, no images) for in-context learning.
    Used with text-capable VLMs as additional context.
    """
    return [
        {
            "role": "user",
            "content": (
                "Geometry: {change_score: 0.02, pct_changed: 1.2, ssim_dissim: 0.03}\n"
                "Classify damage."
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "damage_level": "no-damage",
                "confidence": 0.92,
                "reasoning": "Very low change score and SSIM dissimilarity indicate minimal structural change.",
                "is_damaged": False,
                "severity_level": 0,
            }),
        },
        {
            "role": "user",
            "content": (
                "Geometry: {change_score: 0.78, pct_changed: 81.5, ssim_dissim: 0.71}\n"
                "Classify damage."
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "damage_level": "destroyed",
                "confidence": 0.87,
                "reasoning": "Very high change score and >80% pixel change strongly indicates complete structural loss.",
                "is_damaged": True,
                "severity_level": 3,
            }),
        },
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Minor / major focused prompt (Stage-2 restricted mode)
# ---------------------------------------------------------------------------

MINOR_MAJOR_SYSTEM_PROMPT = """You are a remote sensing damage analyst. Your only task is to distinguish MINOR from MAJOR wildfire building damage in before/after satellite imagery.

You will only be called for buildings where automated methods detected meaningful damage but could not resolve severity. Assume the building IS damaged — your job is to classify how severely.

xView2 definitions:
- minor-damage: Superficial or partial damage. Roof partially affected, scorch marks, small debris nearby. Building is clearly still standing and mostly intact. Wall structure and most of the footprint are preserved.
- major-damage: Significant structural damage. Large roof sections missing, partial wall collapse, heavy burn/charring. Building footprint still recognizable but substantially compromised.

MAJOR GATE — choose major-damage only if AT LEAST 2 of these are clearly visible in the AFTER image:
  1. More than half the roof area is missing, burned through, or collapsed
  2. Visible wall collapse or major structural breach on one or more sides
  3. Heavy interior exposure visible through the roofline across most of the footprint
  4. Severe burn/charring that eliminates most surface features across the footprint

If the MAJOR GATE is clearly not passed, choose minor-damage.
If evidence is genuinely ambiguous, still commit to your best assessment based on the key_evidence — do NOT default to minor simply because you are uncertain.

Always respond with valid JSON matching the provided schema. Think step by step."""

MINOR_MAJOR_SCHEMA = {
    "type": "object",
    "properties": {
        "considered_classes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Always [\"minor-damage\", \"major-damage\"]",
        },
        "major_gate_passed": {
            "type": "boolean",
            "description": "True only if at least 2 major criteria are clearly visible",
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 short visual observations from the AFTER image",
        },
        "why_not_major": {
            "type": "string",
            "description": "Required when damage_level == minor-damage. Why the major gate failed.",
        },
        "why_major": {
            "type": "string",
            "description": "Required when damage_level == major-damage. Which 2+ criteria passed.",
        },
        "damage_level": {
            "type": "string",
            "enum": ["minor-damage", "major-damage"],
            "description": "Final severity classification",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": ["considered_classes", "major_gate_passed", "key_evidence",
                 "damage_level", "confidence"],
}

_MM_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(MINOR_MAJOR_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)


def minor_major_prompt(use_masked: bool = False, diff_overlay: bool = False) -> str:
    """
    Prompt for the restricted minor-vs-major mode.
    use_masked=True describes the 4-image layout; False describes 2-image layout.
    diff_overlay=True describes the 3-image layout with change heatmap.
    """
    if use_masked:
        layout = _IMAGE_LAYOUT_4
    elif diff_overlay:
        layout = _IMAGE_LAYOUT_2_DIFF
    else:
        layout = _IMAGE_LAYOUT_2
    return layout + _MM_SCHEMA_SUFFIX


# ---------------------------------------------------------------------------
# Indicator decomposition prompt (Strategy D)
# ---------------------------------------------------------------------------

INDICATORS_SYSTEM_PROMPT = """You are a remote sensing damage analyst. Your task is to answer five specific yes/no questions about wildfire building damage in before/after satellite imagery.

You will only be called for buildings where automated methods detected potential structural damage. Answer each question based strictly on what is visible in the AFTER image compared to the BEFORE image.

Guidelines:
- Answer "yes", "no", or "unclear" for each indicator question.
- Use "unclear" only when you genuinely cannot determine the answer from the imagery.
- Base answers on structural evidence, not just discoloration or shadows.
- key_evidence should list 2–4 short, specific visual observations from the AFTER image.

Always respond with valid JSON matching the provided schema. Think step by step."""

INDICATORS_SCHEMA = {
    "type": "object",
    "properties": {
        "roof_over_half_missing": {
            "type": "string",
            "enum": ["yes", "no", "unclear"],
            "description": "Is more than half of the roof area missing, burned through, or collapsed?",
        },
        "wall_collapse_visible": {
            "type": "string",
            "enum": ["yes", "no", "unclear"],
            "description": "Are any exterior walls visibly collapsed or partially destroyed (not just scorched)?",
        },
        "footprint_still_standing": {
            "type": "string",
            "enum": ["yes", "no", "unclear"],
            "description": "Is the building footprint recognizable as a standing structure (vs rubble/ash)?",
        },
        "interior_exposed": {
            "type": "string",
            "enum": ["yes", "no", "unclear"],
            "description": "Are there large areas of exposed interior visible from above?",
        },
        "severity_assessment": {
            "type": "string",
            "enum": ["minor", "major", "unclear"],
            "description": "Is the overall damage severity closer to minor cosmetic damage or to major structural damage?",
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2–4 short visual observations from the AFTER image that drove your answers",
        },
    },
    "required": [
        "roof_over_half_missing", "wall_collapse_visible", "footprint_still_standing",
        "interior_exposed", "severity_assessment", "key_evidence",
    ],
}

_IND_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(INDICATORS_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)


def indicators_prompt(use_masked: bool = False, diff_overlay: bool = False) -> str:
    """
    Prompt for the indicator decomposition mode (Strategy D).
    Returns 5 binary indicator answers instead of a direct damage label.
    diff_overlay=True adds a pixel-difference heatmap as Image 3.
    """
    if use_masked:
        layout = _IMAGE_LAYOUT_4
    elif diff_overlay:
        layout = _IMAGE_LAYOUT_2_DIFF
    else:
        layout = _IMAGE_LAYOUT_2
    questions = (
        "Answer these five questions about the building damage:\n"
        "  Q1: Is more than half of the roof area missing, burned through, or collapsed?\n"
        "  Q2: Are any exterior walls visibly collapsed or partially destroyed (not just scorched)?\n"
        "  Q3: Is the building footprint recognizable as a standing structure (vs rubble/ash)?\n"
        "  Q4: Are there large areas of exposed interior visible from above?\n"
        "  Q5: Is the overall damage severity closer to minor cosmetic damage or to major structural damage?\n\n"
    )
    return layout + questions + _IND_SCHEMA_SUFFIX


# ---------------------------------------------------------------------------
# V1 Boundary Judge schema and prompt
# ---------------------------------------------------------------------------

BOUNDARY_V1_SCHEMA = {
    "type": "object",
    "properties": {
        "damage_visible_on_building": {
            "type": "string",
            "enum": ["yes", "no", "unclear"],
            "description": "Is there visible damage to the building structure itself (not surrounding land)?",
        },
        "structure_still_coherent": {
            "type": "string",
            "enum": ["yes", "no", "unclear"],
            "description": "Does the building still appear as a coherent standing structure? Only answer if damage_visible_on_building=yes.",
        },
        "severity_if_damaged": {
            "type": "string",
            "enum": ["minor", "major", "unclear"],
            "description": "Is severity closer to minor or major? Only answer if damage_visible=yes and structure_still_coherent=yes.",
        },
        "abstain": {
            "type": "boolean",
            "description": "True if imagery quality, occlusion, or crop size makes judgment unreliable. Fusion logic ignores this response when true.",
        },
        "confidence_band": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Coarse overall confidence in your boundary answers.",
        },
        "evidence": {
            "type": "object",
            "description": "Interpretable boolean evidence supporting your answers.",
            "properties": {
                "roof_damage_visible": {
                    "type": "boolean",
                    "description": "Any visible damage to the roof in the AFTER image (missing sections, charring, collapse)",
                },
                "building_outline_discernible": {
                    "type": "boolean",
                    "description": "Building outline is still clearly visible in the AFTER image",
                },
                "change_localized_to_building": {
                    "type": "boolean",
                    "description": "The visible change is mainly at the building location, not just in surroundings",
                },
                "surroundings_only_change": {
                    "type": "boolean",
                    "description": "Change appears to be ONLY in the surroundings, not the building itself",
                },
            },
            "required": [
                "roof_damage_visible",
                "building_outline_discernible",
                "change_localized_to_building",
                "surroundings_only_change",
            ],
        },
    },
    "required": [
        "damage_visible_on_building",
        "structure_still_coherent",
        "severity_if_damaged",
        "abstain",
        "confidence_band",
        "evidence",
    ],
}


BOUNDARY_V1_SYSTEM_PROMPT = """You are analyzing overhead satellite imagery of the same building before and after a wildfire.

Your task is NOT to assign a damage class. Instead, answer three structured boundary questions about the building structure.

CRITICAL rules:
- Judge the BUILDING STRUCTURE ONLY. Burned grass, scorched earth, or charred trees around the building do NOT count as building damage.
- Use "unclear" if imagery resolution or quality prevents a confident answer.
- Set abstain=true if: the building footprint is too small to resolve, imagery is blurry or cloud-covered, or the crop is mostly surroundings with little building visible.

Question 1 — Damage detection (Boundary A: no-damage vs minor-damage):
  Is there visible damage to the BUILDING ITSELF (roof material, walls, or structure)?
  Answer: yes / no / unclear

Question 2 — Structural coherence (Boundary B: major-damage vs destroyed):
  Only answer if Q1=yes.
  Does the building still appear as a coherent standing structure (recognizable footprint, walls present)?
  Answer: yes / no / unclear
  "no" means: rubble field, ash pile, or completely flattened — no structure remaining.

Question 3 — Severity (Boundary C: minor vs major):
  Only answer if Q1=yes AND Q2=yes.
  Is the damage severity closer to minor (superficial or partial damage, structure mostly intact)
  or major (significant structural loss, large roof sections gone, partial wall collapse)?
  Answer: minor / major / unclear

Evidence booleans (answer for all cases regardless of damage level):
  - roof_damage_visible: Any visible damage to the roof in AFTER image
  - building_outline_discernible: Building outline still clearly visible in AFTER image
  - change_localized_to_building: Change appears at the building location (not just surroundings)
  - surroundings_only_change: Change is ONLY in surroundings, not the building

Return only valid JSON. No prose explanation outside the JSON."""


_BV1_SCHEMA_SUFFIX = (
    "Answer three boundary questions about this building.\n\n"
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(BOUNDARY_V1_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)


def boundary_v1_prompt(use_masked: bool = False, diff_overlay: bool = False) -> str:
    """
    V1 boundary judge prompt — structured boundary questions, not 4-class classification.
    Targets confusion pairs: no-damage/minor-damage and major-damage/destroyed.
    use_masked=True describes 4-image layout; False describes 2-image layout.
    diff_overlay=True adds a pixel-difference heatmap as Image 3.
    """
    if use_masked:
        layout = _IMAGE_LAYOUT_4
    elif diff_overlay:
        layout = _IMAGE_LAYOUT_2_DIFF
    else:
        layout = _IMAGE_LAYOUT_2
    return layout + _BV1_SCHEMA_SUFFIX


def parse_boundary_v1_response(response_text: str) -> dict[str, Any]:
    """
    Parse the V1 boundary judge JSON response.
    Returns a flat dict with all fields for CSV storage.
    Falls back to full abstain on malformed output.
    """
    import re

    _VALID_YESNO   = {"yes", "no", "unclear"}
    _VALID_SEV     = {"minor", "major", "unclear"}
    _VALID_CONF    = {"low", "medium", "high"}

    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)

        dvb = data.get("damage_visible_on_building", "unclear")
        if dvb not in _VALID_YESNO:
            dvb = "unclear"

        ssc = data.get("structure_still_coherent", "unclear")
        if ssc not in _VALID_YESNO:
            ssc = "unclear"

        sid = data.get("severity_if_damaged", "unclear")
        if sid not in _VALID_SEV:
            sid = "unclear"

        abstain = bool(data.get("abstain", False))
        # Force abstain if the key visible-damage field is unclear
        if dvb == "unclear":
            abstain = True

        conf = data.get("confidence_band", "low")
        if conf not in _VALID_CONF:
            conf = "low"

        ev = data.get("evidence", {})
        if not isinstance(ev, dict):
            ev = {}

        return {
            "damage_visible_on_building":      dvb,
            "structure_still_coherent":        ssc,
            "severity_if_damaged":             sid,
            "abstain":                         abstain,
            "confidence_band":                 conf,
            "ev_roof_damage_visible":          bool(ev.get("roof_damage_visible", False)),
            "ev_building_outline_discernible": bool(ev.get("building_outline_discernible", True)),
            "ev_change_localized_to_building": bool(ev.get("change_localized_to_building", False)),
            "ev_surroundings_only_change":     bool(ev.get("surroundings_only_change", False)),
            "parse_error": "",
        }
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "damage_visible_on_building":      "unclear",
            "structure_still_coherent":        "unclear",
            "severity_if_damaged":             "unclear",
            "abstain":                         True,
            "confidence_band":                 "low",
            "ev_roof_damage_visible":          False,
            "ev_building_outline_discernible": True,
            "ev_change_localized_to_building": False,
            "ev_surroundings_only_change":     False,
            "parse_error":                     str(exc),
        }


def parse_vlm_response(response_text: str) -> dict[str, Any]:
    """
    Parse VLM JSON response. Returns dict with damage_level and extended fields.
    Falls back gracefully on malformed output.
    """
    import re
    # Strip markdown code fences if present
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        # Validate damage_level (accept both 4-class and minor/major-only outputs)
        valid = {"no-damage", "minor-damage", "major-damage", "destroyed"}
        if data.get("damage_level") not in valid:
            data["damage_level"] = "minor-damage"   # safer default in minor/major mode
            data["parse_error"]  = "invalid damage_level, defaulted to minor-damage"
        # Flatten list fields to strings for CSV storage
        data.setdefault("reasoning", "")
        data.setdefault("parse_error", "")
        if isinstance(data.get("key_evidence"), list):
            data["key_evidence"] = "; ".join(data["key_evidence"])
        if isinstance(data.get("considered_classes"), list):
            data["considered_classes"] = "; ".join(data["considered_classes"])
        data.setdefault("key_evidence", "")
        data.setdefault("considered_classes", "")
        data.setdefault("destroyed_gate_passed", "")
        data.setdefault("why_not_destroyed", "")
        data.setdefault("why_destroyed", "")
        # Minor/major mode fields
        data.setdefault("major_gate_passed", "")
        data.setdefault("why_major", "")
        data.setdefault("why_not_major", "")
        data.setdefault("confidence", "")
        # Indicator mode fields
        data.setdefault("roof_over_half_missing", "")
        data.setdefault("wall_collapse_visible", "")
        data.setdefault("footprint_still_standing", "")
        data.setdefault("interior_exposed", "")
        data.setdefault("severity_assessment", "")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "damage_level":          "no-damage",
            "reasoning":             "",
            "key_evidence":          "",
            "considered_classes":    "",
            "destroyed_gate_passed": "",
            "why_not_destroyed":     "",
            "why_destroyed":         "",
            "major_gate_passed":     "",
            "why_major":             "",
            "why_not_major":         "",
            "confidence":            "",
            "roof_over_half_missing": "",
            "wall_collapse_visible":  "",
            "footprint_still_standing": "",
            "interior_exposed":       "",
            "severity_assessment":    "",
            "parse_error":           str(e),
            "raw_response":          response_text[:300],
        }
