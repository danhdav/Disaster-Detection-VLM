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
            "description": "True only if at least 3 destroyed criteria are clearly visible",
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
            "description": "Required when damage_level == destroyed. Which 3+ criteria passed.",
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

Use the xView2 damage scale — your goal is to identify the CORRECT class:
- no-damage:    No visible structural change between BEFORE and AFTER images.
- minor-damage: Superficial or partial damage. Roof partially affected, scorch marks, small debris nearby. Structure clearly still standing and mostly intact.
- major-damage: Significant structural damage. Large portions of roof gone, partial wall collapse, heavy burn/charring, but building footprint still recognizable as a standing structure.
- destroyed:    Complete structural loss. Building is rubble, ash, or missing. Only foundation or a debris field remains.

CLASSIFICATION GATES — evaluate in order:

DESTROYED GATE — label destroyed only if AT LEAST 3 of these are clearly visible in the AFTER image:
  1. Building footprint is mostly rubble, ash, or debris field
  2. Roof completely gone and interior exposed across most of the footprint
  3. Walls largely collapsed (structure no longer standing as a building)
  4. Building appears missing or flattened compared to BEFORE
→ If gate PASSES: destroyed. If gate FAILS but significant structural damage is visible: major-damage.

MAJOR-DAMAGE GATE — choose major-damage if AT LEAST 2 of these are clearly visible in the AFTER image:
  1. More than half the roof area is missing, burned through, or collapsed
  2. One or more walls visibly collapsed or breached
  3. Large areas of exposed building interior visible through the roofline
  4. Heavy burn/charring across most of the footprint
→ If gate PASSES: major-damage. If gate FAILS but clear surface damage exists: minor-damage.

MINOR-DAMAGE INDICATORS — choose minor-damage if you see ANY of these clearly visible in the AFTER image:
  - Visible scorch marks, discoloration, or partial darkening of roof material compared to BEFORE
  - Small sections of missing or damaged roof (less than half the total area)
  - Debris adjacent to the building footprint that was not present in BEFORE
  - Surface-level burn damage without structural loss
→ If any indicator is present and major/destroyed gates are not met: minor-damage.

NO DAMAGE — only if the building structure appears completely unchanged from BEFORE.

When evidence is ambiguous between two classes, commit to your best assessment based on the visual evidence. Apply the gates above strictly — they are the primary decision criteria.

Always respond with valid JSON matching the provided schema. Think step by step."""

# ---------------------------------------------------------------------------
# Scene-specific system prompts
# ---------------------------------------------------------------------------

SANTA_ROSA_SYSTEM_PROMPT = """You are a remote sensing damage analyst classifying building damage from the 2017 Santa Rosa Wildfire (Tubbs Fire) using before/after satellite imagery.

SCENE CONTEXT — Santa Rosa, California:
- Dense suburban residential neighborhoods (Coffey Park, Fountaingrove)
- Typical buildings: 1–2 story single-family homes on grid-pattern streets
- Common roof types: dark gray/brown composition shingle, some clay tile
- Walls: light-colored stucco or painted wood siding
- The fire moved extremely fast at night — buildings are most often either completely destroyed or completely intact; minor and major damage are less common but do occur
- Destroyed buildings leave a bare concrete foundation slab (light gray rectangle) with no structure on top
- Intact buildings show their original dark shingle or tile roof clearly as a dark rectangle matching BEFORE

SANTA ROSA FALSE CUES — do NOT be fooled by:
- A light gray concrete slab = foundation of a DESTROYED home (the building is gone). Distinguish from intact roofs by checking if the dark roof shape from BEFORE is absent.
- Ash-gray surroundings = burned landscaping — the building itself may still be standing
- Darkened or smoke-coated roofs = if the dark building shape still matches BEFORE, structure is likely intact
- A dark rectangular shape matching BEFORE footprint exactly = almost certainly an intact building with shingle roof

Use the xView2 damage scale — identify the CORRECT class:
- no-damage:    Roof structure intact. Same shape and approximate color as BEFORE. Building clearly standing.
- minor-damage: Building standing, roof shows scorch marks, partial discoloration, or small missing patches. Less than 25% of roof affected.
- major-damage: Significant roof loss (>50% missing), partial wall collapse, interior exposed. Footprint still recognizable as a standing structure.
- destroyed:    Only bare concrete foundation slab remains, OR complete rubble/ash field with no structure. Building has vanished.

CLASSIFICATION GATES — evaluate in order:

DESTROYED GATE — label destroyed only if AT LEAST 3 of these are clearly visible:
  1. Bare concrete slab or rubble field where building stood
  2. No roof material visible — structure completely gone
  3. Walls absent — only foundation or debris remains
  4. Building footprint appears flattened or missing vs BEFORE
→ If gate PASSES: destroyed. If gate FAILS but large structural damage is visible: major-damage.

MAJOR-DAMAGE GATE — choose major-damage if AT LEAST 2 of these are clearly visible:
  1. More than half the roof area missing, burned through, or collapsed
  2. Partial wall collapse visible on one or more sides
  3. Large exposed interior visible through the roofline
  4. Heavy charring/burn across most of the visible footprint
→ If gate PASSES: major-damage.

MINOR-DAMAGE INDICATORS — choose minor-damage if you see ANY of these:
  - Scorch marks or discoloration on roof compared to BEFORE
  - Small areas of missing roof (less than half)
  - Debris adjacent to the building not present in BEFORE
→ If any indicator is present and major/destroyed gates not met: minor-damage (not no-damage).

NO DAMAGE — only if building appears completely unchanged from BEFORE.

Commit to your best assessment based on visual evidence. Do not default to no-damage simply because you are uncertain.

Always respond with valid JSON matching the provided schema. Think step by step."""


SOCAL_SYSTEM_PROMPT = """You are a remote sensing damage analyst classifying building damage from Southern California wildfires (Woolsey/Thomas Fire area) using before/after satellite imagery.

SCENE CONTEXT — Southern California:
- Varied terrain: hillside, canyon, and flat residential areas (Malibu, Ventura, Thousand Oaks)
- Typical buildings: 1–2 story homes, some larger estate properties, commercial structures
- Common roof types: clay/concrete tile (reddish-orange or dark gray), flat roofs on commercial buildings, some composition shingle
- Walls: stucco (beige/tan/white), painted concrete block
- Chaparral (scrub brush) surrounds most homes — burned chaparral leaves extensive ash fields AROUND buildings that are NOT building damage
- SoCal fires produce mixed damage — minor and major damage classes are frequent; do not assume buildings are only destroyed or intact
- Hillside terrain means buildings may appear at angles — compare structure shape carefully between BEFORE and AFTER

SOCAL FALSE CUES — do NOT be fooled by:
- Large ash/gray fields surrounding the building = burned chaparral, NOT a debris field from the building
- Reddish-orange or gray tile roofs appearing lighter in AFTER = ash deposit on intact tile roof, NOT destruction
- Tan/beige stucco walls appearing discolored = smoke/heat, building may still be intact
- Hillside shadow differences between images = lighting angle change, not structural change

Use the xView2 damage scale — identify the CORRECT class:
- no-damage:    No structural change. Roof color and shape match BEFORE. Building clearly standing.
- minor-damage: Building standing. Scorch marks on roof, partial discoloration, small debris on roof or nearby. Structure mostly intact, less than 25% of roof affected.
- major-damage: Substantial structural damage. Large sections of roof missing (>50%), partial wall collapse, heavy charring across footprint. Building outline still recognizable.
- destroyed:    Complete structural loss. Building is rubble or ash. Only foundation or bare ground remains inside the red outline.

CLASSIFICATION GATES — evaluate in order:

DESTROYED GATE — label destroyed only if AT LEAST 3 of these are clearly visible:
  1. Building footprint is rubble, ash, or debris field (not surrounding chaparral ash)
  2. Roof completely gone — no tile or shingle material visible over footprint
  3. Walls largely collapsed — structure no longer standing as a building
  4. Building appears missing or flattened compared to BEFORE
→ If gate PASSES: destroyed. If gate FAILS but significant structural damage is visible: major-damage.

MAJOR-DAMAGE GATE — choose major-damage if AT LEAST 2 of these are clearly visible:
  1. More than half the roof area missing, burned through, or collapsed
  2. Partial wall collapse visible on one or more sides
  3. Large areas of exposed interior visible through the roofline
  4. Heavy charring/burn across most of the footprint
→ If gate PASSES: major-damage.

MINOR-DAMAGE INDICATORS — choose minor-damage if you see ANY of these (and major/destroyed gates are not met):
  - Scorch marks, partial discoloration, or darkening of roof surface compared to BEFORE
  - Small roof damage patches (less than half the area)
  - Debris near the building that was not present in BEFORE
  - Partial surface burn without structural loss
→ If any indicator is present: minor-damage (not no-damage).

NO DAMAGE — only if building appears completely unchanged from BEFORE.

Apply the classification gates strictly as the primary decision criteria. When ambiguous, use whichever class has the most visual evidence support.

Always respond with valid JSON matching the provided schema. Think step by step."""


def get_system_prompt(tile_id: str = "") -> str:
    """Return the scene-specific system prompt based on tile_id, or generic if unknown."""
    if "santa-rosa" in tile_id:
        return SANTA_ROSA_SYSTEM_PROMPT
    if "socal" in tile_id:
        return SOCAL_SYSTEM_PROMPT
    return SYSTEM_PROMPT

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


# ---------------------------------------------------------------------------
# Multi-agent architecture: 4 class specialists + supervisor
# ---------------------------------------------------------------------------

SPECIALIST_SCHEMA = {
    "type": "object",
    "properties": {
        "match_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "How strongly does this building match your assigned class? 0=definitely not, 10=definitely yes.",
        },
        "verdict": {
            "type": "string",
            "enum": ["yes", "possible", "no"],
            "description": "Does this building belong to your assigned class?",
        },
        "supporting_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-3 visual observations that support your class",
        },
        "counter_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-2 visual observations that argue against your class (or empty if none)",
        },
    },
    "required": ["match_score", "verdict", "supporting_evidence", "counter_evidence"],
}

_SPECIALIST_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(SPECIALIST_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)

NO_DAMAGE_SPECIALIST_PROMPT = """You are a damage detection specialist. Your ONLY job is to determine if this building has NO structural damage.

Evidence for no-damage:
- Roof shape, color, and texture match BEFORE image — same geometric form
- Walls appear intact and unchanged
- No missing roof sections, no exposed interior, no collapsed walls
- Color change from smoke/ash is acceptable — structure must be geometrically intact
- Building still looks like a building, not rubble

Evidence against no-damage (means damage exists):
- Any roof material visibly missing or displaced
- Any wall collapse or breach
- Footprint shape changed between BEFORE and AFTER

Score 10 if the building is clearly pristine and unchanged. Score 0 if you can clearly see structural damage."""

MINOR_DAMAGE_SPECIALIST_PROMPT = """You are a damage detection specialist. Your ONLY job is to determine if this building has MINOR structural damage.

Evidence for minor-damage:
- Building is clearly still standing with most of its structure intact
- Roof shows scorch marks, partial discoloration, or small missing patches (less than 25% of roof)
- Debris nearby but not covering the footprint
- Some surface-level changes but walls and most of the roof are present
- You can see the building is damaged but it would still be structurally safe

Evidence against minor-damage:
- No visible damage at all → no-damage instead
- More than 25% of roof missing → major-damage instead
- Walls collapsed or building mostly gone → major or destroyed instead

Score 10 if the building clearly shows minor superficial damage. Score 0 if it is clearly undamaged or far more severely damaged."""

MAJOR_DAMAGE_SPECIALIST_PROMPT = """You are a damage detection specialist. Your ONLY job is to determine if this building has MAJOR structural damage.

Evidence for major-damage:
- Large sections of roof clearly missing (more than 50% of roof gone)
- Partial wall collapse visible on one or more sides
- Interior of building exposed and visible from above through the roof
- Heavy burn/charring across most of the footprint
- Building footprint is still recognizable — you can still see where it stood — but it is severely compromised

Evidence against major-damage:
- Roof is mostly intact → minor-damage or no-damage instead
- Building has completely vanished (only foundation/ash) → destroyed instead
- Only superficial discoloration → minor-damage instead

Score 10 if the building clearly shows major structural damage with most roof gone but footprint still visible. Score 0 if clearly undamaged, only superficially damaged, or completely gone."""

DESTROYED_SPECIALIST_PROMPT = """You are a damage detection specialist. Your ONLY job is to determine if this building is COMPLETELY DESTROYED — meaning NO structure remains.

STRICT SCORING RULES — read carefully before scoring:

Score 0–2 (definitely NOT destroyed) if ANY of these are true:
- Any roof material is still visible over the footprint (even partially)
- Walls are still standing on any side
- The building outline is still recognizable as a structure
- The footprint looks similar in shape to the BEFORE image
- You can see the building still exists as a building

Score 3–5 (probably not destroyed) if:
- There is heavy damage but remnants of structure remain
- You are uncertain — lean toward NOT destroyed

Score 7–9 (likely destroyed) ONLY if ALL of these are true:
- No roof material visible anywhere over the footprint
- No standing walls visible
- Footprint has been replaced by rubble, ash, or bare foundation slab
- Building shape from BEFORE is no longer recognizable as a structure

Score 10 (certainly destroyed) ONLY if:
- Only a bare concrete foundation slab remains, OR
- Complete ash or rubble field with no structure whatsoever

CRITICAL: If you can still see the building as a building — even heavily damaged — score 0–3.
A building with a partially collapsed roof is NOT destroyed. Score it 0–3.
When in doubt, score LOW. False positives (scoring high when not destroyed) are much worse than false negatives."""

SUPERVISOR_SCHEMA = {
    "type": "object",
    "properties": {
        "specialist_scores": {
            "type": "object",
            "description": "Summary of each specialist's match_score",
            "properties": {
                "no_damage": {"type": "integer"},
                "minor_damage": {"type": "integer"},
                "major_damage": {"type": "integer"},
                "destroyed": {"type": "integer"},
            },
        },
        "reasoning": {
            "type": "string",
            "description": "1-2 sentences explaining which specialists you agreed/disagreed with and why",
        },
        "damage_level": {
            "type": "string",
            "enum": ["no-damage", "minor-damage", "major-damage", "destroyed"],
            "description": "Final damage classification",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": ["specialist_scores", "reasoning", "damage_level", "confidence"],
}

_SUPERVISOR_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(SUPERVISOR_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)

SUPERVISOR_SYSTEM_PROMPT = """You are a senior damage assessment supervisor. You have access to the original pre/post satellite images AND reports from 4 specialist agents who each evaluated the same building for a specific damage class.

Your job is to visually inspect the images yourself AND review all 4 specialist reports to make the FINAL damage classification. Trust your own direct visual assessment — override specialist reports if the images clearly contradict them.

Decision rules:
1. Start from the LESS SEVERE classes and escalate only when evidence is clear.
2. Choose no-damage if no-damage score ≥ 7 AND destroyed score < 6.
3. Choose minor-damage if minor score ≥ 6 AND the building is clearly still standing.
4. Choose major-damage if major score ≥ 6 AND roof is clearly substantially missing.
5. Choose destroyed ONLY if destroyed score ≥ 8 AND major score < 6 — meaning the building has truly vanished, not just heavily damaged.
6. When destroyed and major-damage scores are close (within 2 points), ALWAYS prefer major-damage.
7. When in doubt between any two classes, pick the LESS SEVERE option.

Key principle: A damaged but standing building is NOT destroyed. Only choose destroyed if the building structure has completely disappeared.

Always respond with valid JSON matching the provided schema."""


def specialist_prompt(specialist_system: str) -> str:
    """Build the user-turn prompt for a class specialist."""
    return _IMAGE_LAYOUT_2 + _SPECIALIST_SCHEMA_SUFFIX


def supervisor_prompt(specialist_reports: dict[str, dict]) -> str:
    """Build the supervisor user-turn prompt from the 4 specialist reports."""
    reports_str = json.dumps(specialist_reports, indent=2)
    return (
        "The pre/post satellite images are provided above for your direct visual inspection.\n\n"
        "Here are the 4 specialist reports for this building:\n\n"
        f"```json\n{reports_str}\n```\n\n"
        "Visually inspect the images AND review the specialist reports, then make your final damage classification.\n\n"
        + _SUPERVISOR_SCHEMA_SUFFIX
    )


def parse_specialist_response(response_text: str) -> dict:
    """Parse a specialist agent response."""
    import re
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        data.setdefault("match_score", 0)
        data.setdefault("verdict", "no")
        data.setdefault("supporting_evidence", [])
        data.setdefault("counter_evidence", [])
        return data
    except (json.JSONDecodeError, ValueError):
        return {"match_score": 0, "verdict": "no", "supporting_evidence": [], "counter_evidence": [], "parse_error": response_text[:100]}


def parse_supervisor_response(response_text: str) -> dict:
    """Parse the supervisor agent response."""
    import re
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        valid = {"no-damage", "minor-damage", "major-damage", "destroyed"}
        if data.get("damage_level") not in valid:
            data["damage_level"] = "no-damage"
            data["parse_error"] = "invalid damage_level"
        data.setdefault("confidence", "low")
        data.setdefault("reasoning", "")
        data.setdefault("parse_error", "")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        return {"damage_level": "no-damage", "confidence": "low", "reasoning": "", "parse_error": str(e)}


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


# ---------------------------------------------------------------------------
# Stage 0 — binary damage detector (damaged vs intact)
# Runs before the full L1+L2 hierarchy. If "no damage", skip hierarchy entirely.
# ---------------------------------------------------------------------------

STAGE0_SCHEMA = {
    "type": "object",
    "properties": {
        "building_damaged": {
            "type": "string",
            "enum": ["yes", "no", "unclear"],
            "description": "Is there ANY visible structural or surface damage to the target building itself?",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "evidence": {
            "type": "string",
            "description": "1-2 sentence observation. What specifically did you compare between BEFORE and AFTER?",
        },
    },
    "required": ["building_damaged", "confidence", "evidence"],
}

_STAGE0_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(STAGE0_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)

STAGE0_SYSTEM_PROMPT = """You are a binary damage detection agent. Your ONLY job is to answer ONE question:

  "Does the target building show ANY visible change or damage between the BEFORE and AFTER images?"

STRICT RULES:
- Judge the TARGET BUILDING ONLY — the one outlined in red. Ignore ALL surrounding land, vegetation, and other structures.
- Burned grass, scorched earth, or ash fields AROUND the building are NOT building damage.
- Answer "no" if the building's roof shape, color, and footprint are geometrically unchanged between BEFORE and AFTER.
- Answer "yes" if you can see ANY of: missing roof material, collapsed walls, exposed interior, changed footprint shape, heavy charring on the building itself.
- Answer "unclear" ONLY if the crop is too small or blurry to make any comparison at all.
- Use "no" confidently if the building shape matches BEFORE — minor color differences from smoke/ash on the building exterior alone are NOT structural damage.

Key visual anchors:
- NO DAMAGE: building outline in AFTER matches BEFORE exactly. Roof still present as a solid shape.
- DAMAGED: visible holes in roof, collapsed sections, missing walls, or the building outline has changed.

When in doubt between "no" and "unclear": choose "no" if you can see the building at all and it looks similar.
When in doubt between "yes" and "unclear": choose "yes" to avoid missing real damage.

Always respond with valid JSON matching the provided schema."""


def stage0_prompt(scene_description: str = "") -> str:
    """User-turn prompt for Stage 0 binary damage detector."""
    scene_block = ""
    if scene_description:
        scene_block = (
            f"SCENE-SPECIFIC CONTEXT:\n{scene_description}\n\n"
            "Use this to understand what INTACT buildings look like in this specific scene.\n\n"
        )
    return (
        _IMAGE_LAYOUT_2
        + scene_block
        + "Compare the BEFORE and AFTER images of the outlined building.\n"
        "Answer: is there any visible damage or change on the building itself?\n\n"
        + _STAGE0_SCHEMA_SUFFIX
    )


def parse_stage0_response(response_text: str) -> dict:
    """Parse Stage 0 binary response."""
    import re
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if data.get("building_damaged") not in ("yes", "no", "unclear"):
            data["building_damaged"] = "unclear"
        data.setdefault("confidence", "low")
        data.setdefault("evidence", "")
        data.setdefault("parse_error", "")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        return {"building_damaged": "unclear", "confidence": "low", "evidence": "", "parse_error": str(e)}


# ---------------------------------------------------------------------------
# Level-2 class specialists (receive L1 evidence report + visual crops)
# ---------------------------------------------------------------------------

L2_NO_DAMAGE_SPECIALIST_PROMPT = """You are a Level-2 damage specialist. Your ONLY job: determine if this building has NO structural damage.

You receive pre/post satellite image crops AND a structured Level-1 Scene Worker report with explicit damage indicator findings.

Evidence for NO DAMAGE:
- structure_still_standing: yes
- roof_over_half_missing: no
- wall_collapse_visible: no
- footprint_replaced_by_rubble: no
- surface_damage_only: yes OR no visible surface damage
- Visually: building matches pre-disaster appearance

Score 10 if the L1 report shows all indicators negative AND the images confirm no visible change.
Score 0 if any structural damage indicator is "yes"."""

L2_MINOR_DAMAGE_SPECIALIST_PROMPT = """You are a Level-2 damage specialist. Your ONLY job: determine if this building has MINOR structural damage.

You receive pre/post satellite image crops AND a structured Level-1 Scene Worker report with explicit damage indicator findings.

Evidence for MINOR DAMAGE:
- structure_still_standing: yes (must be standing)
- surface_damage_only: yes (only surface-level change)
- roof_over_half_missing: no (roof mostly intact — less than 50% affected)
- wall_collapse_visible: no (walls intact)
- footprint_replaced_by_rubble: no (footprint recognizable)
- Visually: scorch marks, small dark patches, minor debris — but building structure preserved

Score 10 if standing building with only surface damage indicated.
Score 0 if no damage at all (no-damage), OR if major structural loss is present (>50% roof gone or wall collapse)."""

L2_MAJOR_DAMAGE_SPECIALIST_PROMPT = """You are a Level-2 damage specialist. Your ONLY job: determine if this building has MAJOR structural damage — severe damage where the building footprint is still recognizable but substantially compromised.

You receive pre/post satellite image crops AND a structured Level-1 Scene Worker report with explicit damage indicator findings.

THE CRITICAL MAJOR-DAMAGE SIGNATURE:
  STRONG evidence for major-damage (score 8-10) when:
    roof_over_half_missing: yes    ← roof is substantially gone
    structure_still_standing: yes  ← building still stands as a recognizable structure
    footprint_replaced_by_rubble: no  ← footprint NOT completely erased

  This is the key pattern that distinguishes major-damage from destroyed:
  The building is severely damaged but still exists as a recognizable structure.

  WEAK or NO evidence (score 0-3) when:
    roof_over_half_missing: no → probably minor or no-damage
    footprint_replaced_by_rubble: yes → probably destroyed
    structure_still_standing: no → probably destroyed

Visually: Look for large holes/gaps in the roof, exposed building interior, partial wall collapse — but the building footprint is still identifiable from BEFORE.

Score 10 if roof >50% gone AND structure still standing as a recognizable building.
Score 0 if building is pristine/lightly damaged, OR if completely gone to rubble."""

L2_DESTROYED_SPECIALIST_PROMPT = """You are a Level-2 damage specialist. Your ONLY job: determine if this building is COMPLETELY DESTROYED — no structure remains at all.

You receive pre/post satellite image crops AND a structured Level-1 Scene Worker report with explicit damage indicator findings.

DESTROYED SIGNATURE — ALL of these must be true for a high score:
  footprint_replaced_by_rubble: yes  ← footprint erased (rubble/ash/bare foundation slab)
  structure_still_standing: no       ← NO structure remaining
  roof_over_half_missing: yes        ← all roof gone
  wall_collapse_visible: yes         ← walls fully collapsed

SCORING RULES:
  Score 9-10: ALL four indicators above are "yes" AND images confirm complete loss
  Score 6-8: Most indicators point to destruction but one is "unclear"
  Score 0-3: structure_still_standing=yes OR footprint_replaced_by_rubble=no

CRITICAL: structure_still_standing=yes means NOT destroyed — score 0-3 regardless of other damage.
A building with a partially collapsed roof is NOT destroyed.

Visually: Look for bare concrete slab (gray rectangle with no structure), or complete ash/rubble field where the building stood in BEFORE."""

_L2_SPECIALIST_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(SPECIALIST_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)

L2_SUPERVISOR_SCHEMA = {
    "type": "object",
    "properties": {
        "l1_assessment": {
            "type": "string",
            "description": "The L1 worker's initial_assessment value",
        },
        "l2_scores": {
            "type": "object",
            "description": "L2 specialist match_scores",
            "properties": {
                "no_damage":    {"type": "integer"},
                "minor_damage": {"type": "integer"},
                "major_damage": {"type": "integer"},
                "destroyed":    {"type": "integer"},
            },
        },
        "key_decision": {
            "type": "string",
            "description": "1-2 sentences explaining how you reconciled L1 indicators, L2 scores, and visual evidence",
        },
        "damage_level": {
            "type": "string",
            "enum": ["no-damage", "minor-damage", "major-damage", "destroyed"],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": ["l1_assessment", "l2_scores", "key_decision", "damage_level", "confidence"],
}

_L2_SUPERVISOR_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(L2_SUPERVISOR_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)

L2_SUPERVISOR_SYSTEM_PROMPT = """You are a senior damage assessment supervisor at Level 2 of the hierarchical system.

You have access to:
1. Pre/post satellite image crops of the target building (your own visual inspection)
2. A structured Level-1 Scene Worker report (explicit damage indicator checklist)
3. Four Level-2 specialist reports (one per damage class)

DECISION PROCESS — use this order:

Step 1 — Apply the L1 indicator decision tree (strict — requires explicit "yes", not "unclear"):
  DESTROYED path:   footprint_replaced_by_rubble=yes AND structure_still_standing=no AND roof_over_half_missing=yes
    → All three must be explicit "yes" (not "unclear") → destroyed
    → If ANY of the three is "unclear": fall through to Step 2 (use L2 scores)

  MAJOR path:       roof_over_half_missing=yes AND structure_still_standing=yes (or unclear)
    → If roof is clearly >50% gone but building still has recognizable structure: major-damage
    → structure_still_standing=unclear → lean toward major, not destroyed

  MINOR path:       surface_damage_only=yes AND structure_still_standing=yes AND roof_over_half_missing=no
    → Only surface changes, building intact: minor-damage

  NO-DAMAGE path:   all indicators are "no" or "unclear" AND no change visible → no-damage

Step 2 — When L1 path is unclear, use L2 specialist scores. Highest score among the 4 classes wins.

Step 3 — Visually verify: if images clearly contradict your decision, override and explain.

ANTI-BIAS RULES (prevent destroyed over-prediction — the #1 accuracy killer):
  - structure_still_standing=unclear → NEVER choose destroyed; prefer major-damage instead
  - footprint_replaced_by_rubble=unclear → NEVER choose destroyed; prefer major-damage instead
  - If destroyed_score and major_score are within 3 points: prefer major-damage
  - major_score ≥ 3 AND (standing=unclear OR rubble=unclear): choose major-damage, not destroyed
  - Never default to no-damage if ANY indicator is explicitly "yes"

Always respond with valid JSON matching the provided schema."""


def l2_specialist_user_prompt(worker_report: dict, scene_description: str = "", cnn_probs: dict | None = None) -> str:
    """User-turn for an L2 specialist: embeds the L1 evidence report + image instructions + schema."""
    ind = worker_report.get("damage_indicators", {})
    changes = worker_report.get("change_observations", [])
    scene_block = ""
    if scene_description:
        scene_block = (
            f"SCENE-SPECIFIC CONTEXT:\n{scene_description}\n\n"
            "Use this context to correctly interpret what buildings in this scene look like when undamaged vs damaged.\n\n"
        )
    report_block = (
        "LEVEL-1 SCENE WORKER REPORT:\n"
        f"  Initial assessment: {worker_report.get('initial_assessment', 'unknown')}\n"
        f"  Evidence summary: {worker_report.get('evidence_summary', '')}\n"
        f"  Change observations: {'; '.join(changes)}\n"
        "  Damage indicators:\n"
        f"    roof_over_half_missing:      {ind.get('roof_over_half_missing', 'unclear')}\n"
        f"    wall_collapse_visible:       {ind.get('wall_collapse_visible', 'unclear')}\n"
        f"    structure_still_standing:    {ind.get('structure_still_standing', 'unclear')}\n"
        f"    footprint_replaced_by_rubble:{ind.get('footprint_replaced_by_rubble', 'unclear')}\n"
        f"    surface_damage_only:         {ind.get('surface_damage_only', 'unclear')}\n\n"
        "Pre/post disaster images of the target building are provided above.\n"
        "Use BOTH the visual evidence AND the Level-1 report to score your class.\n\n"
    )
    cnn_block = _cnn_block(cnn_probs) if cnn_probs else ""
    return _IMAGE_LAYOUT_2 + scene_block + cnn_block + report_block + _L2_SPECIALIST_SCHEMA_SUFFIX


def l2_supervisor_user_prompt(worker_report: dict, specialist_reports: dict, scene_description: str = "", cnn_probs: dict | None = None) -> str:
    """User-turn for L2 supervisor: L1 report + L2 specialist verdicts + image instructions + schema."""
    ind = worker_report.get("damage_indicators", {})
    l1_block = (
        "LEVEL-1 SCENE WORKER REPORT:\n"
        f"  Initial assessment: {worker_report.get('initial_assessment', 'unknown')}\n"
        f"  Evidence summary: {worker_report.get('evidence_summary', '')}\n"
        "  Damage indicators:\n"
        f"    roof_over_half_missing:      {ind.get('roof_over_half_missing', 'unclear')}\n"
        f"    wall_collapse_visible:       {ind.get('wall_collapse_visible', 'unclear')}\n"
        f"    structure_still_standing:    {ind.get('structure_still_standing', 'unclear')}\n"
        f"    footprint_replaced_by_rubble:{ind.get('footprint_replaced_by_rubble', 'unclear')}\n"
        f"    surface_damage_only:         {ind.get('surface_damage_only', 'unclear')}\n\n"
    )
    l2_block = (
        "LEVEL-2 SPECIALIST REPORTS:\n"
        f"```json\n{json.dumps(specialist_reports, indent=2)}\n```\n\n"
    )
    scene_block = ""
    if scene_description:
        scene_block = (
            f"SCENE-SPECIFIC CONTEXT (auto-generated from pre-disaster analysis):\n{scene_description}\n\n"
            "Use this context to interpret whether surface changes in the images are typical for this scene or indicate real damage.\n\n"
        )
    cnn_block = _cnn_block(cnn_probs) if cnn_probs else ""
    return (
        "Pre/post satellite images are provided above for your direct visual inspection.\n\n"
        + scene_block
        + cnn_block
        + l1_block
        + l2_block
        + "Make your final classification using the L1 indicators, L2 scores, scene context, and your visual inspection.\n\n"
        + _L2_SUPERVISOR_SCHEMA_SUFFIX
    )


def parse_l2_supervisor_response(response_text: str) -> dict:
    """Parse the L2 supervisor JSON response."""
    import re
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        valid = {"no-damage", "minor-damage", "major-damage", "destroyed"}
        if data.get("damage_level") not in valid:
            data["damage_level"] = "no-damage"
            data["parse_error"] = "invalid damage_level"
        data.setdefault("confidence", "low")
        data.setdefault("key_decision", "")
        data.setdefault("l1_assessment", "")
        data.setdefault("l2_scores", {})
        data.setdefault("parse_error", "")
        # Alias key_decision → key_evidence for CSV compatibility
        data["key_evidence"] = data["key_decision"]
        return data
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "damage_level": "no-damage", "confidence": "low",
            "key_decision": "", "key_evidence": "",
            "l1_assessment": "", "l2_scores": {}, "parse_error": str(e),
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
