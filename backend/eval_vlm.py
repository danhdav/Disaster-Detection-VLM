"""
VLM evaluation using OpenRouter — compares our updated prompt against ground truth.

Usage:
    uv run python eval_vlm.py                                      # all 191 buildings
    uv run python eval_vlm.py --limit 20                           # quick sample
    uv run python eval_vlm.py --limit 40 --balanced                # ~10 per class
    uv run python eval_vlm.py --model claude-sonnet-4-6 --cnn_probs  # with CNN priors
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests
from PIL import Image

from disaster_bench.models.damage.vlm_prompts import (
    SYSTEM_PROMPT, get_system_prompt, ungrounded_prompt, parse_vlm_response,
    NO_DAMAGE_SPECIALIST_PROMPT, MINOR_DAMAGE_SPECIALIST_PROMPT,
    MAJOR_DAMAGE_SPECIALIST_PROMPT, DESTROYED_SPECIALIST_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT, specialist_prompt, supervisor_prompt,
    parse_specialist_response, parse_supervisor_response,
    BOUNDARY_V1_SYSTEM_PROMPT, boundary_v1_prompt, parse_boundary_v1_response,
    L2_NO_DAMAGE_SPECIALIST_PROMPT, L2_MINOR_DAMAGE_SPECIALIST_PROMPT,
    L2_MAJOR_DAMAGE_SPECIALIST_PROMPT, L2_DESTROYED_SPECIALIST_PROMPT,
    L2_SUPERVISOR_SYSTEM_PROMPT, l2_specialist_user_prompt, l2_supervisor_user_prompt,
    parse_l2_supervisor_response,
    STAGE0_SYSTEM_PROMPT, stage0_prompt, parse_stage0_response,
)

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
HERE = Path(__file__).parent
EVAL_CSV      = HERE / "data" / "vlm_eval_sample.csv"
EVAL_CSV_FULL = HERE / "data" / "vlm_eval_full.csv"   # includes masked paths + WKT
REPORTS_DIR   = HERE / "reports" / "vlm"
SCENE_OVERRIDES_PATH = HERE / "data" / "scene_prompt_overrides.json"


def load_scene_overrides() -> dict[str, str]:
    """Load manually-corrected scene prompts from scene_prompt_overrides.json."""
    if SCENE_OVERRIDES_PATH.exists():
        with open(SCENE_OVERRIDES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} scene prompt overrides from {SCENE_OVERRIDES_PATH.name}")
        return data
    return {}
CNN_PROBS_CSV = Path(r"D:\Aaron\UTD\Spring 26\Capstone Project\Benchmark-Model-xView2\reports\fusion\cnn_probs_ppd5_tta4.csv")

PROMPT_GENERATOR_SYSTEM_PROMPT = """You are an expert remote sensing analyst and prompt engineer.
You will receive a full pre-disaster satellite tile of a wildfire-affected area.

Your task: write a COMPLETE, STANDALONE damage classification system prompt tailored specifically to THIS scene.
The prompt you write will be used as the system prompt for a VLM that classifies individual buildings in this scene.

The prompt you write MUST include ALL of the following sections:

1. SCENE IDENTITY — 2-3 sentences describing what you see: terrain type, neighborhood density, building style, typical roof material and exact color, wall material if visible.

2. INTACT BUILDING ANCHOR — describe exactly what an undamaged building looks like in THIS scene after a wildfire. Be specific to what you see (e.g. "an intact building retains its dark gray rectangular shingle roof matching the BEFORE footprint exactly; walls are tan stucco, unchanged").

3. DAMAGE LEVEL VISUAL DEFINITIONS for this scene:
   - no-damage: what it looks like here specifically
   - minor-damage: what subtle fire damage looks like on THESE building types
   - major-damage: what significant structural damage looks like on THESE building types
   - destroyed: what total loss looks like here (e.g. bare concrete slab, ash field)

4. FALSE CUES WARNING — list 2-3 things that look like damage but are NOT (e.g. burned chaparral around intact buildings, ash-coated tile roofs, shadow changes).

5. CLASSIFICATION GATES — copy these exactly:
   DESTROYED GATE — label destroyed only if AT LEAST 3 are clearly visible:
     1. Footprint is rubble, ash, or bare foundation
     2. Roof completely gone
     3. Walls largely collapsed
     4. Building appears missing vs BEFORE
   → If gate PASSES: destroyed. If gate FAILS but significant damage visible: major-damage.

   MAJOR-DAMAGE GATE — choose major-damage if AT LEAST 2 are clearly visible:
     1. More than half the roof area missing or collapsed
     2. Partial wall collapse on one or more sides
     3. Large exposed interior visible through roofline
     4. Heavy charring across most of footprint
   → If gate PASSES: major-damage.

   MINOR-DAMAGE — choose minor-damage if ANY of:
     - Scorch marks or discoloration vs BEFORE
     - Small roof damage (<50% area)
     - Debris near building not present in BEFORE

   NO DAMAGE — only if completely unchanged from BEFORE.

6. TIEBREAKER RULES:
   - When uncertain between destroyed and major-damage → choose major-damage
   - When uncertain between no-damage and minor-damage → choose minor-damage
   - When uncertain between major-damage and minor-damage → choose major-damage only if the MAJOR gate clearly passes

End with: "Always respond with valid JSON matching the provided schema. Think step by step."

Write ONLY the system prompt text — no preamble, no explanation, no meta-commentary."""

SCENE_WORKER_SYSTEM_PROMPT = """You are a Scene Worker agent in a hierarchical damage assessment system.

You receive THREE inputs:
  1. The full pre-disaster satellite tile (wide scene view — use this to understand the scene context)
  2. A pre-disaster crop of the specific target building (outlined in red)
  3. A post-disaster crop of the same target building (outlined in red)

Your job is to produce a structured evidence report that specialist agents at the next level will use to make the final classification. You are NOT making the final call — you are gathering and organizing evidence.

STEP 1 — Scene context (from the full tile):
  Describe the scene: terrain type, typical roof material/color, building density, any known fire behavior patterns for this area.

STEP 2 — Building pre-disaster description:
  Describe the target building as it appears before the disaster: roof shape, color, approximate size, wall material if visible.

STEP 3 — Post-disaster change observations:
  Compare BEFORE and AFTER crops carefully. List every visible change on the building structure itself.

STEP 4 — Damage indicator checklist:
  Answer each question strictly based on visual evidence. Use "unclear" when the evidence is genuinely ambiguous.

  - roof_over_half_missing: Is more than half the roof area gone, burned through, or collapsed?
  - wall_collapse_visible: Are any walls visibly collapsed or breached (not just scorched)?
  - structure_still_standing: Does the building still appear as a standing structure?
    → "yes" if ANY structural outline is still recognizable (even heavily damaged)
    → "no" ONLY if the building has completely disappeared — only bare slab, ash, or rubble with NO building shape
    → "unclear" if the crop is ambiguous
  - footprint_replaced_by_rubble: Is the footprint now rubble, ash, or bare foundation slab?
    → "yes" ONLY if the ENTIRE footprint is clearly a rubble/ash/slab field with zero building structure
    → "unclear" if partial structure remains
  - surface_damage_only: Is the damage limited to surface discoloration or small patches?

STEP 5 — Initial assessment:
  Based on your observations, what is your best initial damage classification?
  Also write a 1-2 sentence evidence_summary for the specialist agents.

Always respond with valid JSON matching the provided schema. Be precise and evidence-based."""

SCENE_WORKER_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_context": {
            "type": "string",
            "description": "Terrain, roof types, building style, fire behavior context for this scene",
        },
        "building_pre": {
            "type": "string",
            "description": "Description of the target building in the pre-disaster image",
        },
        "building_post": {
            "type": "string",
            "description": "Description of the target building in the post-disaster image",
        },
        "change_observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific visible changes on the building between BEFORE and AFTER",
        },
        "damage_indicators": {
            "type": "object",
            "properties": {
                "roof_over_half_missing":       {"type": "string", "enum": ["yes", "no", "unclear"]},
                "wall_collapse_visible":         {"type": "string", "enum": ["yes", "no", "unclear"]},
                "structure_still_standing":      {"type": "string", "enum": ["yes", "no", "unclear"]},
                "footprint_replaced_by_rubble":  {"type": "string", "enum": ["yes", "no", "unclear"]},
                "surface_damage_only":           {"type": "string", "enum": ["yes", "no", "unclear"]},
            },
            "required": ["roof_over_half_missing", "wall_collapse_visible",
                         "structure_still_standing", "footprint_replaced_by_rubble", "surface_damage_only"],
        },
        "initial_assessment": {
            "type": "string",
            "enum": ["no-damage", "minor-damage", "major-damage", "destroyed"],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "evidence_summary": {
            "type": "string",
            "description": "1-2 sentence summary of key findings for specialist review",
        },
    },
    "required": ["scene_context", "building_pre", "building_post", "change_observations",
                 "damage_indicators", "initial_assessment", "confidence", "evidence_summary"],
}

_WORKER_SCHEMA_SUFFIX = (
    "Respond with JSON matching this schema:\n"
    f"{json.dumps(SCENE_WORKER_SCHEMA, indent=2)}\n\n"
    "Return only valid JSON, no other text."
)


SCENE_SCOUT_SYSTEM_PROMPT = """You are a satellite imagery scene characterization agent.
You will receive 1-3 pre-disaster satellite image crops of buildings from the same geographic scene.

Describe the scene in TWO parts so a damage classifier can correctly identify ALL four damage levels:

PART 1 — Intact building appearance (what no-damage looks like):
- Roof type and exact color (e.g. "dark gray composition shingle", "terracotta clay tile", "flat white membrane")
- Building style and size (e.g. "1-story single-family homes", "2-story hillside residences")
- Wall material/color if visible (e.g. "tan stucco", "white wood siding")
- Visual anchor: what an INTACT building looks like in AFTER (e.g. "intact roof = dark rectangle matching BEFORE footprint exactly")

PART 2 — Damage appearance (what each damage level looks like in THIS scene):
- minor-damage: what subtle damage looks like here (e.g. "scorch marks on shingle, small dark patches on roof, otherwise intact shape")
- major-damage: what significant damage looks like here (e.g. "large bright gaps in roof, partial wall collapse, interior visible but footprint recognizable")
- destroyed: what total loss looks like here (e.g. "bare concrete foundation slab — a light gray rectangle with NO roof material; or ash/rubble field")
- Terrain false cue warning: what surroundings change looks like that is NOT building damage (e.g. "burned chaparral = gray ash field AROUND the building, not inside footprint")

5-8 sentences total. Plain text only — no JSON, no bullet points."""


def load_cnn_probs() -> dict[str, dict]:
    """Load CNN softmax probabilities keyed by uid."""
    probs: dict[str, dict] = {}
    with open(CNN_PROBS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            probs[row["uid"]] = {
                "p_nodmg": float(row["p_nodmg"]),
                "p_minor": float(row["p_minor"]),
                "p_major": float(row["p_major"]),
                "p_dest":  float(row["p_dest"]),
            }
    return probs


def _is_claude(model: str) -> bool:
    return model.startswith("claude-") or model.startswith("anthropic/claude")


VLM_CROP_SIZE = 192  # Match CNN preprocessing: bilinear upscale to 192×192


def img_to_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    # Upscale small crops to 192×192 (same target as CNN) so the VLM sees
    # meaningful spatial detail rather than tiny raw pixels.
    if img.width < VLM_CROP_SIZE or img.height < VLM_CROP_SIZE:
        img = img.resize((VLM_CROP_SIZE, VLM_CROP_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_diff_b64(row: dict, amplify: int = 6) -> str | None:
    """
    Amplified pixel-difference heatmap for a building.
    Uses pre_masked/post_masked paths when available (polygon-masked, background black)
    so the diff only shows change inside the building footprint.
    Falls back to bbox crops if masked paths are missing.
    """
    try:
        import numpy as np
        pre_path  = Path(row.get("pre_masked_path")  or row["pre_path"])
        post_path = Path(row.get("post_masked_path") or row["post_path"])
        if not pre_path.exists():
            pre_path = Path(row["pre_path"])
        if not post_path.exists():
            post_path = Path(row["post_path"])

        pre  = np.array(Image.open(pre_path).convert("RGB")).astype(np.int32)
        post = np.array(Image.open(post_path).convert("RGB")).astype(np.int32)
        if pre.shape != post.shape:
            return None
        diff = np.clip(np.abs(post - pre) * amplify, 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff)
        if diff_img.width < VLM_CROP_SIZE or diff_img.height < VLM_CROP_SIZE:
            diff_img = diff_img.resize((VLM_CROP_SIZE, VLM_CROP_SIZE), Image.BILINEAR)
        buf = io.BytesIO()
        diff_img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _fetch_tile_b64(tile_id: str) -> str | None:
    """Download the full pre-disaster tile from S3 and return as base64 data URL. Returns None on failure."""
    try:
        import boto3, io
        from botocore.config import Config as BotoConfig
        bucket = os.getenv("S3_BUCKET_NAME", "")
        if not bucket:
            return None
        client = boto3.client("s3", config=BotoConfig(region_name="us-east-2", signature_version="v4"))
        key = f"xview2-test-data/images/{tile_id}_pre_disaster.png"
        obj = client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"  [tile fetch failed for {tile_id}]: {e}")
        return None


def generate_scene_prompt(tile_id: str, model: str) -> str | None:
    """Generate a complete custom system prompt for a specific scene using the full S3 tile.
    Returns the prompt string, or None if the tile cannot be fetched."""
    import anthropic
    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    tile_b64 = _fetch_tile_b64(tile_id)
    if not tile_b64:
        return None

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    content = [
        _img_block(tile_b64),
        {"type": "text", "text": (
            f"This is the full pre-disaster satellite tile for scene: {tile_id}\n\n"
            "Write a complete custom damage classification system prompt tailored to THIS specific scene. "
            "Follow all instructions in your system prompt exactly."
        )},
    ]

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=1200, temperature=0,
        system=PROMPT_GENERATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return (resp.content[0].text if resp.content else "").strip() or None


def call_scene_worker(pre_b64: str, post_b64: str, model: str, tile_id: str, scene_description: str = "") -> tuple[dict, float, int]:
    """Level 1 — Scene Worker agent.
    Sends full pre-disaster tile (from S3) + building crops to produce a structured evidence report.
    Returns (report_dict, latency_ms, tokens)."""
    import anthropic
    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    content = []

    # Full pre-disaster tile for scene context
    tile_b64 = _fetch_tile_b64(tile_id)
    if tile_b64:
        content += [
            _img_block(tile_b64),
            {"type": "text", "text": "Image 1: Full pre-disaster satellite tile (scene context). Use this to understand the scene."},
        ]

    # Building crops
    content += [
        _img_block(pre_b64), {"type": "text", "text": "Image 2: Pre-disaster crop — target building outlined in red."},
        _img_block(post_b64), {"type": "text", "text": "Image 3: Post-disaster crop — same target building outlined in red."},
    ]
    if scene_description:
        content.append({"type": "text", "text": (
            f"SCENE-SPECIFIC CONTEXT for {tile_id}:\n{scene_description}\n\n"
            "Use this context to correctly interpret what intact vs damaged buildings look like in this scene."
        )})
    content.append({"type": "text", "text": _WORKER_SCHEMA_SUFFIX})

    t0 = time.perf_counter()
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=1024, temperature=0,
        system=SCENE_WORKER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
    raw = resp.content[0].text if resp.content else ""

    # Parse JSON
    import re
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        report = json.loads(text)
    except Exception:
        report = {
            "scene_context": "", "building_pre": "", "building_post": "",
            "change_observations": [], "initial_assessment": "no-damage",
            "confidence": "low", "evidence_summary": raw[:200],
            "damage_indicators": {
                "roof_over_half_missing": "unclear", "wall_collapse_visible": "unclear",
                "structure_still_standing": "unclear", "footprint_replaced_by_rubble": "unclear",
                "surface_damage_only": "unclear",
            },
            "parse_error": raw[:200],
        }
    return report, latency_ms, tokens


def call_scene_scout(pre_images_b64: list[str], model: str, tile_id: str) -> str:
    """Stage 1 of scene-scout architecture.
    Prefers the full 1024×1024 pre-disaster tile from S3; falls back to building crops.
    Returns a plain-text scene description injected into the Stage 2 system prompt."""
    import anthropic
    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    # Try full tile from S3 first
    tile_b64 = _fetch_tile_b64(tile_id)
    if tile_b64:
        content = [
            _img_block(tile_b64),
            {"type": "text", "text": f"Full pre-disaster satellite tile for scene {tile_id} (1024×1024). Describe the building characteristics. 5-8 sentences, plain text only."},
        ]
    else:
        # Fallback: use building crops
        content = []
        for i, img_b64 in enumerate(pre_images_b64[:3], 1):
            content.append(_img_block(img_b64))
            content.append({"type": "text", "text": f"Pre-disaster building crop {i} from scene {tile_id}."})
        content.append({"type": "text", "text": "Describe the typical building characteristics visible. 3-5 sentences, plain text only."})

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=384, temperature=0,
        system=SCENE_SCOUT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    source = "tile" if tile_b64 else "crops"
    result = (resp.content[0].text if resp.content else "").strip()
    return f"[{source}] {result}"


def call_stage0(pre_b64: str, post_b64: str, model: str, tile_id: str = "", scene_description: str = "") -> tuple[str, str, float, int]:
    """Stage 0 — binary 'damaged vs intact?' detector.
    Returns (building_damaged: 'yes'|'no'|'unclear', confidence, latency_ms, tokens)."""
    import anthropic
    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    content = [
        _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
        _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
        {"type": "text", "text": stage0_prompt(scene_description)},
    ]

    t0 = time.perf_counter()
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=256, temperature=0,
        system=STAGE0_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    raw = resp.content[0].text if resp.content else ""
    tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
    parsed = parse_stage0_response(raw)
    return parsed.get("building_damaged", "unclear"), parsed.get("confidence", "low"), latency_ms, tokens


def call_l2_hierarchy(pre_b64: str, post_b64: str, model: str, tile_id: str = "", scene_description: str = "", use_stage0: bool = False, cnn_probs: dict | None = None) -> tuple[str, float, int]:
    """L1 + L2 divide-and-conquer hierarchy:
      L1 = Scene Worker (full S3 tile + crops) → structured evidence report with damage_indicators
      L2 = 4 class specialists (each sees L1 report + crops) run in parallel → verdicts
      L2 supervisor = L1 report + L2 verdicts + visual inspection → final label

    6 API calls total (1 L1 + 4 L2 specialists in parallel + 1 supervisor).
    Returns (raw_supervisor_json, latency_ms, total_tokens)."""
    from concurrent.futures import ThreadPoolExecutor
    import anthropic

    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    t0 = time.perf_counter()
    total_tokens = 0

    # ── Stage 0: binary damage detector (short-circuit for no-damage) ────────
    if use_stage0:
        damaged, s0_conf, s0_lat, s0_tok = call_stage0(pre_b64, post_b64, model, tile_id, scene_description)
        total_tokens += s0_tok
        print(f"  S0: damaged={damaged} conf={s0_conf}")
        if damaged == "no" and s0_conf == "high":
            latency_ms = (time.perf_counter() - t0) * 1000
            result = json.dumps({"damage_level": "no-damage", "confidence": s0_conf,
                                 "key_decision": f"Stage 0 detected no damage (conf={s0_conf})",
                                 "l1_assessment": "skipped", "l2_scores": {}})
            return result, latency_ms, total_tokens

    # ── L1: Scene Worker ─────────────────────────────────────────────────────
    worker_report, _l1_lat, l1_tokens = call_scene_worker(pre_b64, post_b64, model, tile_id, scene_description=scene_description)
    total_tokens += l1_tokens
    ind = worker_report.get("damage_indicators", {})
    print(f"  L1: {worker_report.get('initial_assessment')} | "
          f"roof_miss={ind.get('roof_over_half_missing')} "
          f"wall_col={ind.get('wall_collapse_visible')} "
          f"standing={ind.get('structure_still_standing')} "
          f"rubble={ind.get('footprint_replaced_by_rubble')} "
          f"surface={ind.get('surface_damage_only')}")

    # ── L2: 4 class specialists in parallel ──────────────────────────────────
    l2_specs = [
        ("no_damage",    L2_NO_DAMAGE_SPECIALIST_PROMPT),
        ("minor_damage", L2_MINOR_DAMAGE_SPECIALIST_PROMPT),
        ("major_damage", L2_MAJOR_DAMAGE_SPECIALIST_PROMPT),
        ("destroyed",    L2_DESTROYED_SPECIALIST_PROMPT),
    ]
    user_prompt_text = l2_specialist_user_prompt(worker_report, scene_description=scene_description, cnn_probs=cnn_probs)

    def _call_l2_spec(args: tuple[str, str]) -> tuple[str, dict, int]:
        name, sys_prompt = args
        _client = anthropic.Anthropic(api_key=api_key)
        content = [
            _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
            _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
            {"type": "text", "text": user_prompt_text},
        ]
        resp = _client.messages.create(
            model=model, max_tokens=256, temperature=0,
            system=sys_prompt,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text if resp.content else ""
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        return name, parse_specialist_response(raw), tokens

    specialist_reports: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, report, tok in executor.map(_call_l2_spec, l2_specs):
            specialist_reports[name] = report
            total_tokens += tok

    scores = {n: r.get("match_score", "?") for n, r in specialist_reports.items()}
    verdicts = {n: r.get("verdict", "?") for n, r in specialist_reports.items()}
    print(f"  L2: scores={scores} verdicts={verdicts}")

    # ── L2 Supervisor ─────────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)
    sup_text = l2_supervisor_user_prompt(worker_report, specialist_reports, scene_description=scene_description, cnn_probs=cnn_probs)
    sup_content = [
        _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
        _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
        {"type": "text", "text": sup_text},
    ]
    resp = client.messages.create(
        model=model, max_tokens=384, temperature=0,
        system=L2_SUPERVISOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": sup_content}],
    )
    raw_sup = resp.content[0].text if resp.content else ""
    total_tokens += (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
    latency_ms = (time.perf_counter() - t0) * 1000
    return raw_sup, latency_ms, total_tokens


def call_vlm(pre_b64: str, post_b64: str, model: str, cnn_probs: dict | None = None, tile_id: str = "", diff_b64: str | None = None, scene_description: str = "") -> tuple[str, float, int]:
    if _is_claude(model):
        return call_anthropic(pre_b64, post_b64, model, cnn_probs=cnn_probs, tile_id=tile_id, diff_b64=diff_b64, scene_description=scene_description)
    return call_openrouter(pre_b64, post_b64, model, cnn_probs=cnn_probs, tile_id=tile_id, diff_b64=diff_b64)


def call_anthropic(pre_b64: str, post_b64: str, model: str, cnn_probs: dict | None = None, tile_id: str = "", diff_b64: str | None = None, scene_description: str = "") -> tuple[str, float, int]:
    import anthropic
    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("claude_api_key not set in .env")

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    # Use custom scene prompt if provided, otherwise fall back to generic scene prompt + optional description
    if scene_description.startswith("__CUSTOM__:"):
        system = scene_description[len("__CUSTOM__:"):]
    else:
        system = get_system_prompt(tile_id)
        if scene_description:
            system += f"\n\nSCENE-SPECIFIC CONTEXT (auto-generated from pre-disaster analysis of this scene):\n{scene_description}"
    prompt = ungrounded_prompt(full_tile=False, cnn_probs=cnn_probs, diff_overlay=bool(diff_b64))
    content = [
        _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
        _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
    ]
    if diff_b64:
        content += [_img_block(diff_b64), {"type": "text", "text": "Change heatmap above (brighter = more change)."}]
    content.append({"type": "text", "text": prompt})

    t0 = time.perf_counter()
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=512, temperature=0,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    raw = resp.content[0].text if resp.content else ""
    tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
    return str(raw), latency_ms, tokens


def call_openrouter(pre_b64: str, post_b64: str, model: str, cnn_probs: dict | None = None, tile_id: str = "", diff_b64: str | None = None) -> tuple[str, float, int]:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    prompt = ungrounded_prompt(full_tile=False, cnn_probs=cnn_probs, diff_overlay=bool(diff_b64))
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Pre-disaster image:"},
        {"type": "image_url", "image_url": {"url": pre_b64}},
        {"type": "text", "text": "Post-disaster image:"},
        {"type": "image_url", "image_url": {"url": post_b64}},
    ]
    if diff_b64:
        content += [
            {"type": "text", "text": "Change heatmap (brighter = more change):"},
            {"type": "image_url", "image_url": {"url": diff_b64}},
        ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": get_system_prompt(tile_id)},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    t0 = time.perf_counter()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000

    body = resp.json()
    raw = body["choices"][0]["message"]["content"]
    if isinstance(raw, list):
        raw = "\n".join(p.get("text", "") for p in raw if isinstance(p, dict) and p.get("type") == "text")
    tokens = body.get("usage", {}).get("total_tokens", 0)
    return str(raw), latency_ms, tokens


def call_multi_agent(pre_b64: str, post_b64: str, model: str, tile_id: str = "") -> tuple[str, float, int]:
    """Run 4 specialist agents in parallel then a supervisor agent. Returns (raw_supervisor_json, latency_ms, total_tokens)."""
    from concurrent.futures import ThreadPoolExecutor
    import anthropic
    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("claude_api_key not set in .env")

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    specialists = [
        ("no_damage",    NO_DAMAGE_SPECIALIST_PROMPT),
        ("minor_damage", MINOR_DAMAGE_SPECIALIST_PROMPT),
        ("major_damage", MAJOR_DAMAGE_SPECIALIST_PROMPT),
        ("destroyed",    DESTROYED_SPECIALIST_PROMPT),
    ]

    def _call_specialist(args: tuple[str, str]) -> tuple[str, dict, int]:
        name, sys_prompt = args
        # Each thread gets its own client instance (anthropic client is not thread-safe to share)
        _client = anthropic.Anthropic(api_key=api_key)
        content = [
            _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
            _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
            {"type": "text", "text": specialist_prompt(sys_prompt)},
        ]
        resp = _client.messages.create(
            model=model, max_tokens=256, temperature=0,
            system=sys_prompt,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text if resp.content else ""
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        return name, parse_specialist_response(raw), tokens

    t0 = time.perf_counter()

    # Run all 4 specialists in parallel — replaces sequential loop + sleep(0.3)
    specialist_reports: dict[str, dict] = {}
    total_tokens = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, report, tokens in executor.map(_call_specialist, specialists):
            specialist_reports[name] = report
            total_tokens += tokens

    # Log specialist scores for debugging
    scores = {n: r.get("match_score", "?") for n, r in specialist_reports.items()}
    verdicts = {n: r.get("verdict", "?") for n, r in specialist_reports.items()}
    print(f"  specialists scores={scores} verdicts={verdicts}")

    # Supervisor call — images included so it can visually verify specialist reports
    client = anthropic.Anthropic(api_key=api_key)
    sup_text = supervisor_prompt(specialist_reports)
    sup_content = [
        _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
        _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
        {"type": "text", "text": sup_text},
    ]
    resp = client.messages.create(
        model=model, max_tokens=256, temperature=0,
        system=SUPERVISOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": sup_content}],
    )
    raw_sup = resp.content[0].text if resp.content else ""
    total_tokens += (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
    latency_ms = (time.perf_counter() - t0) * 1000
    return raw_sup, latency_ms, total_tokens


def call_self_consistency(pre_b64: str, post_b64: str, model: str, tile_id: str = "", n_votes: int = 3, diff_b64: str | None = None, cnn_probs: dict | None = None) -> tuple[str, float, int]:
    """Run the general VLM agent n_votes times at temperature=0.4, return majority-vote result."""
    import anthropic
    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("claude_api_key not set in .env")

    from disaster_bench.models.damage.vlm_prompts import get_system_prompt, ungrounded_prompt, parse_vlm_response, DAMAGE_SCHEMA
    import json

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    system = get_system_prompt(tile_id)
    prompt = ungrounded_prompt(full_tile=False, diff_overlay=bool(diff_b64), cnn_probs=cnn_probs)
    content = [
        _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
        _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
    ]
    if diff_b64:
        content += [_img_block(diff_b64), {"type": "text", "text": "Change heatmap above (brighter = more change)."}]
    content.append({"type": "text", "text": prompt})

    client = anthropic.Anthropic(api_key=api_key)
    votes: list[str] = []
    total_tokens = 0
    t0 = time.perf_counter()

    for _ in range(n_votes):
        resp = client.messages.create(
            model=model, max_tokens=512, temperature=0.4,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text if resp.content else ""
        total_tokens += (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        parsed = parse_vlm_response(raw)
        label = parsed.get("damage_level") or "no-damage"
        votes.append(label)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Majority vote — pure plurality, no severity bias in tie-break
    _severity = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    # Among tied classes, pick the first one in vote order for stability
    winner = next(v for v in votes if counts[v] == max_count)

    print(f"  sc_votes={votes} → {winner}")

    # Return a JSON string matching the supervisor schema shape so existing parsing works
    result = json.dumps({"damage_level": winner, "confidence": "medium", "reasoning": f"self-consistency votes: {votes}"})
    return result, latency_ms, total_tokens


def call_cascade(pre_b64: str, post_b64: str, model: str, tile_id: str = "", diff_b64: str | None = None) -> tuple[str, float, int]:
    """
    2-stage cascade:
    Stage 1 — BOUNDARY_V1 structured questions (1 call, temperature=0):
        Q1: Is damage visible on the building?   → no  = no-damage
        Q2: Is the structure still coherent?     → no  = destroyed
        Q3: Is severity minor or major?          → resolves minor/major
    Stage 2 — self-consistency 3x fallback when Stage 1 is unclear or abstains.
    Cost: 1 call for easy/clear cases, 1+3 calls for ambiguous ones.
    """
    import anthropic
    import json as _json

    api_key = os.getenv("claude_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("claude_api_key not set in .env")

    def _img_block(data_url: str) -> dict:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    client = anthropic.Anthropic(api_key=api_key)
    total_tokens = 0
    t0 = time.perf_counter()

    # ── Stage 1: Boundary questions ──────────────────────────────────────────
    content_s1 = [
        _img_block(pre_b64), {"type": "text", "text": "Pre-disaster image above."},
        _img_block(post_b64), {"type": "text", "text": "Post-disaster image above."},
    ]
    if diff_b64:
        content_s1 += [_img_block(diff_b64), {"type": "text", "text": "Change heatmap above (brighter = more change)."}]
    content_s1.append({"type": "text", "text": boundary_v1_prompt(diff_overlay=bool(diff_b64))})
    resp_s1 = client.messages.create(
        model=model, max_tokens=512, temperature=0,
        system=BOUNDARY_V1_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content_s1}],
    )
    raw_s1 = resp_s1.content[0].text if resp_s1.content else ""
    total_tokens += (resp_s1.usage.input_tokens or 0) + (resp_s1.usage.output_tokens or 0)
    b = parse_boundary_v1_response(raw_s1)

    visible   = b.get("damage_visible_on_building")  # yes / no / unclear
    coherent  = b.get("structure_still_coherent")    # yes / no / unclear
    severity  = b.get("severity_if_damaged")         # minor / major / unclear
    abstain   = b.get("abstain", False)
    conf      = b.get("confidence_band", "low")

    print(f"  s1: visible={visible} coherent={coherent} severity={severity} abstain={abstain} conf={conf}")

    # Map to label when answers are unambiguous
    label: str | None = None
    if not abstain:
        if visible == "no":
            label = "no-damage"
        elif visible == "yes" and coherent == "no":
            label = "destroyed"
        elif visible == "yes" and coherent == "yes":
            if severity == "minor":
                label = "minor-damage"
            elif severity == "major":
                label = "major-damage"

    if label is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  s1_resolved → {label}  (1 call)")
        result = _json.dumps({"damage_level": label, "confidence": conf, "reasoning": "boundary_v1"})
        return result, latency_ms, total_tokens

    # ── Stage 2: Self-consistency fallback ───────────────────────────────────
    print(f"  s1_unclear → self-consistency 3x fallback")
    raw_sc, _sc_lat, sc_tokens = call_self_consistency(pre_b64, post_b64, model, tile_id=tile_id, diff_b64=diff_b64)
    total_tokens += sc_tokens
    latency_ms = (time.perf_counter() - t0) * 1000
    return raw_sc, latency_ms, total_tokens


def load_cases(limit: int, balanced: bool, scenes: list[str] | None = None, use_diff: bool = False) -> list[dict]:
    # Use the richer CSV (with masked paths + WKT) when diff is requested
    csv_path = EVAL_CSV_FULL if (use_diff and EVAL_CSV_FULL.exists()) else EVAL_CSV
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pre = Path(row["pre_path"])
            post = Path(row["post_path"])
            if pre.exists() and post.exists():
                if scenes is None or row["tile_id"] in scenes:
                    rows.append(row)

    if balanced:
        per_class = max(1, limit // 4)
        selected, counts = [], defaultdict(int)
        for row in rows:
            lbl = row["gt_label"]
            if counts[lbl] < per_class:
                selected.append(row)
                counts[lbl] += 1
        return selected[:limit]

    return rows[:limit]


def run(limit: int, balanced: bool, model_override: str | None = None, use_cnn_probs: bool = False, multi_agent: bool = False, self_consistency: bool = False, cascade: bool = False, use_diff: bool = False, scenes: list[str] | None = None, scene_scout: bool = False, scene_worker: bool = False, custom_prompts: bool = False, hierarchy: bool = False, stage0: bool = False) -> None:
    model = model_override or os.getenv("OPENROUTER_VLM_MODEL", "")
    if not model:
        print("ERROR: pass --model or set OPENROUTER_VLM_MODEL in .env")
        sys.exit(1)

    cnn_probs_lookup: dict[str, dict] = {}
    if use_cnn_probs:
        cnn_probs_lookup = load_cnn_probs()
        print(f"  Loaded CNN probs for {len(cnn_probs_lookup)} buildings")

    cases = load_cases(limit, balanced, scenes=scenes, use_diff=use_diff)
    total = len(cases)
    if total == 0:
        print("ERROR: no cases found — check vlm_eval_sample.csv paths")
        sys.exit(1)

    mode = ("multi-agent" if multi_agent else
            "cascade" if cascade else
            "self-consistency-3x" if self_consistency else
            "hierarchy-L1L2" if hierarchy else
            "scene-worker-L1" if scene_worker else
            ("cnn-guided" if use_cnn_probs else "standard"))
    if stage0:
        mode += "+stage0"
    if scene_scout:
        mode += "+scene-scout"
    if custom_prompts:
        mode += "+custom-prompts"
    print(f"\n{'=' * 65}")
    print(f"  VLM Eval  |  model={model}  |  n={total}  |  mode={mode}")
    print(f"{'=' * 65}")

    # Pre-generate custom scene prompts — one call per unique tile_id
    # Manual overrides (from fix_scene_prompts.py) take priority over auto-generated prompts
    custom_scene_prompts: dict[str, str] = {}
    if custom_prompts:
        manual_overrides = load_scene_overrides()
        scene_ids = sorted({r["tile_id"] for r in cases})
        print(f"  Generating custom prompts for {len(scene_ids)} scenes "
              f"({len(manual_overrides)} manual overrides available)...")
        for tile_id in scene_ids:
            if tile_id in manual_overrides:
                custom_scene_prompts[tile_id] = manual_overrides[tile_id]
                print(f"  Prompt [OVERRIDE] {tile_id[:42]}")
                continue
            try:
                prompt = generate_scene_prompt(tile_id, model)
                if prompt:
                    custom_scene_prompts[tile_id] = prompt
                    print(f"  Prompt [AUTO] {tile_id[:42]}: {prompt[:80]}...")
                else:
                    print(f"  Prompt SKIPPED [{tile_id}]: tile not available, using default")
            except Exception as e:
                print(f"  Prompt FAILED [{tile_id}]: {e}")
            time.sleep(0.3)
        print(f"  Ready: {len(custom_scene_prompts)}/{len(scene_ids)} prompts\n")

    # Pre-compute scene scouts (Stage 1) — one call per unique tile_id
    scene_descriptions: dict[str, str] = {}
    if scene_scout:
        scene_to_cases: dict[str, list[dict]] = defaultdict(list)
        for r in cases:
            scene_to_cases[r["tile_id"]].append(r)
        print(f"  Running scene scouts for {len(scene_to_cases)} unique scenes...")
        for tile_id, tile_cases in sorted(scene_to_cases.items()):
            pre_images: list[str] = []
            for r in tile_cases[:3]:
                try:
                    pre_images.append(img_to_b64(Path(r["pre_path"])))
                except Exception:
                    pass
            if pre_images:
                try:
                    desc = call_scene_scout(pre_images, model, tile_id)
                    scene_descriptions[tile_id] = desc
                    print(f"  Scout {tile_id[:42]}: {desc[:90]}...")
                except Exception as e:
                    print(f"  Scout FAILED {tile_id}: {e}")
                time.sleep(0.3)
        print(f"  Scouted {len(scene_descriptions)}/{len(scene_to_cases)} scenes\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"predictions_{ts}.csv"
    fieldnames = ["tile_id", "uid", "gt_label", "pred_label",
                  "confidence", "key_evidence", "parse_error", "latency_ms", "tokens_used"]

    correct = 0
    per_class: dict[str, dict] = {c: {"total": 0, "correct": 0} for c in DAMAGE_CLASSES}

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(cases, 1):
            gt = row["gt_label"]
            tile_id = row["tile_id"]
            uid = row["uid"]

            try:
                pre_path  = Path(row["pre_path"])
                post_path = Path(row["post_path"])
                pre_b64  = img_to_b64(pre_path)
                post_b64 = img_to_b64(post_path)
                diff_b64 = make_diff_b64(row) if use_diff else None
                cnn_probs = cnn_probs_lookup.get(uid) if use_cnn_probs else None
                if custom_prompts and tile_id in custom_scene_prompts:
                    scene_desc = "__CUSTOM__:" + custom_scene_prompts[tile_id]
                elif scene_scout:
                    scene_desc = scene_descriptions.get(tile_id, "")
                else:
                    scene_desc = ""
                # Retry up to 3 times with backoff on rate limit errors
                for attempt in range(3):
                    try:
                        if multi_agent:
                            raw, latency_ms, tokens = call_multi_agent(pre_b64, post_b64, model, tile_id=tile_id)
                        elif cascade:
                            raw, latency_ms, tokens = call_cascade(pre_b64, post_b64, model, tile_id=tile_id, diff_b64=diff_b64)
                        elif self_consistency:
                            raw, latency_ms, tokens = call_self_consistency(pre_b64, post_b64, model, tile_id=tile_id, diff_b64=diff_b64, cnn_probs=cnn_probs)
                        elif hierarchy:
                            raw, latency_ms, tokens = call_l2_hierarchy(pre_b64, post_b64, model, tile_id=tile_id, scene_description=scene_desc, use_stage0=stage0, cnn_probs=cnn_probs)
                        elif scene_worker:
                            worker_report, latency_ms, tokens = call_scene_worker(pre_b64, post_b64, model, tile_id=tile_id)
                            ind = worker_report.get("damage_indicators", {})
                            print(f"  worker: {worker_report.get('initial_assessment')} | "
                                  f"roof_miss={ind.get('roof_over_half_missing')} "
                                  f"wall_col={ind.get('wall_collapse_visible')} "
                                  f"standing={ind.get('structure_still_standing')} "
                                  f"rubble={ind.get('footprint_replaced_by_rubble')} "
                                  f"surface={ind.get('surface_damage_only')}")
                            import json as _json
                            raw = _json.dumps({
                                "damage_level": worker_report.get("initial_assessment", "no-damage"),
                                "confidence": worker_report.get("confidence", "low"),
                                "key_evidence": worker_report.get("evidence_summary", ""),
                            })
                        else:
                            raw, latency_ms, tokens = call_vlm(pre_b64, post_b64, model, cnn_probs=cnn_probs, tile_id=tile_id, diff_b64=diff_b64, scene_description=scene_desc)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        wait = 10 * (attempt + 1)
                        print(f"  [retry {attempt+1}] {e} — waiting {wait}s")
                        time.sleep(wait)
                if hierarchy:
                    parsed = parse_l2_supervisor_response(raw)
                elif multi_agent or self_consistency or cascade:
                    parsed = parse_supervisor_response(raw)
                else:
                    parsed = parse_vlm_response(raw)
                pred = parsed.get("damage_level", "no-damage") or "no-damage"
                parse_err = parsed.get("parse_error", "")
                evidence = parsed.get("key_evidence", "")
                confidence = parsed.get("confidence", "")
            except Exception as exc:
                pred = "error"
                parse_err = str(exc)[:120]
                latency_ms = 0
                tokens = 0
                evidence = ""
                confidence = ""

            ok = pred == gt
            correct += int(ok)
            if gt in per_class:
                per_class[gt]["total"] += 1
                per_class[gt]["correct"] += int(ok)

            tag = "✓" if ok else "✗"
            running_acc = correct / i
            print(f"[{i:3d}/{total}] {tag}  gt={gt:15s}  pred={pred:15s}  "
                  f"acc={running_acc:.0%}  {tile_id[:30]}:{uid[:8]}")

            time.sleep(0.5)  # avoid rate limits

            writer.writerow({
                "tile_id": tile_id,
                "uid": uid,
                "gt_label": gt,
                "pred_label": pred,
                "confidence": confidence,
                "key_evidence": evidence,
                "parse_error": parse_err,
                "latency_ms": round(latency_ms, 1),
                "tokens_used": tokens,
            })
            out_f.flush()

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  RESULTS vs BASELINE (gpt-4.1 55.0%)")
    print(f"{'=' * 65}")
    print(f"  Overall: {correct}/{total} = {correct/total:.1%}")
    for cls in DAMAGE_CLASSES:
        c = per_class[cls]
        if c["total"] > 0:
            pct = c["correct"] / c["total"]
            print(f"  {cls:15s}: {c['correct']}/{c['total']} = {pct:.0%}")
    print(f"\n  Saved to: {out_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=9999, help="Max buildings to test")
    p.add_argument("--balanced", action="store_true", help="Pick ~equal samples per class")
    p.add_argument("--model", default=None, help="Model override (e.g. claude-sonnet-4-6)")
    p.add_argument("--cnn_probs", action="store_true", help="Inject CNN softmax priors into VLM prompt")
    p.add_argument("--diff", action="store_true", help="Add amplified pixel-diff heatmap as Image 3")
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--standard", action="store_true", help="Single-call mode (for comparison baseline)")
    mode_group.add_argument("--self_consistency", action="store_true", help="Run general agent 3x at temp=0.4, majority vote (3x API calls)")
    mode_group.add_argument("--multi_agent", action="store_true", help="4 specialist agents + supervisor (5x API calls)")
    mode_group.add_argument("--cascade", action="store_true", help="2-stage: boundary questions → self-consistency fallback (1–4 calls)")
    mode_group.add_argument("--scene_worker", action="store_true", help="Level 1 hierarchical agent: full tile (S3) + crops → structured evidence report")
    mode_group.add_argument("--hierarchy", action="store_true", help="L1+L2 divide-and-conquer: scene worker → 4 class specialists (parallel) → supervisor (6 calls/building)")
    p.add_argument("--scenes", nargs="+", default=None, help="Filter to specific tile_ids (e.g. socal-fire_00000937)")
    p.add_argument("--scene_scout", action="store_true", help="Enable scene-scout multi-agent: one characterization call per scene injected into classifier prompt")
    p.add_argument("--custom_prompts", action="store_true", help="Generate a unique custom system prompt for each scene from the full S3 tile (1 generation call per scene)")
    p.add_argument("--stage0", action="store_true", help="Add Stage 0 binary damage detector before L1+L2 hierarchy — short-circuits to no-damage if intact detected (saves 6 calls per no-damage building)")
    args = p.parse_args()

    # Default to self-consistency when no mode flag is given
    if not args.standard and not args.multi_agent and not args.cascade and not args.scene_worker and not args.hierarchy:
        args.self_consistency = True

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    run(args.limit, args.balanced, model_override=args.model, use_cnn_probs=args.cnn_probs,
        multi_agent=args.multi_agent, self_consistency=args.self_consistency,
        cascade=args.cascade, use_diff=args.diff, scenes=args.scenes,
        scene_scout=args.scene_scout, scene_worker=args.scene_worker,
        custom_prompts=args.custom_prompts, hierarchy=args.hierarchy, stage0=args.stage0)
