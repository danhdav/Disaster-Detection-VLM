"""
CNN + VLM grounded pipeline for wildfire building damage assessment.

Mirrors the benchmark Track 4 grounded mode:
  - CNN provides fast damage classification (pre/post 6-channel input)
  - Geometry features (change_score, pct_changed, ssim_dissim) ground the VLM prompt
  - GPT-4o receives: pre image + post image + CNN result + geometry context
  - Returns structured JSON assessment matching benchmark output schema

Usage:
    from cnn.vlm_pipeline import DisasterPipeline

    pipeline = DisasterPipeline(weights_path="cnn/weights/model.pt")
    result   = pipeline.assess(pre_bytes, post_bytes, lat=38.49, lon=-122.77)
"""

import base64
import json
import os
import time

from openai import OpenAI

from .model import DAMAGE_CLASSES, DAMAGE_SEVERITY
from .predict import compute_geometry_context, load_model, predict_damage


_SYSTEM_PROMPT = (
    "You are an expert wildfire building damage assessment AI. "
    "You analyze pairs of pre-disaster and post-disaster satellite images. "
    "Always respond in valid JSON matching the schema provided by the user. "
    "Never invent information not supported by the images or the provided metrics."
)

# Used when CNN confidence >= threshold: anchor the VLM to the CNN result
_GROUNDED_PROMPT = """Analyze the building damage shown in this pre/post satellite image pair.

## CNN Pre-Classification (high-confidence grounding)
- Predicted damage class : {pred_label}
- Severity               : {severity}
- Confidence             : {confidence:.0%}
- Margin (certainty gap) : {margin:.3f}
- Class probabilities    : {scores}

## Geometry Change Metrics
- Change score  : {change_score}  (mean absolute pixel difference, 0-1)
- Pct changed   : {pct_changed}%  (fraction of pixels with >10% intensity change)
{ssim_line}

The CNN is highly confident. Use the images AND the grounding context above.
You may override the CNN prediction ONLY if the visual evidence strongly contradicts it.
Respond ONLY with a JSON object using this exact schema:

{{
  "damage_level"     : "<no-damage|minor-damage|major-damage|destroyed>",
  "severity"         : "<LOW|MODERATE|HIGH|SEVERE>",
  "confidence"       : <0.0-1.0>,
  "reasoning"        : "<1-2 sentences explaining the visual evidence>",
  "damage_indicators": ["<indicator 1>", "<indicator 2>", ...],
  "recommended_actions": ["<action 1>", "<action 2>", ...],
  "assessment_summary" : "<2-3 sentence narrative for responders>"
}}"""

# Used when CNN confidence < threshold: VLM decides purely from images + geometry
_BLIND_PROMPT = """Analyze the building damage shown in this pre/post satellite image pair.
The automated classifier was uncertain — rely on the images and geometry metrics below.

## Geometry Change Metrics
- Change score  : {change_score}  (mean absolute pixel difference, 0-1)
- Pct changed   : {pct_changed}%  (fraction of pixels with >10% intensity change)
{ssim_line}

## CNN probabilities (uncertain — treat as weak hints only)
{scores}

Assess the damage level by carefully examining the visual differences between the pre and post
images. Pay attention to: roof integrity, structural collapse, debris, char/burn patterns,
and remaining building footprint.
Respond ONLY with a JSON object using this exact schema:

{{
  "damage_level"     : "<no-damage|minor-damage|major-damage|destroyed>",
  "severity"         : "<LOW|MODERATE|HIGH|SEVERE>",
  "confidence"       : <0.0-1.0>,
  "reasoning"        : "<1-2 sentences explaining the visual evidence>",
  "damage_indicators": ["<indicator 1>", "<indicator 2>", ...],
  "recommended_actions": ["<action 1>", "<action 2>", ...],
  "assessment_summary" : "<2-3 sentence narrative for responders>"
}}"""


def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


class DisasterPipeline:
    """
    Stacked CNN → GPT-4o pipeline with conditional grounding.

    Routing logic (Option B):
      - CNN confidence >= confidence_threshold  →  GROUNDED mode
          VLM is anchored to the CNN prediction; may override only if images strongly disagree.
      - CNN confidence <  confidence_threshold  →  BLIND mode
          CNN output is withheld from VLM; it reasons purely from images + geometry.
          This prevents the VLM from echoing a wrong, uncertain CNN prediction.

    Default threshold: 0.75  (configurable via `confidence_threshold` arg)
    """

    def __init__(
        self,
        weights_path: str,
        openai_api_key: str | None = None,
        confidence_threshold: float = 0.75,
    ):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY env var.")
        self._cnn       = load_model(weights_path)
        self._client    = OpenAI(api_key=api_key)
        self._threshold = confidence_threshold

    def assess(
        self,
        pre_bytes:  bytes,
        post_bytes: bytes,
        lat: float | None = None,
        lon: float | None = None,
        model: str = "gpt-4o",
    ) -> dict:
        """
        Run the CNN → VLM pipeline with conditional grounding.

        Returns:
            Dict with keys: cnn, geometry, vlm, vlm_mode ("grounded"|"blind"),
            model_used, latency_ms, tokens_used.
        """
        t_start = time.perf_counter()

        cnn_result = predict_damage(self._cnn, pre_bytes, post_bytes)
        geom       = compute_geometry_context(pre_bytes, post_bytes)

        ssim_line = (
            f"- SSIM dissimilarity : {geom['ssim_dissim']}  (0=identical, 1=completely different)"
            if "ssim_dissim" in geom else ""
        )

        # Route: grounded if CNN is confident, blind if CNN is uncertain
        cnn_confident = cnn_result["confidence"] >= self._threshold
        if cnn_confident:
            vlm_mode    = "grounded"
            user_prompt = _GROUNDED_PROMPT.format(
                pred_label   = cnn_result["pred_label"],
                severity     = cnn_result["severity"],
                confidence   = cnn_result["confidence"],
                margin       = cnn_result["margin"],
                scores       = cnn_result["scores"],
                change_score = geom["change_score"],
                pct_changed  = geom["pct_changed"],
                ssim_line    = ssim_line,
            )
        else:
            vlm_mode    = "blind"
            user_prompt = _BLIND_PROMPT.format(
                scores       = cnn_result["scores"],
                change_score = geom["change_score"],
                pct_changed  = geom["pct_changed"],
                ssim_line    = ssim_line,
            )

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(pre_bytes)}",  "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(post_bytes)}", "detail": "high"}},
            {"type": "text", "text": user_prompt},
        ]

        response = self._client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
        )

        vlm_result  = json.loads(response.choices[0].message.content)
        latency_ms  = round((time.perf_counter() - t_start) * 1000)
        tokens_used = response.usage.total_tokens if response.usage else 0

        result = {
            "cnn":         cnn_result,
            "geometry":    geom,
            "vlm":         vlm_result,
            "vlm_mode":    vlm_mode,
            "model_used":  model,
            "latency_ms":  latency_ms,
            "tokens_used": tokens_used,
        }
        if lat is not None:
            result["lat"] = lat
        if lon is not None:
            result["lon"] = lon
        return result
