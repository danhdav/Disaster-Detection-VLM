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
    "You are given geometry change metrics computed from the image pair as grounding context. "
    "Always respond in valid JSON matching the schema provided by the user. "
    "Never invent information not supported by the images or the provided metrics."
)

_USER_PROMPT_TEMPLATE = """Analyze the building damage shown in this pre/post satellite image pair.

## CNN Pre-Classification (grounding context)
- Predicted damage class : {pred_label}
- Severity               : {severity}
- Confidence             : {confidence:.0%}
- Margin (certainty gap) : {margin:.3f}
- Class probabilities    : {scores}

## Geometry Change Metrics (grounding context)
- Change score  : {change_score}  (mean absolute pixel difference, 0-1)
- Pct changed   : {pct_changed}%  (fraction of pixels with >10% intensity change)
{ssim_line}

Using the images AND the grounding context above, provide a detailed damage assessment.
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
    Stacked CNN → GPT-4o grounded pipeline for wildfire building damage assessment.

    Follows the benchmark Track 4 grounded pattern:
      1. CNN classifies pre/post image pair → damage label + confidence
      2. Geometry features computed from pixel-level diff → change_score, pct_changed, ssim
      3. VLM receives: both images + CNN result + geometry as grounding context in the prompt
      4. GPT-4o returns structured JSON assessment
    """

    def __init__(self, weights_path: str, openai_api_key: str | None = None):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY env var.")
        self._cnn    = load_model(weights_path)
        self._client = OpenAI(api_key=api_key)

    def assess(
        self,
        pre_bytes:  bytes,
        post_bytes: bytes,
        lat: float | None = None,
        lon: float | None = None,
        model: str = "gpt-4o",
    ) -> dict:
        """
        Run the full CNN → VLM grounded pipeline on a pre/post image pair.

        Args:
            pre_bytes:  Raw bytes of the pre-disaster image.
            post_bytes: Raw bytes of the post-disaster image.
            lat:        Optional building latitude (stored in result).
            lon:        Optional building longitude (stored in result).
            model:      OpenAI model to use.

        Returns:
            Dict with cnn result, geometry context, vlm assessment, and metadata.
        """
        t_start = time.perf_counter()

        cnn_result = predict_damage(self._cnn, pre_bytes, post_bytes)
        geom       = compute_geometry_context(pre_bytes, post_bytes)

        ssim_line = (
            f"- SSIM dissimilarity : {geom['ssim_dissim']}  (0=identical, 1=completely different)"
            if "ssim_dissim" in geom else ""
        )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            pred_label   = cnn_result["pred_label"],
            severity     = cnn_result["severity"],
            confidence   = cnn_result["confidence"],
            margin       = cnn_result["margin"],
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

        vlm_result   = json.loads(response.choices[0].message.content)
        latency_ms   = round((time.perf_counter() - t_start) * 1000)
        tokens_used  = response.usage.total_tokens if response.usage else 0

        result = {
            "cnn":         cnn_result,
            "geometry":    geom,
            "vlm":         vlm_result,
            "model_used":  model,
            "latency_ms":  latency_ms,
            "tokens_used": tokens_used,
        }
        if lat is not None:
            result["lat"] = lat
        if lon is not None:
            result["lon"] = lon
        return result
