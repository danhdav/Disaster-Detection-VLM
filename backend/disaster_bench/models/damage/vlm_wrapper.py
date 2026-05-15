"""
VLM wrapper for damage classification benchmark — GitHub Models only.
Ref §3 VLM benchmarks, §4 tool-grounded vs ungrounded comparison.

Auth: GITHUB_TOKEN (PAT with models scope, or fine-grained token with models:read).
Endpoint: https://models.github.ai/inference/chat/completions

Usage:
    wrapper = get_vlm(model="openai/gpt-4o")
    result  = wrapper.classify(pre_img, post_img,
                               pre_masked=pre_mask, post_masked=post_mask,
                               grounded=True, geometry={...})

Image conventions:
    pre_img / post_img    : padded context crop with building outlined in red (H,W,3) uint8
    pre_masked/post_masked: same padded region, non-building pixels blacked out (H,W,3) uint8
    Cloud VLMs receive all 4 images when masked crops are present.

Each wrapper returns a dict: {damage_level, confidence, reasoning, latency_ms, tokens_used}.
"""
from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from typing import Any

import numpy as np

from disaster_bench.models.damage.vlm_prompts import (
    SYSTEM_PROMPT,
    MINOR_MAJOR_SYSTEM_PROMPT,
    INDICATORS_SYSTEM_PROMPT,
    BOUNDARY_V1_SYSTEM_PROMPT,
    grounded_prompt,
    ungrounded_prompt,
    minor_major_prompt,
    indicators_prompt,
    boundary_v1_prompt,
    parse_vlm_response,
    parse_boundary_v1_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_b64(img: np.ndarray, fmt: str = "PNG") -> str:
    """Convert numpy (H,W,3) uint8 to base64-encoded PNG string."""
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_diff_overlay(pre: np.ndarray, post: np.ndarray, amplify: int = 4) -> np.ndarray:
    """
    Compute a pixel-difference heatmap between pre and post crops.
    Returns (H, W, 3) uint8 where brightness encodes change magnitude.
    Brighter = more change (amplified by `amplify`, clipped to [0, 255]).
    Both inputs must be the same shape (H, W, 3) uint8.
    """
    diff = np.abs(post.astype(np.int32) - pre.astype(np.int32))
    diff_amp = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    return diff_amp


def _load_env_key(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            val = os.environ.get(name, "")
        except ImportError:
            pass
    return val


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VLMBase:
    model_name: str = ""

    def classify(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        *,
        pre_masked: np.ndarray | None = None,
        post_masked: np.ndarray | None = None,
        diff_img: np.ndarray | None = None,
        grounded: bool = False,
        geometry: dict[str, Any] | None = None,
        minor_major: bool = False,
        indicators: bool = False,
        boundary_v1: bool = False,
        cnn_probs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _make_prompt(
        self,
        grounded: bool,
        geometry: dict[str, Any] | None,
        has_masked: bool = False,
        diff_overlay: bool = False,
        minor_major: bool = False,
        indicators: bool = False,
        boundary_v1: bool = False,
        cnn_probs: dict[str, Any] | None = None,
    ) -> str:
        if boundary_v1:
            return boundary_v1_prompt(use_masked=has_masked, diff_overlay=diff_overlay)
        if indicators:
            return indicators_prompt(use_masked=has_masked, diff_overlay=diff_overlay)
        if minor_major:
            return minor_major_prompt(use_masked=has_masked, diff_overlay=diff_overlay)
        if grounded and geometry:
            return grounded_prompt(geometry, use_masked=has_masked, diff_overlay=diff_overlay,
                                   cnn_probs=cnn_probs)
        return ungrounded_prompt(use_masked=has_masked, diff_overlay=diff_overlay,
                                 cnn_probs=cnn_probs)

    def _system_prompt(
        self,
        minor_major: bool = False,
        indicators: bool = False,
        boundary_v1: bool = False,
    ) -> str:
        if boundary_v1:
            return BOUNDARY_V1_SYSTEM_PROMPT
        if indicators:
            return INDICATORS_SYSTEM_PROMPT
        return MINOR_MAJOR_SYSTEM_PROMPT if minor_major else SYSTEM_PROMPT

    def _make_result(
        self,
        raw_text: str,
        latency_ms: float,
        tokens: int = 0,
        mode: str = "",
        boundary_v1: bool = False,
    ) -> dict[str, Any]:
        parsed = parse_boundary_v1_response(raw_text) if boundary_v1 else parse_vlm_response(raw_text)
        return {
            **parsed,
            "model":       self.model_name,
            "mode":        mode,
            "latency_ms":  round(latency_ms, 1),
            "tokens_used": tokens,
        }


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

class OpenAIVLM(VLMBase):
    """
    OpenAI API (api.openai.com).
    Auth: OPENAI_API_KEY in environment or .env file.
    Model IDs: "gpt-4o", "gpt-4o-mini", etc.
    """
    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 512) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self._key = _load_env_key("OPENAI_API_KEY")

    def classify(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        *,
        pre_masked: np.ndarray | None = None,
        post_masked: np.ndarray | None = None,
        diff_img: np.ndarray | None = None,
        grounded: bool = False,
        geometry: dict[str, Any] | None = None,
        minor_major: bool = False,
        indicators: bool = False,
        boundary_v1: bool = False,
        cnn_probs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from openai import OpenAI

        if not self._key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )

        client = OpenAI(api_key=self._key, base_url=self.BASE_URL)
        has_masked = pre_masked is not None and post_masked is not None
        has_diff   = diff_img is not None and not has_masked  # diff overlay only for 2-image path
        mode   = ("grounded" if grounded and geometry else "ungrounded")
        mode  += "-masked" if has_masked else ""
        mode  += "-diff" if has_diff else ""
        mode  += "-mm" if minor_major else ""
        mode  += "-ind" if indicators else ""
        mode  += "-bv1" if boundary_v1 else ""
        mode  += "-cnn" if cnn_probs else ""
        prompt = self._make_prompt(grounded, geometry, has_masked=has_masked,
                                   diff_overlay=has_diff,
                                   minor_major=minor_major, indicators=indicators,
                                   boundary_v1=boundary_v1, cnn_probs=cnn_probs)

        def _img_entry(arr: np.ndarray) -> dict:
            b64 = _img_to_b64(arr)
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

        content: list[dict] = [{"type": "text", "text": prompt}]
        content.append(_img_entry(pre_img))
        content.append(_img_entry(post_img))
        if has_masked:
            content.append(_img_entry(pre_masked))   # type: ignore[arg-type]
            content.append(_img_entry(post_masked))  # type: ignore[arg-type]
        elif has_diff:
            content.append(_img_entry(diff_img))     # type: ignore[arg-type]

        t0 = time.perf_counter()
        # Exponential backoff on 429 / rate-limit errors
        _backoff = 8.0
        _max_retries = 4
        for _attempt in range(_max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self._system_prompt(
                            minor_major, indicators, boundary_v1)},
                        {"role": "user",   "content": content},
                    ],
                )
                break  # success
            except Exception as exc:
                msg = str(exc)
                is_rate = ("429" in msg or "Too many requests" in msg
                           or "rate" in msg.lower())
                if is_rate and _attempt < _max_retries - 1:
                    import random as _rnd
                    wait = _backoff + _rnd.uniform(-1, 1)
                    print(f"  [rate-limit] attempt {_attempt+1}/{_max_retries}, "
                          f"retrying in {wait:.1f}s ...")
                    time.sleep(wait)
                    _backoff = min(_backoff * 2, 60.0)
                else:
                    raise
        latency_ms = (time.perf_counter() - t0) * 1000
        raw  = resp.choices[0].message.content or ""
        toks = resp.usage.total_tokens if resp.usage else 0
        return self._make_result(raw, latency_ms, tokens=toks, mode=mode,
                                 boundary_v1=boundary_v1)


# ---------------------------------------------------------------------------
# Ollama (local inference, no API key, CPU-safe)
# ---------------------------------------------------------------------------

class OllamaVLM(VLMBase):
    """
    Local VLM via Ollama (http://localhost:11434).
    No rate limits, no API key. Runs on CPU with system RAM.
    Sends a single side-by-side composite image (pre | post).

    Recommended models:
        ollama pull llava:7b       # 4.5 GB — good balance
        ollama pull llava-llama3   # 4.7 GB — better reasoning
        ollama pull moondream      # 1.6 GB — fastest, weaker reasoning

    Usage:
        vlm = get_vlm("ollama/llava:7b")
        result = vlm.classify(pre_img, post_img)
    """
    def __init__(self, model: str = "llava:7b", base_url: str = "http://localhost:11434",
                 max_tokens: int = 512) -> None:
        self.model_name   = model
        self.base_url     = base_url.rstrip("/")
        self.max_tokens   = max_tokens

    def _make_composite(self, pre_img: np.ndarray, post_img: np.ndarray,
                        size: int = 256) -> np.ndarray:
        """Stack pre and post side-by-side into a single (size, 2*size, 3) image."""
        from PIL import Image as PILImage
        pre  = PILImage.fromarray(pre_img.astype(np.uint8)).resize((size, size))
        post = PILImage.fromarray(post_img.astype(np.uint8)).resize((size, size))
        composite = PILImage.new("RGB", (size * 2, size))
        composite.paste(pre,  (0, 0))
        composite.paste(post, (size, 0))
        arr = np.array(composite)
        return arr

    # Simplified prompt for local 7B models.
    # Asks for a plain label — avoids JSON schema complexity that causes empty {} output.
    _LOCAL_SYSTEM = (
        "You are a satellite image damage analyst classifying wildfire building damage."
    )
    _LOCAL_PROMPT = (
        "Two satellite images are shown side-by-side: LEFT panel = BEFORE the wildfire, "
        "RIGHT panel = AFTER the wildfire.\n\n"
        "Classify the building damage using exactly one of these labels:\n"
        "  no-damage     — no visible structural change\n"
        "  minor-damage  — partial/superficial damage, building still standing\n"
        "  major-damage  — significant damage, large portions missing but footprint visible\n"
        "  destroyed     — complete loss, only rubble or ash remains\n\n"
        "DESTROYED RULE: only use 'destroyed' if the building is clearly missing or all rubble.\n\n"
        "Reply with exactly this format:\n"
        "LABEL: <one of the four labels>\n"
        "CONFIDENCE: <low|medium|high>\n"
        "REASON: <one sentence>"
    )

    def classify(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        *,
        pre_masked: np.ndarray | None = None,
        post_masked: np.ndarray | None = None,
        grounded: bool = False,
        geometry: dict[str, Any] | None = None,
        diff_img: np.ndarray | None = None,
        minor_major: bool = False,
        indicators: bool = False,
        boundary_v1: bool = False,
        cnn_probs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import requests

        composite = self._make_composite(pre_img, post_img)
        b64 = _img_to_b64(composite)

        geo_block = ""
        if grounded and geometry:
            import json as _json
            geo_block = (f"\nGeometry measurements: {_json.dumps(geometry)}\n"
                         "Use both visual and geometry evidence.\n")

        payload = {
            "model":  self.model_name,
            "messages": [
                {"role": "system",  "content": self._LOCAL_SYSTEM},
                {"role": "user",    "content": self._LOCAL_PROMPT + geo_block,
                 "images": [b64]},
            ],
            "stream":  False,
            # No format: "json" — llava:7b returns {} when forced; parse free text instead
            "options": {"temperature": 0, "num_predict": self.max_tokens},
        }

        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        data = r.json()
        raw  = data.get("message", {}).get("content", "")
        toks = (data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0)
        mode = "grounded" if grounded and geometry else "ungrounded"
        return self._make_result(self._parse_local(raw), latency_ms, tokens=toks, mode=mode)

    @staticmethod
    def _parse_local(text: str) -> str:
        """
        Convert free-text LABEL:/CONFIDENCE:/REASON: response to a JSON string
        that parse_vlm_response can handle.
        Falls back to scanning the text for any damage label keyword.
        """
        import re, json as _json
        valid = ["no-damage", "minor-damage", "major-damage", "destroyed"]

        label = ""
        confidence = ""
        reasoning = ""

        # Try LABEL: <value> pattern first
        m = re.search(r"LABEL\s*:\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().lower().rstrip(".")
            for v in valid:
                if v in candidate:
                    label = v
                    break

        # Fallback: scan entire text for first matching label
        if not label:
            text_lower = text.lower()
            for v in valid:
                if v in text_lower:
                    label = v
                    break

        m = re.search(r"CONFIDENCE\s*:\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            confidence = m.group(1).strip().lower().rstrip(".")

        m = re.search(r"REASON\s*:\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip()

        if not label:
            label = "no-damage"

        return _json.dumps({
            "damage_level":  label,
            "confidence":    confidence if confidence in ("low", "medium", "high") else "",
            "reasoning":     reasoning,
            "key_evidence":  "",
            "considered_classes": "",
            "destroyed_gate_passed": "",
            "why_not_destroyed": "",
            "why_destroyed": "",
        })


# ---------------------------------------------------------------------------
# Anthropic (Claude) API
# ---------------------------------------------------------------------------

class AnthropicVLM(VLMBase):
    """
    Anthropic Claude API (api.anthropic.com).
    Auth: ANTHROPIC_API_KEY in environment or .env file.
    Model IDs: "claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001", etc.
    Prefix with "claude/" when using get_vlm() factory, e.g. get_vlm("claude/claude-sonnet-4-6").
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 512) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self._key = _load_env_key("ANTHROPIC_API_KEY")

    def classify(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        *,
        pre_masked: np.ndarray | None = None,
        post_masked: np.ndarray | None = None,
        diff_img: np.ndarray | None = None,
        grounded: bool = False,
        geometry: dict[str, Any] | None = None,
        minor_major: bool = False,
        indicators: bool = False,
        boundary_v1: bool = False,
        cnn_probs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import anthropic as _anthropic

        if not self._key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )

        has_masked = pre_masked is not None and post_masked is not None
        has_diff   = diff_img is not None and not has_masked
        mode   = ("grounded" if grounded and geometry else "ungrounded")
        mode  += "-masked" if has_masked else ""
        mode  += "-diff" if has_diff else ""
        mode  += "-mm" if minor_major else ""
        mode  += "-ind" if indicators else ""
        mode  += "-bv1" if boundary_v1 else ""
        mode  += "-cnn" if cnn_probs else ""
        system_prompt = self._system_prompt(minor_major, indicators, boundary_v1)
        user_prompt   = self._make_prompt(grounded, geometry, has_masked=has_masked,
                                          diff_overlay=has_diff,
                                          minor_major=minor_major, indicators=indicators,
                                          boundary_v1=boundary_v1, cnn_probs=cnn_probs)

        def _img_block(arr: np.ndarray) -> dict:
            b64 = _img_to_b64(arr)
            return {"type": "image", "source": {"type": "base64",
                    "media_type": "image/png", "data": b64}}

        content: list[dict] = []
        content.append(_img_block(pre_img))
        content.append(_img_block(post_img))
        if has_masked:
            content.append(_img_block(pre_masked))   # type: ignore[arg-type]
            content.append(_img_block(post_masked))  # type: ignore[arg-type]
        elif has_diff:
            content.append(_img_block(diff_img))     # type: ignore[arg-type]
        content.append({"type": "text", "text": user_prompt})

        client = _anthropic.Anthropic(api_key=self._key)
        t0 = time.perf_counter()
        _backoff = 8.0
        _max_retries = 4
        for _attempt in range(_max_retries):
            try:
                resp = client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": content}],
                )
                break
            except Exception as exc:
                msg = str(exc)
                is_rate = ("429" in msg or "rate" in msg.lower()
                           or "overloaded" in msg.lower())
                if is_rate and _attempt < _max_retries - 1:
                    import random as _rnd
                    wait = _backoff + _rnd.uniform(-1, 1)
                    print(f"  [rate-limit] attempt {_attempt+1}/{_max_retries}, "
                          f"retrying in {wait:.1f}s ...")
                    time.sleep(wait)
                    _backoff = min(_backoff * 2, 60.0)
                else:
                    raise

        latency_ms = (time.perf_counter() - t0) * 1000
        raw  = resp.content[0].text if resp.content else ""
        toks = (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0
        return self._make_result(raw, latency_ms, tokens=toks, mode=mode,
                                 boundary_v1=boundary_v1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vlm(model: str = "gpt-4o", **kwargs: Any):
    """
    Return a VLM wrapper for the given model ID.

    OpenAI API (requires OPENAI_API_KEY in .env):
        get_vlm("gpt-4o")
        get_vlm("gpt-4o-mini")
        get_vlm("gpt-4.1")

    Anthropic Claude API (requires ANTHROPIC_API_KEY in .env):
        get_vlm("claude/claude-sonnet-4-6")
        get_vlm("claude/claude-opus-4-6")
        get_vlm("claude/claude-haiku-4-5-20251001")

    Ollama (local, no API key needed, prefix with "ollama/"):
        get_vlm("ollama/llava:7b")
        get_vlm("ollama/llava-llama3")
        get_vlm("ollama/moondream")
    """
    if model.startswith("ollama/"):
        ollama_model = model[len("ollama/"):]
        return OllamaVLM(model=ollama_model, **kwargs)
    if model.startswith("claude/"):
        claude_model = model[len("claude/"):]
        return AnthropicVLM(model=claude_model, **kwargs)
    # Strip legacy "direct/" prefix if present
    if model.startswith("direct/"):
        model = model[len("direct/"):]
    return OpenAIVLM(model=model, **kwargs)
