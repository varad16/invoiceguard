"""
GPT-4V visual tamper detector.

What this catches that pure OCR / structural analysis misses:
  - Font inconsistency within a single field (one digit in a different font
    because the amount was edited)
  - Logo quality issues (low-resolution paste-in from a screen capture)
  - Pixel-level artifacts around edited regions (compression seams, color
    temperature shifts)
  - Alignment shifts where text was moved or replaced
  - Suspicious whitespace/redaction that suggests something was covered up

How:
  We send the invoice image (base64) to GPT-4V with a tightly-scoped prompt
  that asks for JSON output. The prompt enumerates *exactly* the visual
  classes of fraud we care about, asks for evidence per finding, and asks
  for a per-finding confidence in [0, 1]. We then map the JSON onto our
  internal `FraudSignal` type.

Why a strict JSON schema and not free-form text:
  Free-form GPT-4V output is impossible to score automatically and tends to
  hedge ("might be tampered, hard to say"). A schema forces the model to
  either commit to a specific finding with evidence or stay silent.

The OpenAI dependency is lazy-imported so the rest of the codebase (tests,
ELA, layout) runs without an API key configured.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.types import FraudSignal, SignalSource


_SYSTEM_PROMPT = """You are a forensic invoice auditor. You inspect invoice images for visual signs of tampering. You are precise, conservative, and only flag findings you can support with specific visual evidence.

You return ONLY valid JSON matching the schema. No prose, no markdown fences."""


_USER_PROMPT = """Inspect this invoice image for visual signs of tampering. Look specifically for:

1. FONT_INCONSISTENCY — one or more characters in a numeric field (especially TOTAL, SUBTOTAL, line item amounts) rendered in a different font, weight, or size from the surrounding characters. This is the classic edit-the-amount tell.

2. LOGO_QUALITY — the vendor logo appears low-resolution, pixelated, or has compression artifacts that don't match the rest of the document. Suggests a logo pasted from a low-quality screenshot.

3. PIXEL_ARTIFACTS — visible compression seams, color temperature differences, or sharp edges around a region of the document that suggest something was pasted in.

4. ALIGNMENT_SHIFT — text or numbers appear to be misaligned with the column or row they belong to, in a way inconsistent with the document's typesetting elsewhere.

5. WHITESPACE_REDACTION — suspicious whitespace blocks, white rectangles covering text, or visible "patches" of background that suggest something was covered up.

Return JSON of the form:
{
  "findings": [
    {
      "code": "FONT_INCONSISTENCY" | "LOGO_QUALITY" | "PIXEL_ARTIFACTS" | "ALIGNMENT_SHIFT" | "WHITESPACE_REDACTION",
      "description": "<one-sentence specific description of what you see and where>",
      "confidence": <float 0..1>
    }
  ]
}

If the invoice looks clean, return {"findings": []}. Do not hedge. Do not invent findings to seem thorough."""


# Severity weight per fraud class — calibrated against TreeHacks dev set
_WEIGHTS: dict[str, float] = {
    "FONT_INCONSISTENCY": 0.85,    # near-definitive of amount tampering
    "LOGO_QUALITY": 0.45,          # often legitimate (scanned copies)
    "PIXEL_ARTIFACTS": 0.65,
    "ALIGNMENT_SHIFT": 0.55,
    "WHITESPACE_REDACTION": 0.70,
}


def _image_to_data_url(image: Image.Image, max_side: int = 1600) -> str:
    """Encode the image as a base64 data URL, resizing if very large.

    GPT-4V can ingest up to 2048×2048 well; beyond that we waste tokens.
    We also downsize aggressively for very large invoice scans because the
    detail GPT-4V needs (font shape, edge artifacts) lives well below
    1600px on the long edge.
    """
    img = image.copy()
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class GPT4VClient:
    """Thin wrapper. Lazy-instantiates the OpenAI client on first call.

    Reads API key from `OPENAI_API_KEY` env var. If the key is missing or
    the openai package isn't installed, `detect()` raises `RuntimeError`,
    and the orchestrator skips this detector while still running the others.
    """
    model: str = "gpt-4o"          # has vision; cheaper + faster than gpt-4-turbo

    def detect(self, image: Image.Image) -> list[FraudSignal]:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("openai package not installed") from e

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=key)

        data_url = _image_to_data_url(image)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,        # deterministic — forensic, not creative
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": _USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ]},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return self._parse_findings(content)

    @staticmethod
    def _parse_findings(content: str) -> list[FraudSignal]:
        """Parse the GPT-4V JSON response into FraudSignal objects.

        We're defensive here because model output, even with response_format
        set, can occasionally include a stray non-finding key or a slightly
        off-spec finding object. We skip malformed entries rather than
        crashing the whole detector.
        """
        try:
            data: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            return []

        findings = data.get("findings", [])
        if not isinstance(findings, list):
            return []

        out: list[FraudSignal] = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            code = f.get("code")
            desc = f.get("description")
            conf_raw = f.get("confidence")
            if not (isinstance(code, str) and isinstance(desc, str)
                    and isinstance(conf_raw, (int, float))):
                continue
            if code not in _WEIGHTS:
                continue
            confidence = max(0.0, min(1.0, float(conf_raw)))
            out.append(FraudSignal(
                source=SignalSource.GPT4V,
                code=code,
                description=desc,
                weight=_WEIGHTS[code],
                confidence=confidence,
                evidence={},
            ))
        return out


def detect(image: Image.Image, client: GPT4VClient | None = None) -> list[FraudSignal]:
    """Convenience entry point used by the orchestrator."""
    c = client or GPT4VClient()
    return c.detect(image)
