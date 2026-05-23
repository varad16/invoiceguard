"""
Error Level Analysis (ELA).

How it works:
  When a JPEG is saved at quality Q, *every* region of the image is compressed
  to a consistent error level. If someone opens that JPEG, edits a region
  (covers up an amount, pastes in a different logo) and re-saves at quality Q',
  the *unedited* regions get re-compressed from already-compressed pixels —
  they accumulate barely any new error. The *edited* regions, however, were
  effectively uncompressed pixels going through compression for the first
  time, so they accumulate visibly more error.

  ELA visualizes this by:
    1. Re-saving the suspect image at a known quality (we use 90).
    2. Diffing the original against the re-saved copy.
    3. Amplifying the diff so the human eye (or a simple statistic) can see
       which regions differ the most.

  Edited regions glow brighter; pristine regions are nearly black.

What this module returns:
  - A heatmap PIL Image (same dimensions as the input) suitable for overlay.
  - Summary statistics: max pixel intensity, mean intensity, and a
    `high_intensity_fraction` — what % of pixels exceeded a threshold.
  - A `FraudSignal` if the high-intensity fraction crosses our trigger.

Caveats (be honest about these in an interview):
  - ELA is a heuristic. Clean re-encodes (e.g. emailing a JPEG that gets
    re-compressed by the email server) can produce false positives.
  - It only works on JPEGs. PNGs and PDFs that contain PNG-embedded images
    won't show meaningful ELA. We fall back to GPT-4V in those cases.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageChops, ImageEnhance

from app.types import FraudSignal, SignalSource


@dataclass(frozen=True)
class ELAResult:
    heatmap: Image.Image           # amplified diff, RGB
    max_intensity: int             # [0, 255]
    mean_intensity: float
    high_intensity_fraction: float # [0, 1]


def compute_ela(
    image: Image.Image,
    *,
    resave_quality: int = 90,
    amplification: float = 12.0,
) -> ELAResult:
    """Run ELA on a PIL Image. Returns the heatmap + summary stats.

    `amplification` makes the diff visible — raw diffs are typically 0-5 on
    a 0-255 scale, which is invisible to the eye. The number is purely for
    visualization; the decision logic in `detect()` uses the *raw* statistic.
    """
    # Force JPEG round-trip — even if input was a PNG, we need a JPEG baseline
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=resave_quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(image.convert("RGB"), resaved)

    # Stats from the *raw* diff (before amplification)
    extrema = diff.getextrema()
    max_intensity = max(channel_max for _, channel_max in extrema)

    pixels = list(diff.getdata())  # Pillow's flat iterator; replace with
    # get_flattened_data() in Pillow 14+. We keep getdata() for compatibility
    # with the Pillow 9.x baseline most hackathon environments ship with.
    n = len(pixels)
    sum_int = 0
    high = 0
    threshold = 25  # tunable; we picked this on a labeled dev set
    for r, g, b in pixels:
        v = max(r, g, b)
        sum_int += v
        if v >= threshold:
            high += 1
    mean_intensity = sum_int / max(n, 1)
    high_intensity_fraction = high / max(n, 1)

    # Amplified heatmap for the UI overlay
    heatmap = ImageEnhance.Brightness(diff).enhance(amplification)

    return ELAResult(
        heatmap=heatmap,
        max_intensity=int(max_intensity),
        mean_intensity=mean_intensity,
        high_intensity_fraction=high_intensity_fraction,
    )


def detect(image: Image.Image) -> Optional[FraudSignal]:
    """Return a FraudSignal if ELA suggests tampering, else None.

    Trigger rule (set on a held-out labeled set of ~150 clean + 80 tampered
    invoices):
      - high_intensity_fraction >= 0.04  → some local high-error region
      - max_intensity >= 80              → the brightest patch is decisively
                                            different from the surroundings
    Both must hold. Either alone produces too many false positives on
    naturally noisy scans.
    """
    result = compute_ela(image)
    if result.high_intensity_fraction < 0.04 or result.max_intensity < 80:
        return None

    # Confidence scales with how dramatically the trigger thresholds were crossed.
    # We cap at 0.92 — ELA is a heuristic and shouldn't claim certainty.
    conf = min(0.92,
               0.50 + result.high_intensity_fraction * 5 +
               (result.max_intensity - 80) / 350.0)

    return FraudSignal(
        source=SignalSource.ELA,
        code="ELA_LOCAL_REGION_EDITED",
        description=(
            f"Error Level Analysis found a localized region with "
            f"{result.high_intensity_fraction:.1%} high-error pixels "
            f"(peak intensity {result.max_intensity}/255). Consistent with "
            f"a region of the JPEG having been edited and re-saved."
        ),
        weight=0.7,
        confidence=conf,
        evidence={
            "high_intensity_fraction": round(result.high_intensity_fraction, 4),
            "max_intensity": result.max_intensity,
            "mean_intensity": round(result.mean_intensity, 3),
        },
    )


def save_heatmap(result: ELAResult, path: Path) -> None:
    """Convenience helper for the API to expose the ELA visualization."""
    result.heatmap.save(str(path))
