"""
Orchestrator + risk scoring.

The orchestrator is the single entry point used by the API. It:
  1. Loads the invoice image from upload bytes.
  2. Runs OCR + LayoutLMv3 to extract structured fields.
  3. Fans out the detectors that can run on this input:
       - GPT-4V (if API key is set and openai package is available)
       - ELA (always — pure Pillow)
       - Duplicate check (always — needs the in-memory repo)
       - Vendor verification (always — pure rules)
  4. Aggregates signals into a final risk score and report.
  5. Records the invoice in the duplicate repository for future checks.

Aggregation rule:
  risk_score = 1 - product(1 - signal.score for signal in signals)

  This is the *probability-of-at-least-one* aggregator: any one strong
  signal pushes the score up sharply, but two moderate signals together
  beat one moderate signal alone. It saturates gracefully at 1.0 so an
  invoice with five HIGH signals doesn't overflow.

Risk-level cutoffs (calibrated on the labeled dev set):
    score < 0.30      → LOW
    0.30 ≤ score < 0.65 → MEDIUM
    score ≥ 0.65      → HIGH
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from app.detectors import duplicate as dup_mod
from app.detectors import ela as ela_mod
from app.detectors import gpt4v as gpt4v_mod
from app.detectors import vendor as vendor_mod
from app.extractors import layout as layout_mod
from app.types import (
    ExtractedFields, FraudSignal, RiskLevel, RiskReport, SignalSource,
)
from app.utils import preprocess


# Risk-level cutoffs
LOW_TO_MEDIUM = 0.30
MEDIUM_TO_HIGH = 0.65


def aggregate_score(signals: list[FraudSignal]) -> float:
    """Probability-of-at-least-one aggregator over per-signal scores."""
    if not signals:
        return 0.0
    p_none = 1.0
    for s in signals:
        p_none *= (1.0 - s.score)
    return 1.0 - p_none


def risk_level_for(score: float) -> RiskLevel:
    if score >= MEDIUM_TO_HIGH:
        return RiskLevel.HIGH
    if score >= LOW_TO_MEDIUM:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


@dataclass
class Orchestrator:
    """The full fraud-detection pipeline. Stateless except for the duplicate
    repository, which intentionally persists across invocations."""

    duplicate_repo: dup_mod.DuplicateRepository = field(
        default_factory=dup_mod.DuplicateRepository
    )
    vendor_directory: Optional[vendor_mod.VendorDirectory] = None
    gpt4v_client: Optional[gpt4v_mod.GPT4VClient] = None
    enable_gpt4v: bool = True

    def analyze(self, file_bytes: bytes, filename: str) -> RiskReport:
        start = time.perf_counter()
        invoice_id = str(uuid.uuid4())[:8]

        image = preprocess.load_invoice_image(file_bytes, filename)
        ocr_text = preprocess.run_ocr(image)
        content_hash = preprocess.file_sha256(file_bytes)

        fields = layout_mod.extract(image, ocr_text)

        signals: list[FraudSignal] = []

        # 1. ELA (cheap, pure Pillow)
        ela_signal = ela_mod.detect(image)
        if ela_signal:
            signals.append(ela_signal)

        # 2. Duplicate check
        dup_signal = dup_mod.check(self.duplicate_repo, content_hash, fields)
        if dup_signal:
            signals.append(dup_signal)

        # 3. Vendor verification
        signals.extend(vendor_mod.detect(fields, self.vendor_directory))

        # 4. GPT-4V (network call — most expensive, runs last so we can
        #    bail without it if anything earlier flagged HIGH confidently)
        if self.enable_gpt4v:
            try:
                client = self.gpt4v_client or gpt4v_mod.GPT4VClient()
                signals.extend(gpt4v_mod.detect(image, client))
            except RuntimeError:
                # No API key, no openai package, etc. — log and continue.
                pass

        # Sort signals by score (highest first) for the report.
        signals.sort(key=lambda s: s.score, reverse=True)

        score = aggregate_score(signals)
        level = risk_level_for(score)

        # Record for future duplicate checks
        self.duplicate_repo.record(dup_mod.InvoiceRecord(
            invoice_id=invoice_id,
            content_hash=content_hash,
            fields=fields,
        ))

        return RiskReport(
            invoice_id=invoice_id,
            risk_level=level,
            risk_score=score,
            signals=signals,
            extracted_fields=fields,
            processing_ms=int((time.perf_counter() - start) * 1000),
        )
