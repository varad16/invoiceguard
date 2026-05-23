"""
Core domain types for InvoiceGuard.

Everything downstream — detectors, scorers, the API layer — speaks in these
dataclasses. Keeping them immutable + JSON-serializable means the API and the
test fixtures can share the exact same shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as Date
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SignalSource(str, Enum):
    """Which subsystem produced a given fraud signal."""
    GPT4V = "gpt4v"               # visual tamper detection (font, logo, pixels)
    ELA = "ela"                   # Error Level Analysis on JPEG compression
    LAYOUT = "layout"             # LayoutLMv3 structural extraction
    DUPLICATE = "duplicate"       # hash + fuzzy duplicate matcher
    VENDOR = "vendor"             # synthetic vendor verification


@dataclass(frozen=True)
class ExtractedFields:
    """Structured fields extracted by LayoutLMv3 (or its fallback).

    All fields are optional because real-world invoices are messy — a corner
    case might not have a tax_id, or OCR may miss the date. Downstream code
    treats `None` as "unknown" rather than as "missing/suspicious"; the
    vendor verifier handles the missing-field signal separately.
    """
    vendor_name: str | None = None
    vendor_address: str | None = None
    vendor_tax_id: str | None = None
    invoice_number: str | None = None
    invoice_date: Date | None = None
    due_date: Date | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None
    line_items: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""             # OCR fallback dump for downstream regex / debugging

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # dates aren't JSON-serializable
        for k in ("invoice_date", "due_date"):
            if d[k] is not None:
                d[k] = d[k].isoformat()
        return d


@dataclass(frozen=True)
class FraudSignal:
    """A single observation contributing to the final risk score.

    `weight` is the *severity* in [0, 1] — how strongly this signal pushes
    toward HIGH risk. `confidence` is the detector's certainty that the
    signal is real (e.g. GPT-4V's own confidence in its visual analysis).
    The final risk score uses weight × confidence so a low-confidence high-
    severity signal doesn't dominate.
    """
    source: SignalSource
    code: str                     # short identifier, e.g. "FONT_INCONSISTENCY"
    description: str              # human-readable, surfaces in the API response
    weight: float                 # [0, 1] — severity of this kind of fraud
    confidence: float             # [0, 1] — detector's confidence
    evidence: dict[str, Any] = field(default_factory=dict)  # detector-specific extras

    @property
    def score(self) -> float:
        return max(0.0, min(1.0, self.weight)) * max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "score": round(self.score, 3),
                "source": self.source.value}


@dataclass(frozen=True)
class RiskReport:
    """The full fraud-detection result for one invoice.

    `risk_level` is derived from the aggregated score; `signals` is the
    ordered list of contributing observations (highest score first).
    """
    invoice_id: str
    risk_level: RiskLevel
    risk_score: float             # [0, 1]
    signals: list[FraudSignal]
    extracted_fields: ExtractedFields
    processing_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 3),
            "signals": [s.to_dict() for s in self.signals],
            "extracted_fields": self.extracted_fields.to_dict(),
            "processing_ms": self.processing_ms,
        }
