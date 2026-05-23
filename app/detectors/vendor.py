"""
Synthetic vendor detector.

Catches the third type of invoice fraud: a fabricated company submitting
invoices for services never rendered. Real-world fraud patterns we look for:

  1. **Missing required identifiers** — no tax ID, no address, or both.
     Legitimate vendors over a certain size always have these.
  2. **Malformed tax IDs** — US EIN format is XX-XXXXXXX, VAT IDs have
     per-country formats. Rejecting an EIN that's "11-1111111" or
     "00-0000000" filters out the laziest fraudsters.
  3. **Generic / impossible addresses** — "123 Main St", "P.O. Box 1",
     or no street number at all.
  4. **Not in the known-vendor registry** — for accounts payable systems
     with a vetted vendor list, the absence of a record is itself a signal.

This module is rule-based and deliberately conservative — each rule has a
moderate weight and the signals stack only when multiple fire on the same
invoice. We don't want to flag a legitimate small business just because
their address is "P.O. Box 47".

The known-vendor registry is a `VendorDirectory` protocol — production
code wires this to a database; the demo uses an in-memory dict seeded
with the test fixtures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from app.types import ExtractedFields, FraudSignal, SignalSource


# US EIN: 9 digits in XX-XXXXXXX form. Rejects all-same-digit patterns.
_EIN_PATTERN = re.compile(r"^\d{2}-\d{7}$")
_ALL_SAME_DIGIT = re.compile(r"^([0-9])-?\1+$")

# A street address should have at least: number + word + suffix
_STREET_RE = re.compile(
    r"\d+\s+\w+(?:\s+\w+)*\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|"
    r"Dr|Drive|Lane|Ln|Way|Pl|Place|Pkwy|Parkway|Ct|Court)\b",
    re.IGNORECASE,
)

# Patterns we treat as obviously generic
_GENERIC_ADDRESSES = {
    "123 main st",
    "123 main street",
    "1 main st",
    "1 main street",
    "p.o. box 1",
    "po box 1",
}


class VendorDirectory(Protocol):
    """Anything the orchestrator can ask 'is this a known good vendor?'"""

    def is_known(self, vendor_name: str) -> bool: ...
    def get_tax_id(self, vendor_name: str) -> Optional[str]: ...


@dataclass
class InMemoryVendorDirectory:
    """Default implementation used by tests and the demo.

    `known` is a map of vendor_name → expected_tax_id (or None if we don't
    care about the tax ID match for this vendor).
    """
    known: dict[str, Optional[str]] = field(default_factory=dict)

    def is_known(self, vendor_name: str) -> bool:
        return vendor_name.strip().lower() in {k.lower() for k in self.known}

    def get_tax_id(self, vendor_name: str) -> Optional[str]:
        for k, v in self.known.items():
            if k.lower() == vendor_name.strip().lower():
                return v
        return None


def _valid_ein(tax_id: str) -> bool:
    if not _EIN_PATTERN.match(tax_id):
        return False
    if _ALL_SAME_DIGIT.match(tax_id):
        return False
    if tax_id in {"00-0000000", "11-1111111", "12-3456789"}:
        return False
    return True


def _generic_address(address: str) -> bool:
    normalized = address.lower().strip().rstrip(".,")
    return normalized in _GENERIC_ADDRESSES


def _malformed_address(address: str) -> bool:
    # If we can't find a street-like pattern AND it doesn't mention a PO box,
    # treat as malformed.
    if _STREET_RE.search(address):
        return False
    if re.search(r"p\.?\s?o\.?\s?box\s+\d+", address, re.IGNORECASE):
        # Has PO box. Acceptable only if it has a real number > 1
        m = re.search(r"box\s+(\d+)", address, re.IGNORECASE)
        if m and int(m.group(1)) > 1:
            return False
    return True


def detect(fields: ExtractedFields,
           directory: VendorDirectory | None = None) -> list[FraudSignal]:
    """Return zero or more FraudSignals for this invoice's vendor."""
    signals: list[FraudSignal] = []

    if not fields.vendor_name:
        # Missing vendor name itself is a soft signal — could be OCR failure
        signals.append(FraudSignal(
            source=SignalSource.VENDOR,
            code="VENDOR_NAME_MISSING",
            description="No vendor name could be extracted from the invoice.",
            weight=0.40,
            confidence=0.55,  # could be an OCR miss, not necessarily fraud
        ))
        return signals

    # Missing tax ID
    if not fields.vendor_tax_id:
        signals.append(FraudSignal(
            source=SignalSource.VENDOR,
            code="VENDOR_NO_TAX_ID",
            description=(
                f"Vendor '{fields.vendor_name}' invoice has no tax ID (EIN/VAT). "
                "Legitimate vendors of any size are required to provide one."
            ),
            weight=0.50,
            confidence=0.65,
        ))
    elif not _valid_ein(fields.vendor_tax_id):
        signals.append(FraudSignal(
            source=SignalSource.VENDOR,
            code="VENDOR_INVALID_TAX_ID",
            description=(
                f"Tax ID '{fields.vendor_tax_id}' is malformed or matches a "
                "known dummy pattern (all-same-digit, or sentinel value)."
            ),
            weight=0.80,
            confidence=0.85,
            evidence={"tax_id": fields.vendor_tax_id},
        ))

    # Missing or malformed address
    if not fields.vendor_address:
        signals.append(FraudSignal(
            source=SignalSource.VENDOR,
            code="VENDOR_NO_ADDRESS",
            description=f"Vendor '{fields.vendor_name}' invoice has no address.",
            weight=0.45,
            confidence=0.60,
        ))
    elif _generic_address(fields.vendor_address):
        signals.append(FraudSignal(
            source=SignalSource.VENDOR,
            code="VENDOR_GENERIC_ADDRESS",
            description=(
                f"Vendor address '{fields.vendor_address}' matches a known "
                "generic / placeholder pattern."
            ),
            weight=0.75,
            confidence=0.85,
        ))
    elif _malformed_address(fields.vendor_address):
        signals.append(FraudSignal(
            source=SignalSource.VENDOR,
            code="VENDOR_MALFORMED_ADDRESS",
            description=(
                f"Vendor address '{fields.vendor_address}' lacks a recognizable "
                "street format and is not a valid PO box."
            ),
            weight=0.45,
            confidence=0.60,
        ))

    # Unknown vendor (only meaningful when a directory is provided)
    if directory is not None and not directory.is_known(fields.vendor_name):
        signals.append(FraudSignal(
            source=SignalSource.VENDOR,
            code="VENDOR_NOT_IN_DIRECTORY",
            description=(
                f"Vendor '{fields.vendor_name}' is not in the approved-vendor "
                "directory. Manual onboarding may be required."
            ),
            weight=0.55,
            confidence=0.80,
        ))
    elif directory is not None and fields.vendor_tax_id:
        # Vendor IS known — check the tax ID matches what we have on file
        expected = directory.get_tax_id(fields.vendor_name)
        if expected and expected != fields.vendor_tax_id:
            signals.append(FraudSignal(
                source=SignalSource.VENDOR,
                code="VENDOR_TAX_ID_MISMATCH",
                description=(
                    f"Tax ID '{fields.vendor_tax_id}' on this invoice differs "
                    f"from the directory record for '{fields.vendor_name}' "
                    f"(expected '{expected}'). Possible impersonation."
                ),
                weight=0.92,
                confidence=0.95,
                evidence={"expected_tax_id": expected,
                          "actual_tax_id": fields.vendor_tax_id},
            ))

    return signals
