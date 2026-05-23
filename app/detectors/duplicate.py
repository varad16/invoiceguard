"""
Duplicate-submission detector.

Two ways an invoice gets resubmitted fraudulently:
  1. **Exact duplicate** — same file uploaded twice. Catches naive double-
     dipping, file-system mishaps, automated bot submissions. Detected via
     SHA-256 over the raw bytes.
  2. **Near duplicate** — same invoice resubmitted with minor changes
     (date shifted by a week, invoice number incremented). Detected via
     fuzzy matching on extracted fields:
       - vendor name similarity ≥ 85% (token-set / partial-ratio)
       - totals match within $0.01  ← required
       - dates within 30 days       ← optional booster, raises confidence

We deliberately *require* the amount match for a fuzzy-duplicate flag, with
date proximity acting only as a confidence booster. Vendor + date-only is
too permissive — legitimate businesses invoice their suppliers on regular
monthly cycles, so "same vendor, dates within 30 days" by itself flags
all recurring invoices. The amount-equality requirement keeps recall on
the actual fraud pattern (resubmit-with-bumped-amount) while suppressing
recurring-invoice false positives.

The detector holds an in-memory repository of previously-seen invoices.
In production this would be backed by Postgres + a vendor-name embedding
index for sublinear lookup, but for the hackathon scope an O(N) scan over
recent invoices is fast enough — typical accounts payable departments see
< 10k invoices/month.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date, timedelta
from typing import Optional

from app.types import ExtractedFields, FraudSignal, SignalSource


@dataclass
class InvoiceRecord:
    """What we remember about an invoice we've already seen."""
    invoice_id: str
    content_hash: str
    fields: ExtractedFields


def _fuzz_ratio(a: str, b: str) -> int:
    """Vendor-name similarity in [0, 100].

    Uses `rapidfuzz` when available — its `partial_ratio` and `token_set_ratio`
    handle the common patterns of vendor-name variation (abbreviation,
    legal-suffix omission, word reordering) much better than pure Levenshtein.

    The pure-Python fallback uses a token-set Jaccard similarity over
    lowercased words, which is robust to "Acme Corporation" vs "Acme Corp"
    style variations (both share the token "acme"). This is slightly less
    accurate than rapidfuzz on the long tail but is correct enough for the
    hackathon-scope demo.
    """
    if not a or not b:
        return 0
    try:
        from rapidfuzz import fuzz  # type: ignore
        return int(max(fuzz.ratio(a, b),
                       fuzz.partial_ratio(a, b),
                       fuzz.token_set_ratio(a, b)))
    except ImportError:
        pass

    # Fallback: combined token-set Jaccard + prefix-match score.
    # We strip common legal suffixes that vary noisily across submissions.
    suffixes = {"inc", "inc.", "corp", "corp.", "corporation", "llc",
                "ltd", "ltd.", "limited", "co", "co.", "company"}

    def tokens(s: str) -> set[str]:
        return {w for w in s.lower().split() if w not in suffixes}

    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        # If stripping suffixes left nothing, fall back to raw string match.
        return 100 if a.strip().lower() == b.strip().lower() else 0

    intersect = ta & tb
    union = ta | tb
    jaccard = len(intersect) / max(len(union), 1)

    # Bonus: if every token in the *shorter* side appears in the longer side,
    # it's almost certainly an abbreviation match ("Acme" ⊂ "Acme Corporation").
    shorter = ta if len(ta) <= len(tb) else tb
    longer = tb if shorter is ta else ta
    if shorter.issubset(longer):
        # Score blends Jaccard with a strong prefix-subset signal
        return int(min(100, 70 + 30 * jaccard))

    return int(jaccard * 100)


@dataclass
class DuplicateRepository:
    """In-memory store of previously-seen invoices.

    The orchestrator (or a backing service) is responsible for calling
    `record()` after each successful invoice ingest. `check()` is read-only
    and returns at most one match — the highest-similarity prior record.
    """
    records: list[InvoiceRecord] = field(default_factory=list)

    # Tunables — defaults match the project notes
    vendor_threshold: int = 85          # %
    amount_tolerance_cents: int = 1     # cents
    date_window_days: int = 30

    def record(self, rec: InvoiceRecord) -> None:
        self.records.append(rec)

    def check(self, content_hash: str, fields: ExtractedFields) -> Optional[FraudSignal]:
        # Exact-hash duplicate first — strongest signal possible
        for prior in self.records:
            if prior.content_hash == content_hash:
                return FraudSignal(
                    source=SignalSource.DUPLICATE,
                    code="DUP_EXACT_FILE_HASH",
                    description=(
                        f"This invoice's SHA-256 hash matches a previously-seen "
                        f"submission (invoice_id={prior.invoice_id}). Identical "
                        f"file resubmitted."
                    ),
                    weight=0.95,
                    confidence=1.0,
                    evidence={"matched_invoice_id": prior.invoice_id},
                )

        # Fuzzy match — only meaningful if we extracted at least a vendor name
        if not fields.vendor_name:
            return None

        best_match: Optional[tuple[InvoiceRecord, int, bool, bool]] = None
        for prior in self.records:
            if not prior.fields.vendor_name:
                continue
            sim = _fuzz_ratio(fields.vendor_name, prior.fields.vendor_name)
            if sim < self.vendor_threshold:
                continue
            amount_close = self._amounts_close(fields.total, prior.fields.total)
            date_close = self._dates_close(fields.invoice_date, prior.fields.invoice_date)
            # Amount-match is the strong signal — required for a flag. Date
            # alone produces too many false positives because legitimate
            # recurring (monthly, weekly) invoices share both vendor and
            # date-window with prior records.
            #
            # If amount matches: flag, with date as a confidence booster.
            # If amount doesn't match but date does: not enough evidence
            # — return no signal.
            if not amount_close:
                continue
            if best_match is None or sim > best_match[1]:
                best_match = (prior, sim, amount_close, date_close)

        if best_match is None:
            return None

        prior, sim, amount_close, date_close = best_match
        reasons: list[str] = [f"vendor name {sim}% similar",
                              "totals match to within $0.01"]
        if date_close:
            reasons.append(
                f"invoice dates within {self.date_window_days} days"
            )

        # Confidence scales with whether the date signal also corroborates.
        confidence = 0.80 + (0.10 if date_close else 0.0)

        return FraudSignal(
            source=SignalSource.DUPLICATE,
            code="DUP_FUZZY_MATCH",
            description=(
                f"Likely duplicate of invoice_id={prior.invoice_id} — "
                + ", ".join(reasons) + "."
            ),
            weight=0.85,
            confidence=min(confidence, 0.95),
            evidence={
                "matched_invoice_id": prior.invoice_id,
                "vendor_similarity_pct": sim,
                "amount_match": amount_close,
                "date_match": date_close,
            },
        )

    def _amounts_close(self, a: Optional[float], b: Optional[float]) -> bool:
        if a is None or b is None:
            return False
        return abs(a - b) <= self.amount_tolerance_cents / 100.0

    def _dates_close(self, a: Optional[Date], b: Optional[Date]) -> bool:
        if a is None or b is None:
            return False
        return abs((a - b).days) <= self.date_window_days


def check(repo: DuplicateRepository, content_hash: str,
          fields: ExtractedFields) -> Optional[FraudSignal]:
    """Convenience entry point matching the other detector modules' shape."""
    return repo.check(content_hash, fields)
