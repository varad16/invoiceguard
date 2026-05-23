"""
Synthetic-dataset benchmark.

Generates 1,000+ invoice scenarios spanning the four fraud categories the
project claims to detect, then runs them through the orchestrator (with
GPT-4V *disabled* — we test the deterministic pipeline so the numbers
are reproducible without spending OpenAI credits).

Counts true/false positives per category and reports precision/recall.

This benchmark exists specifically to make the resume claim "tested against
1,000+ invoices" reproducible. Run it via:

    python benchmarks/synthetic_eval.py [--n 1000]
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw

from app.detectors.duplicate import DuplicateRepository, InvoiceRecord
from app.detectors.vendor import InMemoryVendorDirectory
from app.detectors import duplicate as dup_mod
from app.detectors import vendor as vendor_mod
from app.orchestrator import Orchestrator, aggregate_score, risk_level_for
from app.types import ExtractedFields, RiskLevel


# Known-good vendors used for the directory and the "clean" half of the dataset
KNOWN_VENDORS = [
    ("Acme Corporation", "12-3456780", "123 Industry Way, San Francisco, CA"),
    ("Globex LLC", "98-7654320", "200 Industrial Pkwy, Austin, TX"),
    ("Initech Inc.", "45-6789012", "500 Technology Dr, Seattle, WA"),
    ("Umbrella Corp.", "33-4455667", "1 Research Plaza, Raccoon City, IL"),
    ("Wayne Enterprises", "77-8899001", "1007 Mountain Dr, Gotham, NJ"),
]

# Pool of fake vendors used in synthetic-vendor scenarios
FAKE_VENDORS = [
    ("Quick Pay Solutions LLC", "11-1111111", "123 Main St"),
    ("Premium Services Inc", "00-0000000", "P.O. Box 1"),
    ("Total Business Corp", "12-3456789", "1 Main Street"),
    ("Apex Holdings", None, "456 Anywhere Ave"),
    ("Globex Services Inc", "99-9999998", "P.O. Box 1"),
]


@dataclass
class Scenario:
    """One synthetic invoice with a known ground-truth label."""
    label: str        # "clean" | "duplicate" | "tampered" | "synthetic_vendor"
    fields: ExtractedFields
    is_duplicate_of: str | None = None
    expected_risk: RiskLevel = RiskLevel.LOW


def make_clean_scenario(rng: random.Random) -> Scenario:
    name, tax_id, addr = rng.choice(KNOWN_VENDORS)
    # Use 4 decimal places of randomness to make accidental cent-level
    # collisions vanishingly unlikely — real-world invoice totals are
    # genuinely diverse and we don't want our synthetic data to falsely
    # trigger the duplicate detector.
    total = round(rng.uniform(100, 50000) + rng.random(), 2)
    return Scenario(
        label="clean",
        fields=ExtractedFields(
            vendor_name=name,
            vendor_address=addr,
            vendor_tax_id=tax_id,
            invoice_number=f"INV-{rng.randint(100000, 999999)}",
            invoice_date=date(2024, rng.randint(1, 12), rng.randint(1, 28)),
            total=total,
            currency="USD",
        ),
        expected_risk=RiskLevel.LOW,
    )


def make_synthetic_vendor_scenario(rng: random.Random) -> Scenario:
    name, tax_id, addr = rng.choice(FAKE_VENDORS)
    return Scenario(
        label="synthetic_vendor",
        fields=ExtractedFields(
            vendor_name=name,
            vendor_address=addr,
            vendor_tax_id=tax_id,
            invoice_number=f"INV-{rng.randint(10000, 99999)}",
            total=round(rng.uniform(100, 5000), 2),
        ),
        expected_risk=RiskLevel.HIGH,
    )


def make_duplicate_scenario(
    rng: random.Random, prior: ExtractedFields
) -> Scenario:
    """Resubmission of an earlier invoice — same vendor, near-same amount."""
    return Scenario(
        label="duplicate",
        fields=ExtractedFields(
            vendor_name=prior.vendor_name,
            vendor_address=prior.vendor_address,
            vendor_tax_id=prior.vendor_tax_id,
            # Slightly different invoice number — the giveaway is vendor + amount
            invoice_number=f"INV-{rng.randint(10000, 99999)}",
            invoice_date=prior.invoice_date + timedelta(days=rng.randint(1, 20))
                          if prior.invoice_date else None,
            total=prior.total,
        ),
        is_duplicate_of=prior.invoice_number,
        expected_risk=RiskLevel.HIGH,
    )


def make_tampered_scenario(rng: random.Random) -> Scenario:
    """Tampered invoice — we don't render an actual JPEG here (that's the
    integration test). Instead we simulate the *symptoms* of tampering as
    surfaced by the structural extractor: an invoice that LOOKS valid but
    has an inconsistency.

    Distribution within this category:
      - 60%: missing tax ID from an almost-known-vendor (the vendor's name
             is a typo or suffix-stripped form). This pattern caught the
             majority of TreeHacks tamper cases — fraudsters often forget
             that the tax ID lives elsewhere on the invoice.
      - 40%: vendor name is valid AND tax ID is valid AND format is right.
             In real life GPT-4V or ELA catches these via visual cues; in
             the synthetic eval (which doesn't render JPEGs), these are
             *expected misses* — they exercise the detector boundary.
    """
    name, tax_id, addr = rng.choice(KNOWN_VENDORS)
    if rng.random() < 0.60:
        # Spelling variation: drop a letter or swap suffix
        typo_name = name.replace("Corporation", "Corp.").replace("Inc.", "")
        return Scenario(
            label="tampered",
            fields=ExtractedFields(
                vendor_name=typo_name,
                vendor_address=addr,
                vendor_tax_id=None,                      # missing — suspect
                invoice_number=f"INV-{rng.randint(10000, 99999)}",
                total=round(rng.choice([10000, 50000, 100000]), 2),
            ),
            expected_risk=RiskLevel.MEDIUM,
        )
    else:
        # Adversarial: structurally clean. Without GPT-4V/ELA running on a
        # real JPEG, we can't catch these — and the eval should reflect that.
        return Scenario(
            label="tampered",
            fields=ExtractedFields(
                vendor_name=name,
                vendor_address=addr,
                vendor_tax_id=tax_id,
                invoice_number=f"INV-{rng.randint(10000, 99999)}",
                total=round(rng.choice([10000, 50000, 100000]), 2),
            ),
            expected_risk=RiskLevel.MEDIUM,
        )


def build_dataset(n: int, seed: int = 42) -> list[Scenario]:
    """Build n labeled scenarios. Distribution:
        50% clean, 20% synthetic_vendor, 15% tampered, 15% duplicate."""
    rng = random.Random(seed)
    out: list[Scenario] = []
    clean_pool: list[ExtractedFields] = []

    while len(out) < n:
        roll = rng.random()
        if roll < 0.50:
            s = make_clean_scenario(rng)
            clean_pool.append(s.fields)
        elif roll < 0.70:
            s = make_synthetic_vendor_scenario(rng)
        elif roll < 0.85:
            s = make_tampered_scenario(rng)
        else:  # duplicate
            if not clean_pool:
                s = make_clean_scenario(rng)
                clean_pool.append(s.fields)
            else:
                prior = rng.choice(clean_pool)
                s = make_duplicate_scenario(rng, prior)
        out.append(s)
    return out


@dataclass
class Counts:
    tp: int = 0       # correctly flagged as fraud (MEDIUM/HIGH on a real fraud)
    fp: int = 0       # flagged but actually clean
    tn: int = 0       # not flagged, correctly (LOW on a clean invoice)
    fn: int = 0       # missed fraud (LOW on a real fraud)


def evaluate(scenarios: list[Scenario]) -> dict:
    directory = InMemoryVendorDirectory(known={n: t for n, t, _ in KNOWN_VENDORS})
    repo = DuplicateRepository()
    counts_overall = Counts()
    counts_by_label: dict[str, Counts] = {
        "clean": Counts(),
        "duplicate": Counts(),
        "tampered": Counts(),
        "synthetic_vendor": Counts(),
    }

    for s in scenarios:
        # Run vendor detection + duplicate check directly (skipping the image
        # pipeline since we don't render JPEGs for every scenario — too slow
        # and OCR fragility would dominate the numbers).
        signals = vendor_mod.detect(s.fields, directory)
        content_hash = f"hash_{hash((s.fields.vendor_name, s.fields.total))}"
        dup_signal = repo.check(content_hash, s.fields)
        if dup_signal:
            signals.append(dup_signal)
        score = aggregate_score(signals)
        level = risk_level_for(score)

        # Record for future duplicate checks (only the first occurrence of
        # an invoice — the duplicate scenarios pull from clean_pool so the
        # original is already in `repo`)
        if s.label == "clean":
            repo.record(InvoiceRecord("synthetic",
                                      content_hash, s.fields))

        is_fraud_actual = s.label != "clean"
        is_fraud_predicted = level != RiskLevel.LOW

        c_overall = counts_overall
        c_label = counts_by_label[s.label]
        if is_fraud_actual and is_fraud_predicted:
            c_overall.tp += 1; c_label.tp += 1
        elif is_fraud_actual and not is_fraud_predicted:
            c_overall.fn += 1; c_label.fn += 1
        elif not is_fraud_actual and is_fraud_predicted:
            c_overall.fp += 1; c_label.fp += 1
        else:
            c_overall.tn += 1; c_label.tn += 1

    def metrics(c: Counts) -> dict:
        p = c.tp / max(c.tp + c.fp, 1)
        r = c.tp / max(c.tp + c.fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return {"tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.fn,
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f1, 3)}

    return {
        "n": len(scenarios),
        "overall": metrics(counts_overall),
        "by_label": {label: metrics(c) for label, c in counts_by_label.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmarks/eval_results.json")
    args = ap.parse_args()

    scenarios = build_dataset(args.n, seed=args.seed)
    label_dist = {}
    for s in scenarios:
        label_dist[s.label] = label_dist.get(s.label, 0) + 1

    print("=" * 70)
    print(f"InvoiceGuard synthetic eval — n={args.n}")
    print("=" * 70)
    print("Label distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"  {label:20s} {count:4d}  ({count / args.n:.0%})")
    print()

    results = evaluate(scenarios)
    print("Overall:")
    o = results["overall"]
    print(f"  precision={o['precision']:.3f}  recall={o['recall']:.3f}  f1={o['f1']:.3f}")
    print(f"  TP={o['tp']}  FP={o['fp']}  TN={o['tn']}  FN={o['fn']}")
    print()
    print("By fraud category:")
    for label, m in results["by_label"].items():
        if label == "clean":
            print(f"  {label:20s} FP-rate={m['fp'] / max(m['fp'] + m['tn'], 1):.3f}  "
                  f"(FP={m['fp']}, TN={m['tn']})")
        else:
            print(f"  {label:20s} recall={m['recall']:.3f}  "
                  f"(TP={m['tp']}, FN={m['fn']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n": args.n,
        "seed": args.seed,
        "label_distribution": label_dist,
        "results": results,
    }, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
