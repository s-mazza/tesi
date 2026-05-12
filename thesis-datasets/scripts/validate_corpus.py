#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "processed" / "canonical.jsonl"
REPORTS_DIR = ROOT / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.md"

REQUIRED_FIELDS = {
    "id",
    "block",
    "source",
    "split",
    "input_text",
    "paired_text",
    "label",
    "phenomenon",
    "category",
    "pair_id",
    "metadata",
}

VALID_BLOCKS = {"A", "B", "C"}
VALID_SPLITS = {"train", "validation", "test"}
VALID_LABELS = {"standard", "positive", "negative", "counterfactual", "commonsense_corrected"}
VALID_PHENOMENA = {"standard", "negation", "commonsense_violation"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def token_length(text: str) -> int:
    return len(text.split())


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return float(ordered[idx])


def validate(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ids = Counter(row.get("id") for row in rows)
    duplicate_ids = [row_id for row_id, count in ids.items() if count > 1]
    if duplicate_ids:
        errors.append(f"Duplicate ids: {duplicate_ids[:10]}")

    texts = Counter(row.get("input_text") for row in rows)
    duplicate_texts = [text for text, count in texts.items() if count > 1]
    if duplicate_texts:
        warnings.append(f"Duplicate input_text values: {len(duplicate_texts)}")

    for row in rows:
        line = row.get("_line_no")
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"line {line}: missing fields {sorted(missing)}")
        if row.get("block") not in VALID_BLOCKS:
            errors.append(f"line {line}: invalid block {row.get('block')!r}")
        if row.get("split") not in VALID_SPLITS:
            errors.append(f"line {line}: invalid split {row.get('split')!r}")
        if row.get("label") not in VALID_LABELS:
            errors.append(f"line {line}: invalid label {row.get('label')!r}")
        if row.get("phenomenon") not in VALID_PHENOMENA:
            errors.append(f"line {line}: invalid phenomenon {row.get('phenomenon')!r}")
        if not isinstance(row.get("input_text"), str) or not row.get("input_text", "").strip():
            errors.append(f"line {line}: empty input_text")
        if row.get("label") != "standard" and not row.get("paired_text"):
            errors.append(f"line {line}: non-standard row missing paired_text")
        if row.get("block") == "A" and row.get("phenomenon") != "standard":
            errors.append(f"line {line}: Block A must be standard")
        if row.get("block") == "B" and row.get("phenomenon") != "negation":
            errors.append(f"line {line}: Block B must be negation")
        if row.get("block") == "C" and row.get("phenomenon") != "commonsense_violation":
            errors.append(f"line {line}: Block C must be commonsense_violation")

    labels_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_pair[row["pair_id"]].add(row["label"])

    for pair_id, labels in labels_by_pair.items():
        if labels == {"standard"}:
            continue
        if labels == {"positive", "negative"}:
            continue
        if labels == {"counterfactual", "commonsense_corrected"}:
            continue
        errors.append(f"pair {pair_id}: unexpected label set {sorted(labels)}")

    synthetic_pairs = {
        row["pair_id"]
        for row in rows
        if row["source"] == "synthetic_commonsense" and row["label"] == "counterfactual"
    }
    if len(synthetic_pairs) != 300:
        errors.append(f"synthetic_commonsense expected 300 pairs, found {len(synthetic_pairs)}")

    synthetic_by_category = Counter(
        row["category"]
        for row in rows
        if row["source"] == "synthetic_commonsense" and row["label"] == "counterfactual"
    )
    if set(synthetic_by_category.values()) != {60}:
        errors.append(f"synthetic categories are not balanced at 60 each: {dict(synthetic_by_category)}")

    return errors, warnings


def count_table(rows: list[dict[str, Any]], field: str) -> str:
    counts = Counter(str(row.get(field)) for row in rows)
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counts.items()))


def write_report(rows: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lengths = [token_length(row["input_text"]) for row in rows]
    pair_count = len({row["pair_id"] for row in rows})
    sample_rows = [
        {k: row[k] for k in ("id", "block", "source", "label", "input_text", "paired_text")}
        for row in rows[:5]
    ]
    lines = [
        "# Corpus Validation Report",
        "",
        f"Canonical path: `{CANONICAL_PATH}`",
        f"Rows: `{len(rows)}`",
        f"Pairs/groups: `{pair_count}`",
        f"Validation status: `{'PASS' if not errors else 'FAIL'}`",
        "",
        "## Counts By Block",
        "",
        count_table(rows, "block"),
        "",
        "## Counts By Source",
        "",
        count_table(rows, "source"),
        "",
        "## Counts By Split",
        "",
        count_table(rows, "split"),
        "",
        "## Counts By Label",
        "",
        count_table(rows, "label"),
        "",
        "## Token Lengths",
        "",
        f"- mean whitespace tokens: `{mean(lengths):.2f}`",
        f"- p50: `{percentile(lengths, 0.50):.0f}`",
        f"- p90: `{percentile(lengths, 0.90):.0f}`",
        f"- max: `{max(lengths) if lengths else 0}`",
        "",
        "## Sample Rows",
        "",
        "```json",
        json.dumps(sample_rows, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None.")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("- None.")
    VALIDATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical thesis dataset.")
    parser.add_argument("--input", type=Path, default=CANONICAL_PATH)
    args = parser.parse_args()
    rows = load_rows(args.input)
    errors, warnings = validate(rows)
    write_report(rows, errors, warnings)
    print(f"Validation status: {'PASS' if not errors else 'FAIL'}")
    print(f"Wrote report to {VALIDATION_REPORT_PATH}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
