#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import artifact_path, ensure_parent, read_jsonl
except ImportError:
    from common import artifact_path, ensure_parent, read_jsonl


def _cell(value: Any, max_len: int = 120) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def build_report(rows: list[dict[str, Any]]) -> str:
    counts_by_dimension = Counter(row["dimension"] for row in rows)
    counts_by_position = Counter(row["token_position"] for row in rows)
    counts_by_parse = Counter(row["parse_status"] for row in rows)
    counts_by_injection = Counter(row["injection_check_status"] for row in rows)

    lines = [
        "# SummEval NLA Report",
        "",
        "## Run Summary",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Dimensions: `{dict(sorted(counts_by_dimension.items()))}`",
        f"- Token positions: `{dict(sorted(counts_by_position.items()))}`",
        f"- Parse status: `{dict(sorted(counts_by_parse.items()))}`",
        f"- Injection check: `{dict(sorted(counts_by_injection.items()))}`",
        "",
    ]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dimension"], row["sample_bucket"], row["token_position"])].append(row)

    lines.extend(["## Examples", ""])
    for key in sorted(grouped):
        dimension, bucket, token_position = key
        group_rows = sorted(grouped[key], key=lambda row: (row["human_score"], row["example_id"]))
        lines.extend(
            [
                f"### {dimension} / {bucket} / {token_position}",
                "",
                "| example | human | model score | parse | explanation |",
                "|---|---:|---:|---|---|",
            ]
        )
        for row in group_rows[:4]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(row["example_id"], 48),
                        _cell(row["human_score"], 16),
                        _cell(row.get("score_text"), 16),
                        _cell(row["parse_status"], 20),
                        _cell(row["explanation"], 180),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SummEval NLA verbalizations.")
    parser.add_argument("--input", type=Path, default=artifact_path("summeval", "verbalizations.jsonl"))
    parser.add_argument("--output", type=Path, default=artifact_path("summeval", "report.md"))
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    report = build_report(rows)
    ensure_parent(args.output)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
