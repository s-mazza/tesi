#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "processed" / "canonical.jsonl"
EXPORT_DIR = ROOT / "processed" / "nla"
REPORT_PATH = ROOT / "reports" / "nla_export_report.md"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export canonical corpus as NLA AV manifest.")
    parser.add_argument("--input", type=Path, default=CANONICAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--limit-per-block", type=int, default=50)
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for row in rows:
        block = row["block"]
        if counts[block] >= args.limit_per_block:
            continue
        selected.append(row)
        counts[block] += 1

    manifest = [
        {
            "canonical_id": row["id"],
            "pair_id": row["pair_id"],
            "block": row["block"],
            "source": row["source"],
            "split": row["split"],
            "label": row["label"],
            "phenomenon": row["phenomenon"],
            "category": row["category"],
            "text": row["input_text"],
            "paired_text": row["paired_text"],
            "activation_target": {
                "model_id": args.model_id,
                "layer": args.layer,
                "representation": "residual_stream",
                "token_position": "last_token",
            },
        }
        for row in selected
    ]
    manifest_path = args.output_dir / "activation_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# NLA AV Export Report",
                "",
                f"Manifest: `{manifest_path}`",
                f"Rows exported: `{len(manifest)}`",
                f"Model target: `{args.model_id}`",
                f"Layer target: `{args.layer}`",
                "",
                "This manifest is ready for the activation-extraction stage of",
                "Anthropic-style Natural Language Autoencoders. It deliberately",
                "keeps canonical ids, pair ids, and original/corrected pair fields",
                "so AV verbalizations can be scored for semantic flips.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote NLA manifest to {manifest_path}")
    print(f"Wrote report to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
