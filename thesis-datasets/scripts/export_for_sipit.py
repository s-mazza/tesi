#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "processed" / "canonical.jsonl"
EXPORT_DIR = ROOT / "processed" / "sipit"
REPORT_PATH = ROOT / "reports" / "sipit_export_report.md"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def simple_token_ids(text: str, max_tokens: int) -> list[int]:
    """Deterministic placeholder ids for CPU-side smoke tests.

    The real SIPIT run should tokenize with the target model tokenizer. This
    export keeps canonical ids and texts ready while avoiding a GPU/model
    dependency during the pre-call dataset stage.
    """
    tokens = text.split()[:max_tokens]
    ids = [
        int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 50000
        for token in tokens
    ]
    if len(ids) < max_tokens:
        ids.extend([0] * (max_tokens - len(ids)))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Export canonical corpus for SIPIT smoke runs.")
    parser.add_argument("--input", type=Path, default=CANONICAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--max-tokens", type=int, default=32)
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

    tokenized = {
        "dataset_name": "thesis_canonical_smoke",
        "prompt_tokens": args.max_tokens,
        "token_ids": [simple_token_ids(row["input_text"], args.max_tokens) for row in selected],
        "start_ids": [0 for _ in selected],
        "sample_ids": list(range(len(selected))),
    }
    (args.output_dir / "thesis_canonical_smoke_tokenized.json").write_text(
        json.dumps(tokenized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sidecar = [
        {
            "sample_id": idx,
            "canonical_id": row["id"],
            "block": row["block"],
            "source": row["source"],
            "label": row["label"],
            "input_text": row["input_text"],
            "paired_text": row["paired_text"],
        }
        for idx, row in enumerate(selected)
    ]
    (args.output_dir / "id_sidecar.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# SIPIT Export Report",
                "",
                f"Export dir: `{args.output_dir}`",
                f"Rows exported: `{len(selected)}`",
                f"Rows per block limit: `{args.limit_per_block}`",
                f"Prompt token length: `{args.max_tokens}`",
                "",
                "This pre-call export is a CPU-side smoke manifest. For real SIPIT",
                "experiments, tokenize these same canonical rows with the target",
                "Hugging Face tokenizer and feed them through SIPIT's native",
                "`DatasetCollection` path.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote SIPIT export to {args.output_dir}")
    print(f"Wrote report to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
