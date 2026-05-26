#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

try:
    from .common import (
        DIMENSIONS,
        SUMMEVAL_DATA_URL,
        artifact_path,
        ensure_parent,
        sha256_text,
        short_hash,
        write_jsonl,
    )
except ImportError:
    from common import (
        DIMENSIONS,
        SUMMEVAL_DATA_URL,
        artifact_path,
        ensure_parent,
        sha256_text,
        short_hash,
        write_jsonl,
    )


PROMPT_TEMPLATES = {
    "coherence": """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Coherence:""",
    "consistency": """You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Penalize summaries that contain hallucinated facts.

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Consistency:""",
    "fluency": """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.


Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Fluency (1-3):""",
    "relevance": """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Penalize summaries which contain redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Relevance:""",
}


def download_summeval(path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    ensure_parent(path)
    with urllib.request.urlopen(SUMMEVAL_DATA_URL, timeout=60) as response:
        path.write_bytes(response.read())


def load_summeval(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}")
    return rows


def render_prompt(dimension: str, document: str, summary: str) -> str:
    template = PROMPT_TEMPLATES[dimension]
    return template.replace("{{Document}}", document).replace("{{Summary}}", summary)


def _row_key(row: dict[str, Any]) -> str:
    return f"{row.get('doc_id', '')}::{row.get('system_id', '')}"


def _score(row: dict[str, Any], dimension: str) -> float:
    return float(row["scores"][dimension])


def _pick_diverse(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        if doc_id in seen_docs:
            continue
        picked.append(row)
        seen_docs.add(doc_id)
        if len(picked) == n:
            return picked
    for row in rows:
        if _row_key(row) in {_row_key(p) for p in picked}:
            continue
        picked.append(row)
        if len(picked) == n:
            return picked
    return picked


def select_examples(
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
    stratify_dimension: str,
) -> list[tuple[str, dict[str, Any]]]:
    if max_examples < 2 or max_examples % 2 != 0:
        raise ValueError("--max-examples must be an even integer >= 2")
    n_per_bucket = max_examples // 2
    sorted_rows = sorted(rows, key=_row_key)
    low = sorted(
        sorted_rows,
        key=lambda row: (_score(row, stratify_dimension), _score(row, "overall"), _row_key(row)),
    )
    high = sorted(
        sorted_rows,
        key=lambda row: (-_score(row, stratify_dimension), -_score(row, "overall"), _row_key(row)),
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    selected.extend((f"low_{stratify_dimension}", row) for row in _pick_diverse(low, n_per_bucket))
    selected.extend((f"high_{stratify_dimension}", row) for row in _pick_diverse(high, n_per_bucket))
    if len(selected) != max_examples:
        raise ValueError(f"Could only select {len(selected)} examples from SummEval")
    return selected


def build_manifest(
    rows: list[dict[str, Any]],
    *,
    max_examples: int = 12,
    stratify_dimension: str = "consistency",
    source_sha256: str = "",
) -> list[dict[str, Any]]:
    selected = select_examples(
        rows,
        max_examples=max_examples,
        stratify_dimension=stratify_dimension,
    )
    manifest: list[dict[str, Any]] = []
    for rank, (bucket, row) in enumerate(selected):
        base_key = _row_key(row)
        base_id = f"summeval_{rank:02d}_{bucket}_{short_hash(base_key)}"
        for dimension in DIMENSIONS:
            prompt = render_prompt(dimension, row["source"], row["system_output"])
            prompt_hash = sha256_text(prompt)
            manifest.append(
                {
                    "example_id": f"{base_id}_{dimension}",
                    "base_example_id": base_id,
                    "dataset": "SummEval",
                    "source": "nlpyang/geval:data/summeval.json",
                    "source_sha256": source_sha256,
                    "doc_id": row["doc_id"],
                    "system_id": row["system_id"],
                    "dimension": dimension,
                    "human_score": float(row["scores"][dimension]),
                    "overall_score": float(row["scores"]["overall"]),
                    "sample_bucket": bucket,
                    "source_article": row["source"],
                    "reference_summary": row["reference"],
                    "candidate_summary": row["system_output"],
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "prompt_source": "geval_summeval_detailed_inline",
                }
            )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the SummEval manifest for NLA activation extraction.")
    parser.add_argument("--source-json", type=Path, default=artifact_path("summeval", "raw", "summeval.json"))
    parser.add_argument("--output", type=Path, default=artifact_path("summeval", "task_manifest.jsonl"))
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--stratify-dimension", choices=DIMENSIONS, default="consistency")
    parser.add_argument("--download", action="store_true", help="Download SummEval if --source-json is absent.")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if args.download or args.force_download or not args.source_json.exists():
        download_summeval(args.source_json, force=args.force_download)

    raw_bytes = args.source_json.read_bytes()
    rows = load_summeval(args.source_json)
    manifest = build_manifest(
        rows,
        max_examples=args.max_examples,
        stratify_dimension=args.stratify_dimension,
        source_sha256=sha256_text(raw_bytes.decode("utf-8")),
    )
    count = write_jsonl(args.output, manifest)
    print(f"Wrote {count} prompt rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
