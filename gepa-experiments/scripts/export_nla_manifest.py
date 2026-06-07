#!/usr/bin/env python3
"""Export dataset examples as prompts for Qwen2.5 NLA activation extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geval_gepa.prompts import seed_instructions
from geval_gepa.tasks import get_task, split_examples


def render_prompt(*, instructions: str, source_text: str, fact: str, candidate_output: str) -> str:
    return "\n\n".join(
        [
            instructions,
            "Source/context:",
            source_text,
            "Reference/fact:",
            fact or "_nofact",
            "Candidate output:",
            candidate_output,
            "Evaluation form:",
            "Rationale:",
            "Score:",
        ]
    )


def build_manifest(args: argparse.Namespace) -> list[dict[str, object]]:
    task = get_task(args.dataset)
    rows = task.load(args.data_source or None, args.dimension or task.default_dimension)
    train, val, test = split_examples(
        rows,
        train_groups=args.train_groups,
        val_groups=args.val_groups,
        test_groups=args.test_groups,
        seed=args.seed,
    )
    split_rows = {"train": train, "validation": val, "test": test}[args.split]
    if not split_rows:
        return []
    min_score, max_score = split_rows[0].min_score, split_rows[0].max_score
    instructions = seed_instructions(
        dataset=task.dataset,
        dimension=split_rows[0].dimension,
        min_score=min_score,
        max_score=max_score,
    )
    return [
        {
            "dataset": row.dataset,
            "dimension": row.dimension,
            "example_id": row.example_id,
            "group_id": row.group_id,
            "system_id": row.system_id,
            "human_score": row.human_score,
            "min_score": row.min_score,
            "max_score": row.max_score,
            "source_text": row.source_text,
            "candidate_output": row.candidate_output,
            "reference": row.reference,
            "fact": row.fact,
            "prompt_source": "geval_gepa_seed_prompt",
            "token_policy": args.token_policy,
            "prompt": render_prompt(
                instructions=instructions,
                source_text=row.source_text,
                fact=row.fact or row.reference or "_nofact",
                candidate_output=row.candidate_output,
            ),
        }
        for row in split_rows[: args.limit]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dimension", default="")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--train-groups", type=int, default=40)
    parser.add_argument("--val-groups", type=int, default=10)
    parser.add_argument("--test-groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--token-policy", default="semantic_multi")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} NLA manifest rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
