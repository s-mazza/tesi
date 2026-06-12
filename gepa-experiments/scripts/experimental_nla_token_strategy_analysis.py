#!/usr/bin/env python3
"""Analyze isolated NLA token-selection strategies without changing the runner."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from export_nla_manifest import render_prompt
from geval_gepa.prompts import seed_instructions
from geval_gepa.tasks import get_task, split_examples


WORD_RE = re.compile(r"\b[\w'-]{4,}\b")
WEAK_TOKENS = {
    "from",
    "that",
    "this",
    "with",
    "there",
    "they",
    "their",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "about",
    "like",
    "just",
    "yeah",
    "hello",
    "said",
    "think",
    "recently",
}


@dataclass(frozen=True)
class TokenChoice:
    strategy: str
    example_id: str
    group_id: str
    field: str
    position: str
    token_text: str
    word_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="topical_chat")
    parser.add_argument("--dimension", default="engagingness")
    parser.add_argument("--data-source", default="gepa-experiments/cache/tc_usr_data.json")
    parser.add_argument("--split", choices=("train", "validation", "test", "gepa"), default="gepa")
    parser.add_argument("--train-groups", type=int, default=40)
    parser.add_argument("--val-groups", type=int, default=10)
    parser.add_argument("--test-groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_manifest_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    task = get_task(args.dataset)
    rows = task.load(args.data_source or None, args.dimension or task.default_dimension)
    train, val, test = split_examples(
        rows,
        train_groups=args.train_groups,
        val_groups=args.val_groups,
        test_groups=args.test_groups,
        seed=args.seed,
    )
    split_rows = {"train": train, "validation": val, "test": test, "gepa": train + val}[args.split]
    if args.limit is not None:
        split_rows = split_rows[: args.limit]
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
            "source_text": row.source_text,
            "candidate_output": row.candidate_output,
            "reference": row.reference,
            "fact": row.fact,
            "prompt": render_prompt(
                instructions=instructions,
                source_text=row.source_text,
                fact=row.fact or row.reference or "_nofact",
                candidate_output=row.candidate_output,
            ),
        }
        for row in split_rows
    ]


def strategies() -> dict[str, dict[str, Any]]:
    return {
        "current_fixed_6": {"budgets": {"candidate": 3, "source": 1, "reference": 2}},
        "candidate_only_6": {"budgets": {"candidate": 6, "source": 0, "reference": 0}},
        "candidate_source_6": {"budgets": {"candidate": 4, "source": 2, "reference": 0}},
        "candidate_reference_6": {"budgets": {"candidate": 4, "source": 0, "reference": 2}},
        "balanced_6": {"budgets": {"candidate": 2, "source": 2, "reference": 2}},
        "candidate_only_10": {"budgets": {"candidate": 10, "source": 0, "reference": 0}},
        "candidate_source_10": {"budgets": {"candidate": 7, "source": 3, "reference": 0}},
        "candidate_no_first_6": {"budgets": {"candidate": 6, "source": 0, "reference": 0}, "avoid_first": True},
        "candidate_content_6": {
            "budgets": {"candidate": 6, "source": 0, "reference": 0},
            "avoid_first": True,
            "filter_weak": True,
        },
        "candidate_content_10": {
            "budgets": {"candidate": 10, "source": 0, "reference": 0},
            "avoid_first": True,
            "filter_weak": True,
        },
        "candidate_source_content_8": {
            "budgets": {"candidate": 6, "source": 2, "reference": 0},
            "avoid_first": True,
            "filter_weak": True,
        },
        "hybrid_context_dedup_6": {
            "budgets": {"candidate": 4, "source": 1, "reference": 1},
            "avoid_first": True,
            "filter_weak": True,
            "dedupe_context_fields": {"source", "reference"},
        },
        "hybrid_context_dedup_8": {
            "budgets": {"candidate": 6, "source": 1, "reference": 1},
            "avoid_first": True,
            "filter_weak": True,
            "dedupe_context_fields": {"source", "reference"},
        },
    }


def choose_tokens(rows: list[dict[str, Any]]) -> list[TokenChoice]:
    choices: list[TokenChoice] = []
    seen_context_fields: set[tuple[str, str, str]] = set()
    for row in rows:
        fields = {
            "candidate": str(row.get("candidate_output") or ""),
            "source": str(row.get("source_text") or ""),
            "reference": str(row.get("fact") or row.get("reference") or ""),
        }
        for strategy_name, spec in strategies().items():
            budgets = spec["budgets"]
            for field, budget in budgets.items():
                if budget <= 0:
                    continue
                group_id = str(row["group_id"])
                dedupe_context_fields = set(spec.get("dedupe_context_fields", set()))
                if field in dedupe_context_fields:
                    context_key = (strategy_name, group_id, field)
                    if context_key in seen_context_fields:
                        continue
                text = fields[field]
                if not text or text == "_nofact":
                    continue
                if field in dedupe_context_fields:
                    seen_context_fields.add(context_key)
                for position, token_text, word_index in sample_words(
                    text,
                    budget,
                    avoid_first=bool(spec.get("avoid_first")),
                    filter_weak=bool(spec.get("filter_weak")),
                ):
                    choices.append(
                        TokenChoice(
                            strategy=strategy_name,
                            example_id=str(row["example_id"]),
                            group_id=group_id,
                            field=field,
                            position=position,
                            token_text=token_text,
                            word_index=word_index,
                        )
                    )
    return choices


def sample_words(text: str, limit: int, *, avoid_first: bool = False, filter_weak: bool = False) -> list[tuple[str, str, int]]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []
    indexes = [len(matches) // 2, len(matches) - 1]
    if not avoid_first:
        indexes.append(0)
    if limit > 3:
        step = max(1, len(matches) // limit)
        indexes.extend(range(0, len(matches), step))
    selected: list[int] = []
    for index in indexes:
        token = matches[index].group(0).lower()
        if filter_weak and token in WEAK_TOKENS:
            continue
        if avoid_first and index == 0 and len(matches) > 2:
            continue
        if index not in selected:
            selected.append(index)
        if len(selected) >= limit:
            break
    return [(label_for_index(index, matches), matches[index].group(0), index) for index in selected]


def label_for_index(index: int, matches: list[re.Match[str]]) -> str:
    if index == len(matches) // 2:
        return "middle"
    if index == len(matches) - 1:
        return "last"
    if index == 0:
        return "first"
    return f"content_{index}"


def summarize(choices: list[TokenChoice], examples: int) -> list[dict[str, Any]]:
    rows = []
    for strategy_name in strategies():
        selected = [choice for choice in choices if choice.strategy == strategy_name]
        token_keys = [(choice.field, choice.position, choice.token_text.lower()) for choice in selected]
        token_counts = Counter(token_keys)
        field_counts = Counter(choice.field for choice in selected)
        position_counts = Counter(f"{choice.field}_{choice.position}" for choice in selected)
        weak = sum(1 for choice in selected if choice.token_text.lower() in WEAK_TOKENS)
        duplicates = sum(count for count in token_counts.values() if count > 1)
        rows.append(
            {
                "strategy": strategy_name,
                "examples": examples,
                "selected_tokens": len(selected),
                "avg_tokens_per_example": len(selected) / examples if examples else 0,
                "unique_field_position_tokens": len(token_counts),
                "duplicate_token_rows": duplicates,
                "duplicate_token_row_pct": 100.0 * duplicates / len(selected) if selected else 0,
                "weak_token_rows": weak,
                "weak_token_pct": 100.0 * weak / len(selected) if selected else 0,
                "candidate_rows": field_counts.get("candidate", 0),
                "source_rows": field_counts.get("source", 0),
                "reference_rows": field_counts.get("reference", 0),
                "top_positions": json.dumps(position_counts.most_common(8)),
                "top_repeated_tokens": json.dumps(
                    [
                        {"count": count, "field": field, "position": position, "token": token}
                        for (field, position, token), count in token_counts.most_common(8)
                        if count > 1
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, choices: list[TokenChoice]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for choice in choices:
            handle.write(json.dumps(choice.__dict__, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experimental NLA Token Strategy Analysis",
        "",
        "This report compares token-selection strategies only. It does not change the GEPA runner or main NLA pipeline.",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- `{strategy}`: avg_tokens={avg_tokens_per_example:.2f}, "
            "candidate_rows={candidate_rows}, source_rows={source_rows}, reference_rows={reference_rows}, "
            "duplicate_pct={duplicate_token_row_pct:.2f}, weak_pct={weak_token_pct:.2f}".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision Use",
            "",
            "- Prefer strategies with high candidate coverage and low weak/duplicate token rates.",
            "- Do not merge any strategy into the main pipeline from this analysis alone.",
            "- Use this analysis only to choose isolated GEPA strategy jobs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_rows = build_manifest_rows(args)
    choices = choose_tokens(manifest_rows)
    summary_rows = summarize(choices, examples=len(manifest_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "token_strategy_summary.csv", summary_rows)
    write_jsonl(args.output_dir / "token_strategy_choices.jsonl", choices)
    write_markdown(args.output_dir / "token_strategy_report.md", summary_rows)
    print(f"Wrote token strategy analysis for {len(manifest_rows)} examples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
