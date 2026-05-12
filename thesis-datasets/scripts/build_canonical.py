#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"
REPORTS_DIR = ROOT / "reports"
SOURCES_DIR = ROOT / "sources"
CANONICAL_PATH = PROCESSED_DIR / "canonical.jsonl"
BUILD_REPORT_PATH = REPORTS_DIR / "build_report.md"
BLOCK_A_STANDARD_PATH = SOURCES_DIR / "block_a_standard_sentences.json"

os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "cache" / "huggingface" / "datasets"))

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - handled at runtime.
    load_dataset = None  # type: ignore


SEMEVAL_BASE = (
    "https://raw.githubusercontent.com/"
    "wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation/"
    "master/ALL%20data"
)


@dataclass(frozen=True)
class CanonicalRow:
    id: str
    block: str
    source: str
    split: str
    input_text: str
    paired_text: str | None
    label: str
    phenomenon: str
    category: str | None
    pair_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def load_json_list(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{path} must contain a JSON list of strings.")
    return [normalize_text(item) for item in data if normalize_text(item)]


def add_pair_rows(
    rows: list[CanonicalRow],
    *,
    block: str,
    source: str,
    split: str,
    pair_id: str,
    first_text: str,
    second_text: str,
    first_label: str,
    second_label: str,
    phenomenon: str,
    category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    metadata = metadata or {}
    first_text = normalize_text(first_text)
    second_text = normalize_text(second_text)
    if not first_text or not second_text:
        return
    rows.append(
        CanonicalRow(
            id=f"{block}_{source}_{split}_{pair_id}_{first_label}",
            block=block,
            source=source,
            split=split,
            input_text=first_text,
            paired_text=second_text,
            label=first_label,
            phenomenon=phenomenon,
            category=category,
            pair_id=f"{source}_{split}_{pair_id}",
            metadata=metadata,
        )
    )
    rows.append(
        CanonicalRow(
            id=f"{block}_{source}_{split}_{pair_id}_{second_label}",
            block=block,
            source=source,
            split=split,
            input_text=second_text,
            paired_text=first_text,
            label=second_label,
            phenomenon=phenomenon,
            category=category,
            pair_id=f"{source}_{split}_{pair_id}",
            metadata=metadata,
        )
    )


def build_standard_rows(rows: list[CanonicalRow]) -> None:
    prompts = load_json_list(BLOCK_A_STANDARD_PATH)
    for idx, text in enumerate(prompts):
        split = "test" if idx % 5 == 0 else "train"
        rows.append(
            CanonicalRow(
                id=f"A_manual_{split}_{idx:04d}_standard",
                block="A",
                source="manual",
                split=split,
                input_text=text,
                paired_text=None,
                label="standard",
                phenomenon="standard",
                category=None,
                pair_id=f"manual_{split}_{idx:04d}",
                metadata={"construction": "controlled_short_sentence"},
            )
        )


def require_datasets() -> None:
    if load_dataset is None:
        raise RuntimeError(
            "Python package `datasets` is required. Install from "
            "requirements-thesis-datasets.txt."
        )


def build_jina_negation_rows(rows: list[CanonicalRow], per_split: int) -> None:
    require_datasets()
    for split in ("train", "test"):
        ds = load_dataset("jinaai/negation-dataset", split=split, streaming=True)
        for idx, ex in enumerate(ds):
            if idx >= per_split:
                break
            add_pair_rows(
                rows,
                block="B",
                source="jina_negation",
                split=split,
                pair_id=f"{idx:05d}",
                first_text=ex["entailment"],
                second_text=ex["negative"],
                first_label="positive",
                second_label="negative",
                phenomenon="negation",
                metadata={
                    "anchor": normalize_text(ex.get("anchor")),
                    "dataset": "jinaai/negation-dataset",
                },
            )


def build_this_is_not_rows(rows: list[CanonicalRow], max_pairs: int) -> None:
    require_datasets()
    ds = load_dataset("HiTZ/This-is-not-a-dataset", split="test", streaming=True)
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"positive": None, "negative": None})

    for ex in ds:
        test_id = str(ex["test_id"])
        if ex.get("isDistractor"):
            continue
        sentence = normalize_text(ex.get("sentence"))
        if not sentence:
            continue
        bucket = "positive" if bool(ex.get("label")) else "negative"
        if grouped[test_id][bucket] is None:
            grouped[test_id][bucket] = dict(ex)
        if sum(1 for item in grouped.values() if item["positive"] and item["negative"]) >= max_pairs:
            break

    produced = 0
    for test_id in sorted(grouped, key=lambda x: int(x)):
        pair = grouped[test_id]
        pos = pair["positive"]
        neg = pair["negative"]
        if not pos or not neg:
            continue
        add_pair_rows(
            rows,
            block="B",
            source="this_is_not",
            split="test",
            pair_id=test_id,
            first_text=pos["sentence"],
            second_text=neg["sentence"],
            first_label="positive",
            second_label="negative",
            phenomenon="negation",
            metadata={
                "pattern_id": pos.get("pattern_id"),
                "pattern": pos.get("pattern"),
                "negation_type": neg.get("negation_type"),
                "semantic_type": neg.get("semantic_type"),
                "syntactic_scope": neg.get("syntactic_scope"),
                "dataset": "HiTZ/This-is-not-a-dataset",
            },
        )
        produced += 1
        if produced >= max_pairs:
            break


def fetch_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        output_path.write_bytes(response.read())


def build_semeval_rows(rows: list[CanonicalRow], per_split: int) -> None:
    split_map = {"train": "train.csv", "validation": "dev.csv", "test": "test.csv"}
    for split, filename in split_map.items():
        raw_path = RAW_DIR / "semeval2020_task4" / filename
        fetch_url(f"{SEMEVAL_BASE}/{filename}", raw_path)
        with raw_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, ex in enumerate(reader):
                if idx >= per_split:
                    break
                add_pair_rows(
                    rows,
                    block="C",
                    source="semeval2020_task4",
                    split=split,
                    pair_id=f"{idx:05d}",
                    first_text=ex["Incorrect Statement"],
                    second_text=ex["Correct Statement"],
                    first_label="counterfactual",
                    second_label="commonsense_corrected",
                    phenomenon="commonsense_violation",
                    category="commonsense",
                    metadata={
                        "right_reasons": [
                            normalize_text(ex.get("Right Reason1")),
                            normalize_text(ex.get("Right Reason2")),
                            normalize_text(ex.get("Right Reason3")),
                        ],
                        "dataset": "SemEval 2020 Task 4 ComVE",
                    },
                )


SYNTHETIC_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "physics": [
        ("The fire cooled the water.", "The fire heated the water."),
        ("The stone floated upward after being dropped.", "The stone fell downward after being dropped."),
        ("The ice cube became warmer in the freezer.", "The ice cube became colder in the freezer."),
        ("The shadow appeared brighter than the lamp.", "The shadow appeared darker than the lamp."),
        ("The metal spoon froze in boiling water.", "The metal spoon heated in boiling water."),
        ("The balloon sank quickly through the air.", "The balloon floated through the air."),
        ("The magnet pushed the iron nail away.", "The magnet pulled the iron nail closer."),
        ("The candle flame made the room colder.", "The candle flame made the room warmer."),
        ("The glass expanded after it shattered.", "The glass broke into smaller pieces after it shattered."),
        ("The smoke flowed downward into the fire.", "The smoke rose upward from the fire."),
    ],
    "biology": [
        ("The fish walked across the desert.", "The fish swam through the water."),
        ("The tree drank milk from a bowl.", "The tree absorbed water through its roots."),
        ("The bird grew gills before flying.", "The bird used wings to fly."),
        ("The cactus needed daily ocean tides to survive.", "The cactus needed little water to survive."),
        ("The dog photosynthesized in the sunlight.", "The plant photosynthesized in the sunlight."),
        ("The mushroom chased a mouse through the field.", "The cat chased a mouse through the field."),
        ("The newborn elephant hatched from an egg.", "The newborn elephant was born alive."),
        ("The salmon climbed a tree to lay eggs.", "The salmon swam upstream to lay eggs."),
        ("The rose smelled the gardener.", "The gardener smelled the rose."),
        ("The bee pollinated the moon at night.", "The bee pollinated the flower during the day."),
    ],
    "causality": [
        ("The alarm rang because everyone woke up.", "Everyone woke up because the alarm rang."),
        ("The window broke before the ball hit it.", "The window broke after the ball hit it."),
        ("The cup spilled because the table became dry.", "The table became wet because the cup spilled."),
        ("The car stopped because the engine started.", "The car moved because the engine started."),
        ("The plants died because they received enough water.", "The plants survived because they received enough water."),
        ("The room became dark because the light was switched on.", "The room became bright because the light was switched on."),
        ("The bread toasted because it was placed in the fridge.", "The bread toasted because it was placed in the toaster."),
        ("The door opened because it was locked tighter.", "The door stayed closed because it was locked tighter."),
        ("The message arrived before it was sent.", "The message arrived after it was sent."),
        ("The shirt dried because it was soaked in rain.", "The shirt became wet because it was soaked in rain."),
    ],
    "time": [
        ("The child graduated before starting school.", "The child graduated after starting school."),
        ("The morning came after midnight and before evening.", "The morning came after night and before afternoon."),
        ("The cake was eaten before it was baked.", "The cake was eaten after it was baked."),
        ("The train arrived yesterday after leaving tomorrow.", "The train arrived after it left."),
        ("The seed produced fruit before it sprouted.", "The seed produced fruit after it sprouted."),
        ("The letter was answered before it was written.", "The letter was answered after it was written."),
        ("The patient recovered before getting sick.", "The patient recovered after getting sick."),
        ("The exam was graded before students took it.", "The exam was graded after students took it."),
        ("The sun set before it rose on the same day.", "The sun set after it rose on the same day."),
        ("The baby spoke fluently before being born.", "The baby spoke after being born and growing older."),
    ],
    "quantity": [
        ("The basket held fewer apples after adding ten apples.", "The basket held more apples after adding ten apples."),
        ("Half of the cake was larger than the whole cake.", "Half of the cake was smaller than the whole cake."),
        ("Two coins plus two coins made one coin.", "Two coins plus two coins made four coins."),
        ("The empty bottle contained more water than the full bottle.", "The full bottle contained more water than the empty bottle."),
        ("A kilometer was shorter than a meter.", "A kilometer was longer than a meter."),
        ("The crowd shrank when more people entered.", "The crowd grew when more people entered."),
        ("The bill decreased after adding a new charge.", "The bill increased after adding a new charge."),
        ("A dozen eggs contained five eggs.", "A dozen eggs contained twelve eggs."),
        ("The box became lighter after stones were added.", "The box became heavier after stones were added."),
        ("Ten minutes lasted less time than one minute.", "Ten minutes lasted more time than one minute."),
    ],
}


def build_synthetic_rows(rows: list[CanonicalRow], seed: int) -> None:
    rng = random.Random(seed)
    variants = [
        ("In the report, {sentence}", "In the report, {corrected}"),
        ("A witness said that {sentence}", "A witness said that {corrected}"),
        ("The example states: {sentence}", "The example states: {corrected}"),
        ("A short caption says that {sentence}", "A short caption says that {corrected}"),
        ("The scenario claims that {sentence}", "The scenario claims that {corrected}"),
        ("A student wrote that {sentence}", "A student wrote that {corrected}"),
    ]
    for category, pairs in SYNTHETIC_TEMPLATES.items():
        for idx in range(60):
            original, corrected = pairs[idx % len(pairs)]
            prefix_original, prefix_corrected = variants[idx // len(pairs)]
            original_text = prefix_original.format(sentence=original[0].lower() + original[1:])
            corrected_text = prefix_corrected.format(corrected=corrected[0].lower() + corrected[1:])
            if idx >= 50:
                split = "test"
            elif idx >= 45:
                split = "validation"
            else:
                split = "train"
            jitter = rng.randint(0, 999999)
            add_pair_rows(
                rows,
                block="C",
                source="synthetic_commonsense",
                split=split,
                pair_id=f"{category}_{idx:02d}",
                first_text=original_text,
                second_text=corrected_text,
                first_label="counterfactual",
                second_label="commonsense_corrected",
                phenomenon="commonsense_violation",
                category=category,
                metadata={
                    "template_id": idx % len(pairs),
                    "wrapper_id": idx // len(pairs),
                    "seed_jitter": jitter,
                    "dataset": "controlled_synthetic",
                },
            )


def write_jsonl(rows: Iterable[CanonicalRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")


def count_by(rows: list[CanonicalRow], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(getattr(row, field_name))] += 1
    return dict(sorted(counts.items()))


def write_build_report(rows: list[CanonicalRow], warnings: list[str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pair_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_counts[row.source].add(row.pair_id)

    lines = [
        "# Dataset Build Report",
        "",
        f"Canonical path: `{CANONICAL_PATH}`",
        f"Total rows: `{len(rows)}`",
        f"Total pairs/groups: `{len({row.pair_id for row in rows})}`",
        "",
        "## Counts",
        "",
        f"- By block: `{count_by(rows, 'block')}`",
        f"- By phenomenon: `{count_by(rows, 'phenomenon')}`",
        f"- By source: `{count_by(rows, 'source')}`",
        f"- By split: `{count_by(rows, 'split')}`",
        f"- By label: `{count_by(rows, 'label')}`",
        "",
        "## Pair Counts By Source",
        "",
    ]
    for source, pairs in sorted(pair_counts.items()):
        lines.append(f"- `{source}`: `{len(pairs)}` pairs/groups")

    lines.extend(
        [
            "",
            "## Sources",
            "",
            "- Manual controlled standard sentences for Block A.",
            "- Hugging Face `jinaai/negation-dataset` for Block B.",
            "- Hugging Face `HiTZ/This-is-not-a-dataset` for Block B.",
            "- Official SemEval 2020 Task 4 ComVE CSVs from the public GitHub repository for Block C.",
            "- Controlled synthetic commonsense violations for Block C.",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    BUILD_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> int:
    rows: list[CanonicalRow] = []
    warnings: list[str] = []

    build_standard_rows(rows)

    builders = [
        ("jina_negation", lambda: build_jina_negation_rows(rows, args.jina_per_split)),
        ("this_is_not", lambda: build_this_is_not_rows(rows, args.this_is_not_pairs)),
        ("semeval2020_task4", lambda: build_semeval_rows(rows, args.semeval_per_split)),
    ]
    for name, builder in builders:
        try:
            builder()
        except Exception as exc:
            warnings.append(f"{name} unavailable: {exc}")

    build_synthetic_rows(rows, seed=args.seed)
    rows = sorted(rows, key=lambda row: row.id)
    write_jsonl(rows, CANONICAL_PATH)
    write_build_report(rows, warnings)
    print(f"Wrote {len(rows)} rows to {CANONICAL_PATH}")
    print(f"Wrote report to {BUILD_REPORT_PATH}")
    if warnings:
        print("Warnings:", *warnings, sep="\n- ", file=sys.stderr)
    return 0 if not warnings else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical thesis inversion datasets.")
    parser.add_argument("--jina-per-split", type=int, default=120)
    parser.add_argument("--this-is-not-pairs", type=int, default=120)
    parser.add_argument("--semeval-per-split", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
