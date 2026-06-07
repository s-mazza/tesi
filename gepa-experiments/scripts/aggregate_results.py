#!/usr/bin/env python3
"""Aggregate GEPA metrics into a thesis-ready CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = ("agreement", "pearson", "spearman", "mae", "coverage", "parsed", "total")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def iter_runs(results_root: Path):
    for metrics_path in sorted(results_root.rglob("metrics_*.csv")):
        run_dir = metrics_path.parent
        suffix = metrics_path.stem.removeprefix("metrics_")
        config_path = run_dir / f"run_config_{suffix}.json"
        if not config_path.exists():
            config_path = latest_file(run_dir, "run_config_*.json")
        yield run_dir, metrics_path, config_path


def row_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_rows(results_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run_dir, metrics_path, config_path in iter_runs(results_root):
        config: dict[str, Any] = {}
        if config_path and config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = read_csv(metrics_path)
        by_program = {row.get("program", ""): row for row in metrics}
        baseline = by_program.get("baseline", {})
        for row in metrics:
            record: dict[str, Any] = {
                "run_dir": str(run_dir),
                "metrics_file": str(metrics_path),
                "config_file": str(config_path) if config_path else "",
                "dataset": config.get("dataset", "topical_chat"),
                "dimension": config.get("dimension", config.get("label", "")),
                "program": row.get("program", ""),
                "proposer_model": config.get("proposer_model", ""),
                "perplexity_feedback": config.get("perplexity_feedback", False),
                "nla_feedback": config.get("nla_feedback", False),
                "aux_judge_feedback": config.get("aux_judge_feedback", False),
                "seed": config.get("seed", ""),
                "train_rows": config.get("rows", {}).get("gepa_train", ""),
                "val_rows": config.get("rows", {}).get("gepa_validation", ""),
                "test_rows": config.get("rows", {}).get("final_test", ""),
            }
            for key in METRIC_FIELDS:
                record[key] = row.get(key, "")
                value = row_float(row, key)
                base_value = row_float(baseline, key)
                if row.get("program") != "baseline" and value is not None and base_value not in (None, 0):
                    delta = value - base_value
                    if key == "mae":
                        record[f"{key}_improvement"] = -delta
                        record[f"{key}_relative_improvement_pct"] = (-delta / base_value) * 100
                    else:
                        record[f"{key}_improvement"] = delta
                        record[f"{key}_relative_improvement_pct"] = (delta / base_value) * 100
                else:
                    record[f"{key}_improvement"] = ""
                    record[f"{key}_relative_improvement_pct"] = ""
            output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("gepa-experiments/results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = build_rows(args.results_root)
    if not rows:
        raise SystemExit(f"No metrics_*.csv files found under {args.results_root}")
    fieldnames = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("w", newline="", encoding="utf-8")
    else:
        import sys

        handle = sys.stdout
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
