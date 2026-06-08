#!/usr/bin/env python3
"""Diagnose why an NLA GEPA run differs from its closest non-NLA control."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METRIC_KEYS = ("agreement", "pearson", "spearman", "kendall_tau", "mae", "coverage")
CONFIG_KEYS = (
    "dataset",
    "dimension",
    "seed",
    "train_groups",
    "val_groups",
    "test_groups",
    "judge_model",
    "proposer_model",
    "proposer_temperature",
    "proposer_max_tokens",
    "instruction_proposer",
    "perplexity_feedback",
)


def latest_file(run_dir: Path, pattern: str) -> Path | None:
    files = sorted(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_metrics(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    output: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            program = row.get("program", "")
            output[program] = {}
            for key, value in row.items():
                if key == "program":
                    continue
                output[program][key] = _float_or_nan(value)
    return output


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = latest_file(run_dir, "metrics_*.csv")
    suffix = strip_prefix(metrics_path.stem, "metrics_") if metrics_path else ""
    config_path = run_dir / f"run_config_{suffix}.json" if suffix else latest_file(run_dir, "run_config_*.json")
    if config_path is not None and not config_path.exists():
        config_path = latest_file(run_dir, "run_config_*.json")
    return {
        "run_dir": run_dir,
        "metrics_path": metrics_path,
        "config_path": config_path,
        "baseline_predictions_path": run_dir / f"baseline_predictions_{suffix}.jsonl" if suffix else latest_file(run_dir, "baseline_predictions_*.jsonl"),
        "optimized_predictions_path": run_dir / f"optimized_predictions_{suffix}.jsonl" if suffix else latest_file(run_dir, "optimized_predictions_*.jsonl"),
        "trajectory_path": run_dir / f"prompt_trajectory_{suffix}.jsonl" if suffix else latest_file(run_dir, "prompt_trajectory_*.jsonl"),
        "nla_path": run_dir / f"nla_verbalizations_{suffix}.jsonl" if suffix else latest_file(run_dir, "nla_verbalizations_*.jsonl"),
        "prompt_path": run_dir / f"optimized_prompt_{suffix}.txt" if suffix else latest_file(run_dir, "optimized_prompt_*.txt"),
        "metrics": read_metrics(metrics_path),
        "config": read_json(config_path),
    }


def strip_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def compare_configs(control: dict[str, Any], nla: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in CONFIG_KEYS:
        control_value = control["config"].get(key, "")
        nla_value = nla["config"].get(key, "")
        rows.append(
            {
                "key": key,
                "control": control_value,
                "nla": nla_value,
                "matches": control_value == nla_value,
            }
        )
    for key in ("nla_feedback", "nla_backend", "nla_precomputed_path", "nla_max_tokens_per_example"):
        rows.append(
            {
                "key": key,
                "control": control["config"].get(key, ""),
                "nla": nla["config"].get(key, ""),
                "matches": control["config"].get(key, "") == nla["config"].get(key, ""),
            }
        )
    return rows


def compare_metrics(control: dict[str, Any], nla: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for program in ("baseline", "optimized"):
        for key in METRIC_KEYS:
            control_value = control["metrics"].get(program, {}).get(key, float("nan"))
            nla_value = nla["metrics"].get(program, {}).get(key, float("nan"))
            delta = nla_value - control_value
            rows.append(
                {
                    "program": program,
                    "metric": key,
                    "control": control_value,
                    "nla": nla_value,
                    "delta_nla_minus_control": delta,
                    "relative_delta_pct": _relative_delta_pct(delta, control_value),
                    "higher_is_better": key != "mae",
                }
            )
    return rows


def _relative_delta_pct(delta: float, baseline: float) -> float:
    if not math.isfinite(delta) or not math.isfinite(baseline) or baseline == 0:
        return float("nan")
    return 100.0 * delta / baseline


def prediction_comparison(control_run: dict[str, Any], nla_run: dict[str, Any]) -> list[dict[str, Any]]:
    control = _prediction_by_example(read_jsonl(control_run["optimized_predictions_path"]))
    nla = _prediction_by_example(read_jsonl(nla_run["optimized_predictions_path"]))
    rows = []
    for example_id in sorted(set(control) & set(nla)):
        c = control[example_id]
        n = nla[example_id]
        target = _float_or_nan(n.get("target", c.get("target")))
        control_prediction = _float_or_nan(c.get("prediction"))
        nla_prediction = _float_or_nan(n.get("prediction"))
        control_abs_error = abs(control_prediction - target)
        nla_abs_error = abs(nla_prediction - target)
        rows.append(
            {
                "example_id": example_id,
                "group_id": n.get("group_id", c.get("group_id", "")),
                "target": target,
                "control_prediction": control_prediction,
                "nla_prediction": nla_prediction,
                "control_abs_error": control_abs_error,
                "nla_abs_error": nla_abs_error,
                "abs_error_delta_nla_minus_control": nla_abs_error - control_abs_error,
                "control_parse_status": c.get("parse_status", ""),
                "nla_parse_status": n.get("parse_status", ""),
            }
        )
    return rows


def _prediction_by_example(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("example_id", "")): row for row in rows if row.get("example_id")}


def nla_quality(nla_run: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(nla_run["nla_path"])
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_status = Counter()
    token_status = Counter()
    token_position_prefix = Counter()
    texts = Counter()
    lengths = []
    suspicious = 0
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if example_id:
            by_example[example_id].append(row)
        parse_status[str(row.get("parse_status", "missing"))] += 1
        token_status[str(row.get("token_status", "unknown"))] += 1
        token_position = str(row.get("token_position", "missing"))
        token_position_prefix[token_position.split("_", 1)[0]] += 1
        text = str(row.get("verbalization") or row.get("explanation") or row.get("raw_generation") or "").strip()
        texts[text] += 1
        lengths.append(len(text.split()))
        if _looks_suspicious_nla_text(text):
            suspicious += 1
    return {
        "rows": len(rows),
        "covered_examples": len(by_example),
        "avg_rows_per_covered_example": len(rows) / len(by_example) if by_example else 0.0,
        "parse_status": dict(parse_status),
        "token_status": dict(token_status),
        "token_position_prefix": dict(token_position_prefix),
        "avg_verbalization_words": sum(lengths) / len(lengths) if lengths else 0.0,
        "duplicate_text_rows": sum(count for text, count in texts.items() if text and count > 1),
        "suspicious_rows": suspicious,
        "top_repeated_text": [{"count": count, "text": text[:200]} for text, count in texts.most_common(5) if text],
    }


def _looks_suspicious_nla_text(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    markers = ("dry run only", "placeholder", "activation", "unavailable", "unknown")
    return any(marker in normalized for marker in markers)


def trajectory_stats(run: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(run["trajectory_path"])
    prompt_lengths = [len(str(row.get("prompt_text", "")).split()) for row in rows]
    accepted = sum(1 for row in rows if row.get("accepted"))
    return {
        "candidates": len(rows),
        "accepted_candidates": accepted,
        "avg_prompt_words": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0,
        "max_prompt_words": max(prompt_lengths) if prompt_lengths else 0,
    }


def write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "example_id",
        "group_id",
        "target",
        "control_prediction",
        "nla_prediction",
        "control_abs_error",
        "nla_abs_error",
        "abs_error_delta_nla_minus_control",
        "control_parse_status",
        "nla_parse_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(control_run_dir: Path, nla_run_dir: Path) -> dict[str, Any]:
    control = collect_run(control_run_dir)
    nla = collect_run(nla_run_dir)
    prediction_rows = prediction_comparison(control, nla)
    improved = sum(1 for row in prediction_rows if row["abs_error_delta_nla_minus_control"] < 0)
    worsened = sum(1 for row in prediction_rows if row["abs_error_delta_nla_minus_control"] > 0)
    unchanged = len(prediction_rows) - improved - worsened
    return {
        "control": control,
        "nla": nla,
        "config_comparison": compare_configs(control, nla),
        "metric_comparison": compare_metrics(control, nla),
        "prediction_rows": prediction_rows,
        "prediction_summary": {
            "joined_examples": len(prediction_rows),
            "nla_improved_examples": improved,
            "nla_worsened_examples": worsened,
            "unchanged_examples": unchanged,
        },
        "nla_quality": nla_quality(nla),
        "control_trajectory": trajectory_stats(control),
        "nla_trajectory": trajectory_stats(nla),
    }


def render_markdown(report: dict[str, Any], prediction_csv: Path) -> str:
    lines = [
        "# NLA Run Diagnostic Report",
        "",
        "## Runs",
        f"- control: `{report['control']['run_dir']}`",
        f"- nla: `{report['nla']['run_dir']}`",
        f"- prediction comparison csv: `{prediction_csv}`",
        "",
        "## Config Check",
    ]
    for row in report["config_comparison"]:
        mark = "OK" if row["matches"] else "DIFF"
        lines.append(f"- {mark} `{row['key']}`: control=`{row['control']}` nla=`{row['nla']}`")

    lines.extend(["", "## Metric Deltas"])
    for row in report["metric_comparison"]:
        delta = _format_float(row["delta_nla_minus_control"])
        rel = _format_float(row["relative_delta_pct"])
        direction = "higher better" if row["higher_is_better"] else "lower better"
        lines.append(
            f"- {row['program']} `{row['metric']}` ({direction}): "
            f"control={_format_float(row['control'])}, nla={_format_float(row['nla'])}, "
            f"delta={delta}, rel_pct={rel}"
        )

    summary = report["prediction_summary"]
    quality = report["nla_quality"]
    lines.extend(
        [
            "",
            "## Prediction-Level Error Movement",
            f"- joined examples: {summary['joined_examples']}",
            f"- NLA improved abs error: {summary['nla_improved_examples']}",
            f"- NLA worsened abs error: {summary['nla_worsened_examples']}",
            f"- unchanged: {summary['unchanged_examples']}",
            "",
            "## NLA Feedback Quality",
            f"- verbalization rows: {quality['rows']}",
            f"- covered examples: {quality['covered_examples']}",
            f"- avg rows per covered example: {_format_float(quality['avg_rows_per_covered_example'])}",
            f"- avg verbalization words: {_format_float(quality['avg_verbalization_words'])}",
            f"- suspicious rows: {quality['suspicious_rows']}",
            f"- duplicate text rows: {quality['duplicate_text_rows']}",
            f"- parse status: `{quality['parse_status']}`",
            f"- token status: `{quality['token_status']}`",
            f"- token position prefixes: `{quality['token_position_prefix']}`",
            "",
            "## Trajectory",
            f"- control: `{report['control_trajectory']}`",
            f"- nla: `{report['nla_trajectory']}`",
            "",
            "## Next Checks",
            "- If config rows other than NLA differ, rerun a stricter 1-to-1 control.",
            "- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.",
            "- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.",
            "- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.",
            "- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.",
        ]
    )
    if quality["top_repeated_text"]:
        lines.extend(["", "## Top Repeated NLA Text"])
        for item in quality["top_repeated_text"]:
            lines.append(f"- count={item['count']}: {item['text']}")
    return "\n".join(lines) + "\n"


def _format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--nla-run", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.control_run, args.nla_run)
    prediction_csv = args.prediction_csv or args.output_md.with_suffix(".prediction_errors.csv")
    write_prediction_csv(prediction_csv, report["prediction_rows"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report, prediction_csv), encoding="utf-8")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {prediction_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
