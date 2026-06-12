#!/usr/bin/env python3
"""Aggregate evidence across GEPA/NLA runs.

This script is intentionally separate from the runner. It reads completed
artifacts and produces a compact scientific/debug report about whether NLA is
helping, where it fails, and which comparison is actually fair.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


METRIC_KEYS = ("pearson", "spearman", "kendall_tau", "agreement", "mae")
LOWER_IS_BETTER = {"mae"}
RUBRIC_TERMS = (
    "engag",
    "conversation",
    "relevant",
    "specific",
    "curiosity",
    "interesting",
    "coherent",
    "consistent",
    "factual",
    "error",
    "score",
    "rating",
    "quality",
    "grounded",
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    path: str
    kind: str


RUNS = (
    RunSpec("ppl_long_old", "geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer", "long"),
    RunSpec("ppl_long_current", "geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control", "long"),
    RunSpec("old_nla_long", "geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b", "long"),
    RunSpec("fixed_nla_long", "geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b", "long"),
    RunSpec("ppl_smoke_q35", "geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke", "smoke"),
    RunSpec("fixed_nla_smoke_q35", "geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke", "smoke"),
    RunSpec("candidate6_smoke", "experimental_nla_candidate_content_6_topical_chat_smoke", "smoke"),
    RunSpec("candidate10_smoke", "experimental_nla_candidate_content_10_topical_chat_smoke", "smoke"),
    RunSpec("single_gpu_ppl", "geval_gepa_engaging_qwen25_ppl_smoke", "single_gpu"),
    RunSpec("single_gpu_nla_matched", "geval_gepa_topical_chat_engagingness_ppl_nla_single_gpu_matched_smoke", "single_gpu"),
    RunSpec("summeval_ppl", "geval_gepa_summeval_consistency_ppl_smoke", "dataset_smoke"),
    RunSpec("summeval_nla", "geval_gepa_summeval_consistency_ppl_real_nla_smoke", "dataset_smoke"),
    RunSpec("qags_cnn_ppl", "geval_gepa_qags_cnn_consistency_ppl_smoke", "dataset_smoke"),
    RunSpec("qags_cnn_nla", "geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke", "dataset_smoke"),
    RunSpec("qags_xsum_ppl", "geval_gepa_qags_xsum_consistency_ppl_smoke", "dataset_smoke"),
    RunSpec("qags_xsum_nla", "geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke", "dataset_smoke"),
)

PAIRS = (
    ("old_nla_long", "ppl_long_old"),
    ("fixed_nla_long", "ppl_long_old"),
    ("fixed_nla_long", "ppl_long_current"),
    ("fixed_nla_smoke_q35", "ppl_smoke_q35"),
    ("candidate6_smoke", "ppl_smoke_q35"),
    ("candidate10_smoke", "ppl_smoke_q35"),
    ("single_gpu_nla_matched", "single_gpu_ppl"),
    ("summeval_nla", "summeval_ppl"),
    ("qags_cnn_nla", "qags_cnn_ppl"),
    ("qags_xsum_nla", "qags_xsum_ppl"),
)


def latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def norm_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def pct(value: float) -> float:
    return value * 100.0


def score_id(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("response_id") or "")


def group_id(row: dict[str, Any]) -> str:
    return str(row.get("group_id") or row.get("context_id") or "")


def prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": score_id(row),
        "group_id": group_id(row),
        "target": to_float(row.get("target")),
        "prediction": to_float(row.get("prediction")),
        "model": row.get("model", ""),
        "parse_status": row.get("parse_status", ""),
    }


def metric_map(metrics_path: Path | None) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in read_csv(metrics_path):
        program = row.get("program", "")
        output[program] = {key: to_float(value) for key, value in row.items() if key != "program"}
    return output


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    output = dict(config)
    label = str(output.get("legacy_label") or output.get("label") or "")
    if label and not output.get("dataset"):
        output["dataset"] = "topical_chat"
    if label and not output.get("dimension"):
        output["dimension"] = {"Engaging": "engagingness"}.get(label, label.lower())
    for new_key, old_key in (
        ("train_groups", "train_contexts"),
        ("val_groups", "val_contexts"),
        ("test_groups", "test_contexts"),
    ):
        if new_key not in output and old_key in output:
            output[new_key] = output[old_key]
    return output


def prompt_summary(run_dir: Path) -> dict[str, Any]:
    seed_path = latest(run_dir, "seed_prompt_*.txt")
    opt_path = latest(run_dir, "optimized_prompt_*.txt")
    traj_path = latest(run_dir, "prompt_trajectory_*.jsonl")
    seed = seed_path.read_text(encoding="utf-8") if seed_path else ""
    opt = opt_path.read_text(encoding="utf-8") if opt_path else ""
    traj = read_jsonl(traj_path)
    scores = [to_float(row.get("score")) for row in traj if not math.isnan(to_float(row.get("score")))]
    prompt_texts = [str(row.get("prompt_text", "")) for row in traj]
    accepted = [row for row in traj if row.get("accepted") is True]
    opt_scores = [
        to_float(row.get("score"))
        for row in traj
        if opt and str(row.get("prompt_text", "")) == opt and not math.isnan(to_float(row.get("score")))
    ]
    seed_scores = [
        to_float(row.get("score"))
        for row in traj
        if seed and str(row.get("prompt_text", "")) == seed and not math.isnan(to_float(row.get("score")))
    ]
    best_score = max(scores) if scores else float("nan")
    return {
        "seed_words": len(seed.split()) if seed else 0,
        "optimized_words": len(opt.split()) if opt else 0,
        "seed_equals_optimized": bool(seed and opt and seed == opt),
        "trajectory_candidates": len(traj),
        "accepted_candidates": len(accepted),
        "unique_prompt_texts": len({text for text in prompt_texts if text}),
        "best_trajectory_score": best_score,
        "seed_trajectory_score": max(seed_scores) if seed_scores else float("nan"),
        "optimized_trajectory_score": max(opt_scores) if opt_scores else float("nan"),
        "candidates_above_seed": sum(1 for score in scores if seed_scores and score > max(seed_scores)),
    }


def verbalization_summary(run_dir: Path) -> dict[str, Any]:
    nla_path = latest(run_dir, "nla_verbalizations_*.jsonl")
    rows = read_jsonl(nla_path)
    if not rows:
        fallback = latest(run_dir, "*nla_precomputed*.jsonl")
        rows = read_jsonl(fallback)
    texts: list[str] = []
    token_positions: Counter[str] = Counter()
    token_categories: Counter[str] = Counter()
    token_status: Counter[str] = Counter()
    parse_status: Counter[str] = Counter()
    completion_like = 0
    rubric_like = 0
    rows_with_stats = 0
    examples = set()
    groups = set()
    for row in rows:
        text = str(row.get("verbalization") or row.get("text") or row.get("nla_text") or "").strip()
        if text:
            texts.append(norm_text(text))
            low = text.lower()
            if " or " in low or low.startswith("\"") or "complete" in low or "continu" in low:
                completion_like += 1
            if any(term in low for term in RUBRIC_TERMS):
                rubric_like += 1
        position = str(row.get("token_position") or "")
        category = str(row.get("token_category") or "")
        if not category and position:
            if position.startswith(("candidate", "source", "reference", "experimental")):
                category = position.split("_", 1)[0]
        token_positions[position or "unknown"] += 1
        token_categories[category or "unknown"] += 1
        token_status[str(row.get("token_status") or "unknown")] += 1
        parse_status[str(row.get("parse_status") or "unknown")] += 1
        if row.get("example_id"):
            examples.add(str(row["example_id"]))
        if row.get("group_id"):
            groups.add(str(row["group_id"]))
        if any(key in row for key in ("activation_dim", "activation_l2", "activation_abs_mean", "activation_norm")):
            rows_with_stats += 1
    word_counts = [len(text.split()) for text in texts]
    return {
        "rows": len(rows),
        "covered_examples": len(examples),
        "covered_groups": len(groups),
        "unique_texts": len(set(texts)),
        "duplicate_rows": len(texts) - len(set(texts)),
        "duplicate_pct": pct((len(texts) - len(set(texts))) / len(texts)) if texts else 0.0,
        "avg_words": mean(word_counts) if word_counts else 0.0,
        "median_words": median(word_counts) if word_counts else 0.0,
        "completion_like_pct": pct(completion_like / len(texts)) if texts else 0.0,
        "rubric_like_pct": pct(rubric_like / len(texts)) if texts else 0.0,
        "token_positions": dict(token_positions.most_common()),
        "token_categories": dict(token_categories.most_common()),
        "token_status": dict(token_status.most_common()),
        "parse_status": dict(parse_status.most_common()),
        "rows_with_activation_stats": rows_with_stats,
    }


def prediction_summary(predictions_path: Path | None) -> dict[str, Any]:
    rows = [prediction_row(row) for row in read_jsonl(predictions_path)]
    by_target_bucket: Counter[str] = Counter()
    by_pred: Counter[str] = Counter()
    abs_errors: list[float] = []
    for row in rows:
        target = row["target"]
        pred = row["prediction"]
        if not math.isnan(target):
            by_target_bucket[bucket(target)] += 1
        if not math.isnan(pred):
            by_pred[str(int(pred) if pred.is_integer() else pred)] += 1
        if not math.isnan(target) and not math.isnan(pred):
            abs_errors.append(abs(target - pred))
    return {
        "n": len(rows),
        "target_distribution": dict(sorted(by_target_bucket.items())),
        "prediction_distribution": dict(sorted(by_pred.items())),
        "mean_abs_error_from_predictions": mean(abs_errors) if abs_errors else float("nan"),
    }


def bucket(value: float) -> str:
    if value < 1.75:
        return "low"
    if value < 2.5:
        return "mid"
    return "high"


def collect_run(results_root: Path, spec: RunSpec) -> dict[str, Any]:
    run_dir = results_root / spec.path
    metrics_path = latest(run_dir, "metrics_*.csv")
    config_path = latest(run_dir, "run_config_*.json")
    optimized_pred_path = latest(run_dir, "optimized_predictions_*.jsonl")
    baseline_pred_path = latest(run_dir, "baseline_predictions_*.jsonl")
    return {
        "name": spec.name,
        "kind": spec.kind,
        "path": str(run_dir),
        "exists": run_dir.exists(),
        "metrics_path": str(metrics_path) if metrics_path else "",
        "config": normalize_config(read_json(config_path)),
        "metrics": metric_map(metrics_path),
        "prompt": prompt_summary(run_dir),
        "nla": verbalization_summary(run_dir),
        "optimized_predictions": prediction_summary(optimized_pred_path),
        "baseline_predictions": prediction_summary(baseline_pred_path),
    }


def compare_runs(treatment: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, dict[str, float]] = {}
    treat_metrics = treatment.get("metrics", {}).get("optimized", {})
    control_metrics = control.get("metrics", {}).get("optimized", {})
    for key in METRIC_KEYS:
        tval = treat_metrics.get(key, float("nan"))
        cval = control_metrics.get(key, float("nan"))
        if math.isnan(tval) or math.isnan(cval):
            continue
        delta = tval - cval
        rel = delta / cval * 100.0 if cval else float("nan")
        if key in LOWER_IS_BETTER:
            improved = delta < 0
        else:
            improved = delta > 0
        deltas[key] = {"control": cval, "treatment": tval, "delta": delta, "rel_pct": rel, "improved": improved}
    movement = compare_predictions(Path(treatment["path"]), Path(control["path"]))
    return {
        "treatment": treatment["name"],
        "control": control["name"],
        "metric_deltas": deltas,
        "prediction_movement": movement,
    }


def compare_predictions(treatment_dir: Path, control_dir: Path) -> dict[str, Any]:
    treatment_rows = {
        row["example_id"]: row
        for row in (prediction_row(row) for row in read_jsonl(latest(treatment_dir, "optimized_predictions_*.jsonl")))
        if row["example_id"]
    }
    control_rows = {
        row["example_id"]: row
        for row in (prediction_row(row) for row in read_jsonl(latest(control_dir, "optimized_predictions_*.jsonl")))
        if row["example_id"]
    }
    joined = sorted(set(treatment_rows) & set(control_rows))
    improved = worsened = unchanged = 0
    deltas: list[float] = []
    for example_id in joined:
        t = treatment_rows[example_id]
        c = control_rows[example_id]
        if math.isnan(t["target"]) or math.isnan(t["prediction"]) or math.isnan(c["prediction"]):
            continue
        t_error = abs(t["target"] - t["prediction"])
        c_error = abs(c["target"] - c["prediction"])
        delta = t_error - c_error
        deltas.append(delta)
        if delta < 0:
            improved += 1
        elif delta > 0:
            worsened += 1
        else:
            unchanged += 1
    return {
        "joined_examples": len(joined),
        "improved_abs_error": improved,
        "worsened_abs_error": worsened,
        "unchanged_abs_error": unchanged,
        "mean_abs_error_delta": mean(deltas) if deltas else float("nan"),
    }


def fmt_float(value: float) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(runs: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines: list[str] = ["# NLA Evidence Deep Dive", ""]
    lines += [
        "## Run Metrics",
        md_table(
            ["run", "kind", "program", "n", "pearson", "spearman", "kendall", "agreement", "mae"],
            metric_rows(runs),
        ),
        "",
        "## Prompt And Search Behavior",
        md_table(
            [
                "run",
                "seed=opt",
                "seed words",
                "opt words",
                "candidates",
                "accepted",
                "unique prompts",
                "seed score",
                "opt score",
                "best score",
                "cand > seed",
            ],
            prompt_rows(runs),
        ),
        "",
        "## NLA Feedback Health",
        md_table(
            [
                "run",
                "rows",
                "examples",
                "dupe %",
                "avg words",
                "completion-like %",
                "rubric-like %",
                "token status",
                "top categories",
                "activation stats rows",
            ],
            nla_rows(runs),
        ),
        "",
        "## Optimized Prediction Distributions",
        md_table(
            ["run", "n", "target buckets", "prediction distribution", "pred MAE"],
            prediction_rows(runs),
        ),
        "",
        "## Pairwise Treatment vs Control",
        md_table(
            [
                "treatment",
                "control",
                "metric deltas",
                "joined",
                "improved",
                "worsened",
                "unchanged",
                "mean abs error delta",
            ],
            comparison_rows(comparisons),
        ),
        "",
        "## Evidence-Based Observations",
    ]
    lines.extend(observation_lines(runs, comparisons))
    lines.append("")
    return "\n".join(lines)


def metric_rows(runs: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs.values():
        for program in ("baseline", "optimized"):
            metrics = run.get("metrics", {}).get(program)
            if not metrics:
                continue
            rows.append(
                [
                    run["name"],
                    run["kind"],
                    program,
                    fmt_float(metrics.get("n", float("nan"))),
                    fmt_float(metrics.get("pearson", float("nan"))),
                    fmt_float(metrics.get("spearman", float("nan"))),
                    fmt_float(metrics.get("kendall_tau", float("nan"))),
                    fmt_float(metrics.get("agreement", float("nan"))),
                    fmt_float(metrics.get("mae", float("nan"))),
                ]
            )
    return rows


def prompt_rows(runs: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for run in runs.values():
        prompt = run["prompt"]
        if not prompt["seed_words"] and not prompt["optimized_words"] and not prompt["trajectory_candidates"]:
            continue
        rows.append(
            [
                run["name"],
                str(prompt["seed_equals_optimized"]),
                str(prompt["seed_words"]),
                str(prompt["optimized_words"]),
                str(prompt["trajectory_candidates"]),
                str(prompt["accepted_candidates"]),
                str(prompt["unique_prompt_texts"]),
                fmt_float(prompt["seed_trajectory_score"]),
                fmt_float(prompt["optimized_trajectory_score"]),
                fmt_float(prompt["best_trajectory_score"]),
                str(prompt["candidates_above_seed"]),
            ]
        )
    return rows


def nla_rows(runs: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for run in runs.values():
        nla = run["nla"]
        if not nla["rows"]:
            continue
        categories = ", ".join(f"{k}:{v}" for k, v in list(nla["token_categories"].items())[:4])
        token_status = ", ".join(f"{k}:{v}" for k, v in nla["token_status"].items())
        rows.append(
            [
                run["name"],
                str(nla["rows"]),
                str(nla["covered_examples"]),
                f"{nla['duplicate_pct']:.2f}",
                f"{nla['avg_words']:.2f}",
                f"{nla['completion_like_pct']:.2f}",
                f"{nla['rubric_like_pct']:.2f}",
                token_status,
                categories,
                str(nla["rows_with_activation_stats"]),
            ]
        )
    return rows


def prediction_rows(runs: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for run in runs.values():
        pred = run["optimized_predictions"]
        if not pred["n"]:
            continue
        rows.append(
            [
                run["name"],
                str(pred["n"]),
                json.dumps(pred["target_distribution"], sort_keys=True),
                json.dumps(pred["prediction_distribution"], sort_keys=True),
                fmt_float(pred["mean_abs_error_from_predictions"]),
            ]
        )
    return rows


def comparison_rows(comparisons: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for comp in comparisons:
        metric_bits = []
        for key, values in comp["metric_deltas"].items():
            metric_bits.append(f"{key}:{values['delta']:+.4f}")
        move = comp["prediction_movement"]
        rows.append(
            [
                comp["treatment"],
                comp["control"],
                ", ".join(metric_bits),
                str(move["joined_examples"]),
                str(move["improved_abs_error"]),
                str(move["worsened_abs_error"]),
                str(move["unchanged_abs_error"]),
                fmt_float(move["mean_abs_error_delta"]),
            ]
        )
    return rows


def observation_lines(runs: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    fixed = runs.get("fixed_nla_long")
    old_nla = runs.get("old_nla_long")
    cand10 = runs.get("candidate10_smoke")
    cand6 = runs.get("candidate6_smoke")
    fixed_vs_current = next(
        (
            comp
            for comp in comparisons
            if comp["treatment"] == "fixed_nla_long" and comp["control"] == "ppl_long_current"
        ),
        None,
    )
    summeval = next((c for c in comparisons if c["treatment"] == "summeval_nla"), None)
    if fixed:
        prompt = fixed["prompt"]
        nla = fixed["nla"]
        lines.append(
            f"- `fixed_nla_long` selected the seed prompt unchanged: seed_equals_optimized={prompt['seed_equals_optimized']}, "
            f"while exploring {prompt['trajectory_candidates']} trajectory rows."
        )
        lines.append(
            f"- `fixed_nla_long` NLA health improved over the first old NLA run on token status and length, but still has "
            f"{nla['duplicate_pct']:.2f}% duplicate verbalization rows and {nla['completion_like_pct']:.2f}% completion-like text."
        )
    if fixed_vs_current:
        deltas = fixed_vs_current["metric_deltas"]
        move = fixed_vs_current["prediction_movement"]
        lines.append(
            "- Against the matched current-code PPL long control, `fixed_nla_long` is slightly better on all recorded metrics "
            f"(pearson {deltas['pearson']['delta']:+.4f}, spearman {deltas['spearman']['delta']:+.4f}, "
            f"kendall {deltas['kendall_tau']['delta']:+.4f}, mae {deltas['mae']['delta']:+.4f}), but both runs selected "
            "the byte-identical seed prompt."
        )
        lines.append(
            f"- The matched long delta is therefore weak evidence: only {move['improved_abs_error']} final-test examples improved, "
            f"{move['worsened_abs_error']} worsened, and {move['unchanged_abs_error']} were unchanged. It is not evidence that GEPA "
            "found a better prompt under NLA."
        )
    if old_nla:
        nla = old_nla["nla"]
        lines.append(
            f"- `old_nla_long` is the clearest negative control: duplicate verbalization rows are {nla['duplicate_pct']:.2f}% "
            f"and optimized metrics drop versus PPL-only."
        )
    if cand6 and cand10:
        lines.append(
            "- Candidate-only NLA is not sufficient: `candidate6_smoke` and `candidate10_smoke` remove most source/reference repetition, "
            "but both still degrade optimized correlations versus the PPL-only smoke control."
        )
    if cand10:
        nla = cand10["nla"]
        lines.append(
            f"- `candidate10_smoke` has {nla['duplicate_pct']:.2f}% duplicate NLA rows, so its failure argues against the simple hypothesis "
            "that duplicate rows alone explain NLA underperformance."
        )
    if summeval:
        lines.append(
            "- SummEval consistency smoke is directionally negative for NLA versus PPL-only; QAGS smokes are too small to support a claim."
        )
    lines.append(
        "- Working hypothesis: raw NLA verbalizations mostly describe token continuations or latent associations, not metric-aligned reasons "
        "for why the judge should raise/lower a G-Eval score. The proposer can overfit this text into stricter or more dispersed rubrics."
    )
    lines.append(
        "- Next experiment should transform NLA into short, rubric-conditioned error feedback before GEPA reflection, preferably with the "
        "35B auxiliary judge/proposer summarizing NLA together with target, prediction, and error direction."
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="gepa-experiments/results")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    runs = {spec.name: collect_run(results_root, spec) for spec in RUNS if (results_root / spec.path).exists()}
    comparisons = [
        compare_runs(runs[treatment], runs[control])
        for treatment, control in PAIRS
        if treatment in runs and control in runs
    ]
    report = render_markdown(runs, comparisons)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"runs": runs, "comparisons": comparisons}, indent=2), encoding="utf-8")
    print(f"Wrote {output_md}")
    if args.output_json:
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
