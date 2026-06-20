#!/usr/bin/env python3
"""Generate Chapter 5 result figures as PDF assets.

The input numbers are copied from the auditable CSV/diagnostic artifacts cited
in Chapter 5. Keeping this script in the LaTeX tree makes the figure assets easy
to regenerate before uploading the thesis to Overleaf.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "chapter5"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#2f5f8f",
    "green": "#4c8c4a",
    "orange": "#c9822b",
    "red": "#b8574f",
    "gray": "#6c757d",
}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def soft_prompt_deltas() -> None:
    labels = ["16 tok", "8 tok", "32 tok", "1024 ctx", "seed 43", "seed 44"]
    validation = [0.0313, 0.0047, 0.0431, 0.0277, -0.0733, -0.0632]
    final_test = [0.0557, 0.0218, -0.0384, 0.0233, 0.0863, -0.1008]
    xs = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([x - width / 2 for x in xs], validation, width, label="Validation", color=COLORS["blue"])
    ax.bar([x + width / 2 for x in xs], final_test, width, label="Final test", color=COLORS["orange"])
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_ylabel("Pearson delta")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-0.12, 0.10)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax)
    save(fig, "soft_prompt_deltas.pdf")


def gepa_long_run_metrics() -> None:
    labels = ["Old PPL", "Current PPL", "Raw NLA", "Fixed NLA"]
    pearson = [0.6328, 0.6582, 0.5111, 0.6812]
    spearman = [0.6199, 0.6582, 0.4908, 0.6771]
    agreement = [0.7889, 0.7417, 0.6917, 0.7528]
    xs = range(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([x - width for x in xs], pearson, width, label="Pearson", color=COLORS["blue"])
    ax.bar(list(xs), spearman, width, label="Spearman", color=COLORS["green"])
    ax.bar([x + width for x in xs], agreement, width, label="Agreement", color=COLORS["orange"])
    ax.set_ylabel("Metric value")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.35, 0.85)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax)
    save(fig, "gepa_long_run_metrics.pdf")


def nla_feedback_health() -> None:
    labels = ["Raw NLA", "Fixed NLA", "Cand. 6", "Cand. 10"]
    duplicate = [61.44, 42.18, 0.00, 0.00]
    completion_like = [97.11, 89.73, 92.51, 90.14]
    rubric_like = [91.67, 8.45, 9.09, 8.45]
    xs = range(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([x - width for x in xs], duplicate, width, label="Duplicate", color=COLORS["gray"])
    ax.bar(list(xs), completion_like, width, label="Completion-like", color=COLORS["red"])
    ax.bar([x + width for x in xs], rubric_like, width, label="Rubric-like", color=COLORS["blue"])
    ax.set_ylabel("Rows (%)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax)
    save(fig, "nla_feedback_health.pdf")


def candidate_sort_key(candidate_id: str) -> tuple[int, int, str]:
    text = str(candidate_id)
    if "." in text:
        head, tail = text.split(".", 1)
        if head.isdigit() and tail.isdigit():
            return int(head), int(tail), text
    if text.isdigit():
        return int(text), 0, text
    return 10**9, 0, text


def load_gepa_viz_scores(relative_path: str) -> tuple[list[int], list[float], list[float], list[int]]:
    data = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    candidates = data.get("candidates", {})
    xs: list[int] = []
    scores: list[float] = []
    best_scores: list[float] = []
    prompt_words: list[int] = []
    current_best = float("-inf")

    for candidate_id, candidate in sorted(candidates.items(), key=lambda item: candidate_sort_key(item[0])):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("score"), (int, float)):
            continue
        xs.append(len(xs))
        score = float(candidate["score"])
        scores.append(score)
        current_best = max(current_best, score)
        best_scores.append(current_best)
        prompt_words.append(len(str(candidate.get("prompt") or "").split()))

    return xs, scores, best_scores, prompt_words


def gepa_viz_search_trajectory() -> None:
    runs = [
        (
            "PPL control",
            "gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control/"
            "gepa_viz_run_20260611T155014Z.json",
            COLORS["blue"],
        ),
        (
            "Raw NLA",
            "gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b/"
            "gepa_viz_run_20260608T061600Z.json",
            COLORS["red"],
        ),
        (
            "Fixed NLA",
            "gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b/"
            "gepa_viz_run_20260610T163111Z.json",
            COLORS["green"],
        ),
    ]

    fig, (score_ax, length_ax) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for label, path, color in runs:
        xs, scores, best_scores, prompt_words = load_gepa_viz_scores(path)
        score_ax.scatter(xs, scores, color=color, alpha=0.42, s=16, linewidth=0, zorder=2)
        score_ax.plot(xs, best_scores, label=label, color=color, linewidth=2.1, zorder=3)
        length_ax.plot(xs, prompt_words, label=label, color=color, linewidth=1.2, alpha=0.85)

    score_ax.set_ylabel("Best validation score")
    score_ax.set_ylim(0.53, 0.72)
    score_ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    style_axes(score_ax)

    length_ax.set_xlabel("Evaluated GEPA candidate index")
    length_ax.set_ylabel("Prompt words")
    length_ax.set_ylim(120, 520)
    style_axes(length_ax)

    save(fig, "gepa_viz_search_trajectory.pdf")


def main() -> None:
    soft_prompt_deltas()
    gepa_long_run_metrics()
    nla_feedback_health()
    gepa_viz_search_trajectory()


if __name__ == "__main__":
    main()
