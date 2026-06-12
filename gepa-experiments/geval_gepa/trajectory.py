"""GEPA trajectory artifact helpers."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any


def write_fallback_gepa_viz_run(
    path: Path,
    *,
    trainset: list[Any],
    valset: list[Any],
    seed_prompt: str,
    optimized_prompt: str,
    optimized_score: float | None,
) -> None:
    """Write a minimal run.json compatible with the documented gepa-viz schema."""

    examples = [_example_to_json(row) for row in valset]
    candidates: dict[str, dict[str, Any]] = {
        "0": {
            "prompt": seed_prompt,
            "parent": None,
            "score": None,
            "predictions": None,
            "minibatch": None,
        }
    }
    if optimized_prompt != seed_prompt or optimized_score is not None:
        candidates["1"] = {
            "prompt": optimized_prompt,
            "parent": "0",
            "score": optimized_score,
            "predictions": None,
            "minibatch": None,
        }
    payload = {
        "examples": examples,
        "train_example_count": len(trainset),
        "candidates": candidates,
        "fallback_export": True,
        "note": "gepa-viz callback was unavailable or did not write a run file; candidate details are minimal.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_prompt_trajectory(run_json_path: Path, output_path: Path) -> int:
    """Export prompt candidates from gepa-viz run.json into jsonl rows."""

    data = json.loads(run_json_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError(f"{run_json_path} does not contain a candidates object")

    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in sorted(candidates.items(), key=lambda item: _candidate_sort_key(item[0])):
        if not isinstance(candidate, dict):
            continue
        parent_id = candidate.get("parent")
        prompt = str(candidate.get("prompt") or "")
        parent_prompt = ""
        if parent_id is not None and str(parent_id) in candidates and isinstance(candidates[str(parent_id)], dict):
            parent_prompt = str(candidates[str(parent_id)].get("prompt") or "")
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "parent_id": None if parent_id is None else str(parent_id),
                "prompt_text": prompt,
                "diff_from_parent": _prompt_diff(parent_prompt, prompt) if parent_prompt else "",
                "score": candidate.get("score"),
                "accepted": candidate.get("score") is not None,
                "prediction_count": _safe_len(candidate.get("predictions")),
                "minibatch_count": _safe_len(candidate.get("minibatch")),
                "feedback": _collect_feedback(candidate.get("minibatch")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def _example_to_json(example: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in ("dataset", "dimension", "example_id", "group_id", "source_text", "fact", "candidate_output"):
        if hasattr(example, key):
            data[key] = getattr(example, key)
    if hasattr(example, "toDict"):
        try:
            value = example.toDict()
            if isinstance(value, dict):
                data.update(value)
        except Exception:
            pass
    return data


def _prompt_diff(parent: str, child: str) -> str:
    return "".join(
        difflib.unified_diff(
            parent.splitlines(keepends=True),
            child.splitlines(keepends=True),
            fromfile="parent",
            tofile="candidate",
            lineterm="",
        )
    )


def _safe_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _collect_feedback(minibatch: Any) -> str:
    if not isinstance(minibatch, list):
        return ""
    feedback = []
    for item in minibatch:
        if isinstance(item, dict) and item.get("feedback"):
            feedback.append(str(item["feedback"]))
    return "\n\n".join(feedback)


def _candidate_sort_key(candidate_id: str) -> tuple[int, int, str]:
    text = str(candidate_id)
    if "." in text:
        head, tail = text.split(".", 1)
        if head.isdigit() and tail.isdigit():
            return int(head), int(tail), text
    if text.isdigit():
        return int(text), 0, text
    return 10**9, 0, text
