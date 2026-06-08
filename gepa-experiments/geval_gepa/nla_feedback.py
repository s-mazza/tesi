"""NLA verbalization feedback plumbing for GEPA."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NlaVerbalization:
    example_id: str
    group_id: str
    dataset: str
    dimension: str
    token_position: str
    token_text: str
    verbalization: str
    parse_status: str
    layer: int
    checkpoint: str
    backend: str


class NlaFeedbackProvider:
    """Provide per-example NLA verbalizations as textual GEPA feedback."""

    def __init__(
        self,
        *,
        checkpoint: str,
        layer: int,
        backend: str,
        max_tokens_per_example: int,
        precomputed_path: str = "",
    ) -> None:
        if backend not in {"dry_run", "precomputed"}:
            raise ValueError(f"Unsupported NLA feedback backend: {backend}")
        self.checkpoint = checkpoint
        self.layer = layer
        self.backend = backend
        self.max_tokens_per_example = max_tokens_per_example
        self._precomputed = _load_precomputed(precomputed_path) if backend == "precomputed" else {}
        self._emitted: dict[str, list[NlaVerbalization]] = {}
        self._lock = threading.Lock()

    def feedback_for(self, example: Any) -> str:
        example_id = str(getattr(example, "example_id", "") or getattr(example, "response_id", ""))
        if not example_id:
            return "NLA verbalizations unavailable: example_id is missing."

        with self._lock:
            cached = self._emitted.get(example_id)
        if cached is None:
            rows = self._precomputed.get(example_id) if self.backend == "precomputed" else None
            verbalizations = (
                self._from_precomputed(example, rows or [])
                if rows
                else self._dry_run_verbalizations(example)
            )
            with self._lock:
                self._emitted[example_id] = verbalizations
        else:
            verbalizations = cached

        if not verbalizations:
            return "NLA verbalizations unavailable: no token-level verbalization rows found."

        lines = [
            "NLA multi-token verbalizations from Qwen2.5-7B layer "
            f"{self.layer} ({self.backend}; use as weak interpretive feedback, not as ground truth):"
        ]
        for item in verbalizations[: self.max_tokens_per_example]:
            text = item.verbalization.replace("\n", " ").strip()
            lines.append(f"- {item.token_position} token={item.token_text!r}: {text}")
        return "\n".join(lines)

    def precomputed_stats_for(self, examples: list[Any]) -> dict[str, Any]:
        if self.backend != "precomputed":
            return {
                "backend": self.backend,
                "examples": len(examples),
                "covered_examples": 0,
                "coverage": 0.0,
                "rows": 0,
                "useful_rows": 0,
                "missing_example_ids": [],
            }

        missing: list[str] = []
        rows = 0
        useful_rows = 0
        covered_examples = 0
        for example in examples:
            example_id = str(getattr(example, "example_id", "") or getattr(example, "response_id", ""))
            example_rows = self._precomputed.get(example_id, [])
            if example_rows:
                covered_examples += 1
                rows += len(example_rows)
                useful_rows += sum(1 for row in example_rows if _is_useful_precomputed_row(row))
            else:
                missing.append(example_id)
        total = len(examples)
        return {
            "backend": self.backend,
            "examples": total,
            "covered_examples": covered_examples,
            "coverage": covered_examples / total if total else 0.0,
            "rows": rows,
            "useful_rows": useful_rows,
            "missing_example_ids": missing[:20],
        }

    def write_artifact(self, path: Path) -> int:
        rows: list[NlaVerbalization] = []
        for items in self._emitted.values():
            rows.extend(items)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
        return len(rows)

    def _from_precomputed(self, example: Any, rows: list[dict[str, Any]]) -> list[NlaVerbalization]:
        output: list[NlaVerbalization] = []
        for index, row in enumerate(rows[: self.max_tokens_per_example]):
            output.append(
                NlaVerbalization(
                    example_id=str(getattr(example, "example_id", "") or getattr(example, "response_id", "")),
                    group_id=str(getattr(example, "group_id", "") or getattr(example, "context_id", "")),
                    dataset=str(getattr(example, "dataset", "")),
                    dimension=str(getattr(example, "dimension", "")),
                    token_position=str(row.get("token_position") or f"precomputed_{index}"),
                    token_text=str(row.get("token_text") or row.get("token") or ""),
                    verbalization=str(row.get("verbalization") or row.get("explanation") or row.get("raw_generation") or ""),
                    parse_status=str(row.get("parse_status") or "unknown"),
                    layer=int(row.get("layer") or self.layer),
                    checkpoint=str(row.get("nla_model_id") or row.get("checkpoint") or self.checkpoint),
                    backend="precomputed",
                )
            )
        return output

    def _dry_run_verbalizations(self, example: Any) -> list[NlaVerbalization]:
        source = str(getattr(example, "source_text", "") or getattr(example, "context", ""))
        candidate = str(getattr(example, "candidate_output", "") or getattr(example, "response", ""))
        sampled = _sample_semantic_tokens(source=source, candidate=candidate, limit=self.max_tokens_per_example)
        return [
            NlaVerbalization(
                example_id=str(getattr(example, "example_id", "") or getattr(example, "response_id", "")),
                group_id=str(getattr(example, "group_id", "") or getattr(example, "context_id", "")),
                dataset=str(getattr(example, "dataset", "")),
                dimension=str(getattr(example, "dimension", "")),
                token_position=position,
                token_text=token,
                verbalization=(
                    "DRY RUN ONLY: placeholder for a real NLA activation verbalization at this semantic token. "
                    "Do not use this backend for scientific results."
                ),
                parse_status="dry_run",
                layer=self.layer,
                checkpoint=self.checkpoint,
                backend="dry_run",
            )
            for position, token in sampled
        ]


def _load_precomputed(path: str) -> dict[str, list[dict[str, Any]]]:
    if not path:
        raise ValueError("NLA_BACKEND=precomputed requires --nla-precomputed-path")
    by_example: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id") or row.get("base_example_id") or "")
            if not example_id:
                continue
            by_example.setdefault(example_id, []).append(row)
    return by_example


def _is_useful_precomputed_row(row: dict[str, Any]) -> bool:
    text = str(row.get("verbalization") or row.get("explanation") or row.get("raw_generation") or "").strip()
    parse_status = str(row.get("parse_status") or "").strip().lower()
    token_status = str(row.get("token_status") or "ok").strip().lower()
    if not text:
        return False
    if token_status and token_status not in {"ok", "unknown"}:
        return False
    return parse_status in {"ok", "partial_tags", "unknown", ""}


def _sample_semantic_tokens(*, source: str, candidate: str, limit: int) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for prefix, text in (("source", source), ("candidate", candidate)):
        words = re.findall(r"\b[\w'-]{4,}\b", text)
        if not words:
            continue
        indexes = sorted({0, len(words) // 2, len(words) - 1})
        for index in indexes:
            tokens.append((f"{prefix}_{index}", words[index]))
            if len(tokens) >= limit:
                return tokens
    return tokens
