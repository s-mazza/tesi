"""Dataset loading utilities for the Topical-Chat USR annotations."""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_USR_URL = "https://shikib.com/tc_usr_data.json"

LABEL_SCALES: dict[str, tuple[int, int]] = {
    "Understandable": (0, 1),
    "Natural": (1, 3),
    "Maintains Context": (1, 3),
    "Engaging": (1, 3),
    "Uses Knowledge": (0, 1),
    "Overall": (1, 5),
}


@dataclass(frozen=True)
class UsrResponseExample:
    """One candidate response with aggregated human annotations."""

    context_id: str
    response_id: str
    context: str
    fact: str
    response: str
    model: str
    human_scores: dict[str, float]
    raw_annotations: dict[str, list[float]]

    def human_score(self, label: str) -> float:
        try:
            return self.human_scores[label]
        except KeyError as exc:
            available = ", ".join(sorted(self.human_scores))
            raise KeyError(f"Unknown label {label!r}. Available labels: {available}") from exc


def load_usr_examples(source: str | Path = DEFAULT_USR_URL) -> list[UsrResponseExample]:
    """Load and flatten the USR Topical-Chat JSON into response-level rows."""

    records = _read_json(source)
    examples: list[UsrResponseExample] = []

    for context_index, record in enumerate(records):
        context_id = str(record.get("id") or f"context_{context_index:03d}")
        context = _require_str(record, "context")
        fact = _require_str(record, "fact")
        responses = record.get("responses")
        if not isinstance(responses, list):
            raise ValueError(f"{context_id}: expected 'responses' to be a list")

        for response_index, response_record in enumerate(responses):
            if not isinstance(response_record, dict):
                raise ValueError(f"{context_id}: response {response_index} is not an object")

            raw_annotations = {
                label: _numeric_list(response_record.get(label), context_id, response_index, label)
                for label in LABEL_SCALES
            }
            human_scores = {
                label: sum(values) / len(values)
                for label, values in raw_annotations.items()
            }
            examples.append(
                UsrResponseExample(
                    context_id=context_id,
                    response_id=f"{context_id}_response_{response_index:02d}",
                    context=context,
                    fact=fact,
                    response=_require_str(response_record, "response"),
                    model=_require_str(response_record, "model"),
                    human_scores=human_scores,
                    raw_annotations=raw_annotations,
                )
            )

    return examples


def split_by_context(
    examples: Iterable[UsrResponseExample],
    *,
    train_contexts: int,
    val_contexts: int,
    test_contexts: int,
    seed: int,
) -> tuple[list[UsrResponseExample], list[UsrResponseExample], list[UsrResponseExample]]:
    """Split examples by context id to avoid response variants leaking across splits."""

    rows = list(examples)
    context_ids = sorted({row.context_id for row in rows})
    required = train_contexts + val_contexts + test_contexts
    if required > len(context_ids):
        raise ValueError(f"Requested {required} contexts, but dataset has {len(context_ids)}")

    rng = random.Random(seed)
    rng.shuffle(context_ids)
    train_ids = set(context_ids[:train_contexts])
    val_ids = set(context_ids[train_contexts : train_contexts + val_contexts])
    test_ids = set(context_ids[train_contexts + val_contexts : required])

    train = [row for row in rows if row.context_id in train_ids]
    val = [row for row in rows if row.context_id in val_ids]
    test = [row for row in rows if row.context_id in test_ids]
    return train, val, test


def _read_json(source: str | Path) -> Any:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        with urllib.request.urlopen(source_text, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    with Path(source).open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_str(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Expected field {field!r} to be a string")
    return value


def _numeric_list(value: Any, context_id: str, response_index: int, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context_id} response {response_index}: missing annotation list {label!r}")
    values: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"{context_id} response {response_index}: non-numeric {label!r} value {item!r}")
        values.append(float(item))
    lo, hi = LABEL_SCALES[label]
    if any(item < lo or item > hi for item in values):
        raise ValueError(f"{context_id} response {response_index}: {label!r} outside scale {lo}-{hi}")
    return values
