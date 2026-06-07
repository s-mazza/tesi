"""Task registry for G-EVAL-style prompt optimization experiments."""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .data import DEFAULT_USR_URL, LABEL_SCALES as USR_LABEL_SCALES, load_usr_examples


SUMMEVAL_URL = "https://raw.githubusercontent.com/nlpyang/geval/main/data/summeval.json"
QAGS_CNN_URL = "https://raw.githubusercontent.com/nlpyang/geval/main/data/qags_cnndm.json"
QAGS_XSUM_URL = "https://raw.githubusercontent.com/nlpyang/geval/main/data/qags_xsum.json"


@dataclass(frozen=True)
class EvalExample:
    """One normalized response/summary to score against human annotations."""

    dataset: str
    dimension: str
    example_id: str
    group_id: str
    source_text: str
    candidate_output: str
    human_score: float
    min_score: int
    max_score: int
    context: str = ""
    fact: str = "_nofact"
    reference: str = ""
    system_id: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskSpec:
    """Dataset/dimension configuration used by the runner and Slurm configs."""

    dataset: str
    dimensions: tuple[str, ...]
    default_source: str
    default_dimension: str
    task_type: str
    loader: Callable[[str | Path, str], list[EvalExample]]

    def load(self, source: str | Path | None, dimension: str) -> list[EvalExample]:
        resolved_source = source or self.default_source
        canonical_dimension = canonicalize_dimension(self.dataset, dimension or self.default_dimension)
        return self.loader(resolved_source, canonical_dimension)


SUMMEVAL_DIMENSIONS = ("fluency", "coherence", "consistency", "relevance")
TOPICAL_CHAT_DIMENSIONS = ("naturalness", "coherence", "engagingness", "groundedness")
QAGS_DIMENSIONS = ("consistency",)

TASK_REGISTRY: dict[str, TaskSpec] = {}


_DATASET_ALIASES = {
    "topical-chat": "topical_chat",
    "topical_chat": "topical_chat",
    "topicalchat": "topical_chat",
    "usr": "topical_chat",
    "summeval": "summeval",
    "qags-cnn": "qags_cnn",
    "qags_cnn": "qags_cnn",
    "qags-cnndm": "qags_cnn",
    "qags-cnndailymail": "qags_cnn",
    "qags-xsum": "qags_xsum",
    "qags_xsum": "qags_xsum",
}

_TOPICAL_DIMENSION_TO_USR_LABEL = {
    "naturalness": "Natural",
    "natural": "Natural",
    "coherence": "Maintains Context",
    "maintains_context": "Maintains Context",
    "maintains context": "Maintains Context",
    "engagingness": "Engaging",
    "engaging": "Engaging",
    "groundedness": "Uses Knowledge",
    "uses_knowledge": "Uses Knowledge",
    "uses knowledge": "Uses Knowledge",
}


def get_task(dataset: str) -> TaskSpec:
    key = _DATASET_ALIASES.get(dataset.strip().lower(), dataset.strip().lower())
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown dataset {dataset!r}. Available datasets: {available}") from exc


def canonicalize_dimension(dataset: str, dimension: str) -> str:
    key = dimension.strip().lower().replace("-", "_")
    if dataset == "topical_chat":
        label = _TOPICAL_DIMENSION_TO_USR_LABEL.get(key.replace("_", " "))
        if label is None:
            label = _TOPICAL_DIMENSION_TO_USR_LABEL.get(key)
        if label == "Natural":
            return "naturalness"
        if label == "Maintains Context":
            return "coherence"
        if label == "Engaging":
            return "engagingness"
        if label == "Uses Knowledge":
            return "groundedness"
    if dataset == "summeval" and key in SUMMEVAL_DIMENSIONS:
        return key
    if dataset in {"qags_cnn", "qags_xsum"} and key in QAGS_DIMENSIONS:
        return key
    spec = TASK_REGISTRY.get(dataset)
    available = ", ".join(spec.dimensions) if spec else "unknown"
    raise ValueError(f"Unsupported dimension {dimension!r} for dataset {dataset!r}. Available: {available}")


def split_examples(
    examples: Iterable[EvalExample],
    *,
    train_groups: int,
    val_groups: int,
    test_groups: int,
    seed: int,
) -> tuple[list[EvalExample], list[EvalExample], list[EvalExample]]:
    """Split by group id to avoid related candidates leaking across splits."""

    rows = list(examples)
    group_ids = sorted({row.group_id for row in rows})
    required = train_groups + val_groups + test_groups
    if required > len(group_ids):
        raise ValueError(f"Requested {required} groups, but dataset has {len(group_ids)}")

    rng = random.Random(seed)
    rng.shuffle(group_ids)
    train_ids = set(group_ids[:train_groups])
    val_ids = set(group_ids[train_groups : train_groups + val_groups])
    test_ids = set(group_ids[train_groups + val_groups : required])
    return (
        [row for row in rows if row.group_id in train_ids],
        [row for row in rows if row.group_id in val_ids],
        [row for row in rows if row.group_id in test_ids],
    )


def write_split_manifest(path: Path, train: list[EvalExample], val: list[EvalExample], test: list[EvalExample]) -> None:
    payload = {
        "gepa_train": [row.example_id for row in train],
        "gepa_validation": [row.example_id for row in val],
        "final_test": [row.example_id for row in test],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_topical_chat(source: str | Path, dimension: str) -> list[EvalExample]:
    usr_label = {
        "naturalness": "Natural",
        "coherence": "Maintains Context",
        "engagingness": "Engaging",
        "groundedness": "Uses Knowledge",
    }[dimension]
    min_score, max_score = USR_LABEL_SCALES[usr_label]
    examples = []
    for row in load_usr_examples(source):
        examples.append(
            EvalExample(
                dataset="topical_chat",
                dimension=dimension,
                example_id=row.response_id,
                group_id=row.context_id,
                source_text=row.context,
                context=row.context,
                fact=row.fact,
                candidate_output=row.response,
                human_score=row.human_score(usr_label),
                min_score=min_score,
                max_score=max_score,
                system_id=row.model,
                metadata={"usr_label": usr_label, "raw_annotations": row.raw_annotations.get(usr_label, [])},
            )
        )
    return examples


def _load_summeval(source: str | Path, dimension: str) -> list[EvalExample]:
    records = _read_json(source)
    rows = _coerce_records(records)
    examples: list[EvalExample] = []
    for index, record in enumerate(rows):
        source_text = _first_str(record, ("source", "source_article", "article", "document", "text"))
        candidate = _first_str(record, ("summary", "candidate_summary", "system_output", "decoded", "output"))
        reference = _first_str(record, ("reference", "reference_summary", "ref_summary", "gold_summary"), default="")
        system_id = _first_str(record, ("system_id", "model_id", "model", "system"), default="")
        score = _extract_score(record, dimension)
        group_id = _first_str(record, ("doc_id", "document_id", "source_id", "id"), default=f"summeval_{index:05d}")
        example_id = _first_str(
            record,
            ("example_id", "summary_id", "id"),
            default=f"{group_id}_{system_id or index}_{dimension}",
        )
        examples.append(
            EvalExample(
                dataset="summeval",
                dimension=dimension,
                example_id=str(example_id),
                group_id=str(group_id),
                source_text=source_text,
                candidate_output=candidate,
                reference=reference,
                human_score=score,
                min_score=1,
                max_score=5,
                system_id=system_id,
                metadata={key: value for key, value in record.items() if key not in {"source", "summary"}},
            )
        )
    return examples


def _load_qags(source: str | Path, dimension: str) -> list[EvalExample]:
    del dimension
    records = _read_json(source)
    rows = _coerce_records(records)
    examples: list[EvalExample] = []
    for index, record in enumerate(rows):
        source_text = _first_str(record, ("source", "source_article", "article", "document", "text"))
        candidate = _first_str(record, ("summary", "candidate_summary", "system_output", "decoded", "output"))
        reference = _first_str(record, ("reference", "reference_summary", "ref_summary", "gold_summary"), default="")
        system_id = _first_str(record, ("system_id", "model_id", "model", "system"), default="")
        score = _extract_score(record, "consistency")
        group_id = _first_str(record, ("doc_id", "document_id", "source_id", "id"), default=f"qags_{index:05d}")
        example_id = _first_str(
            record,
            ("example_id", "summary_id", "id"),
            default=f"{group_id}_{system_id or index}_consistency",
        )
        examples.append(
            EvalExample(
                dataset="qags_cnn" if "cnn" in str(source).lower() else "qags_xsum",
                dimension="consistency",
                example_id=str(example_id),
                group_id=str(group_id),
                source_text=source_text,
                candidate_output=candidate,
                reference=reference,
                human_score=score,
                min_score=1,
                max_score=5,
                system_id=system_id,
                metadata={key: value for key, value in record.items() if key not in {"source", "summary"}},
            )
        )
    return examples


def _read_json(source: str | Path) -> Any:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        with urllib.request.urlopen(source_text, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    with Path(source).open(encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_records(records: Any) -> list[dict[str, Any]]:
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    if isinstance(records, dict):
        for key in ("data", "examples", "records", "summaries"):
            value = records.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Expected dataset JSON to be a list of objects or a dict containing a records list")


def _first_str(record: dict[str, Any], keys: tuple[str, ...], *, default: str | None = None) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    if default is not None:
        return default
    raise ValueError(f"Record is missing one of string fields: {', '.join(keys)}")


def _extract_score(record: dict[str, Any], dimension: str) -> float:
    candidates = (
        dimension,
        dimension.lower(),
        dimension.capitalize(),
        f"{dimension}_score",
        f"{dimension.lower()}_score",
        "human_score",
        "score",
    )
    for key in candidates:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
            return sum(float(item) for item in value) / len(value)
        if isinstance(value, dict):
            numeric_values = [float(item) for item in value.values() if isinstance(item, (int, float))]
            if numeric_values:
                return sum(numeric_values) / len(numeric_values)
    raise ValueError(f"Record is missing a numeric human score for dimension {dimension!r}")


TASK_REGISTRY.update(
    {
        "topical_chat": TaskSpec(
            dataset="topical_chat",
            dimensions=TOPICAL_CHAT_DIMENSIONS,
            default_source=DEFAULT_USR_URL,
            default_dimension="engagingness",
            task_type="dialogue",
            loader=_load_topical_chat,
        ),
        "summeval": TaskSpec(
            dataset="summeval",
            dimensions=SUMMEVAL_DIMENSIONS,
            default_source=SUMMEVAL_URL,
            default_dimension="consistency",
            task_type="summarization",
            loader=_load_summeval,
        ),
        "qags_cnn": TaskSpec(
            dataset="qags_cnn",
            dimensions=QAGS_DIMENSIONS,
            default_source=QAGS_CNN_URL,
            default_dimension="consistency",
            task_type="factuality",
            loader=_load_qags,
        ),
        "qags_xsum": TaskSpec(
            dataset="qags_xsum",
            dimensions=QAGS_DIMENSIONS,
            default_source=QAGS_XSUM_URL,
            default_dimension="consistency",
            task_type="factuality",
            loader=_load_qags,
        ),
    }
)
