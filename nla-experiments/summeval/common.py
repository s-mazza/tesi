from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get("NLA_ARTIFACT_ROOT", REPO_ROOT / "nla-artifacts")
)

SUMMEVAL_DATA_URL = (
    "https://raw.githubusercontent.com/nlpyang/geval/main/data/summeval.json"
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NLA_AV_MODEL_ID = "kitft/nla-qwen2.5-7b-L20-av"
NLA_LAYER = 20
NLA_D_MODEL = 3584

DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
SCORE_RE = re.compile(r"(?<!\d)([1-5](?:\.\d+)?)(?!\d)")


def artifact_path(*parts: str) -> Path:
    return DEFAULT_ARTIFACT_ROOT.joinpath(*parts)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return sha256_text(text)[:n]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def extract_score_text(text: str) -> str | None:
    match = SCORE_RE.search(text)
    return match.group(1) if match else None


def extract_explanation(raw_text: str) -> tuple[str, str]:
    match = EXPLANATION_RE.search(raw_text)
    if match:
        return match.group(1).strip(), "ok"
    stripped = raw_text.strip()
    if stripped.startswith("<explanation>"):
        return stripped.removeprefix("<explanation>").strip(), "partial_tags"
    return stripped, "missing_tags" if stripped else "empty"


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    cjk = 0
    for char in text:
        code = ord(char)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            cjk += 1
    return cjk / max(len(text), 1)
