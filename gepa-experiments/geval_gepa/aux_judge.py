"""Optional LLM-as-a-judge feedback for GEPA prompt proposal."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuxJudgeRecord:
    example_id: str
    group_id: str
    dataset: str
    dimension: str
    target: float
    prediction: int | None
    agreement: float
    raw_response: str
    feedback: str
    extra_feedback: str
    status: str


class AuxJudgeFeedbackProvider:
    """Use a stronger judge model only to create textual proposer feedback."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._records: dict[str, AuxJudgeRecord] = {}
        self._lock = threading.Lock()

    def feedback_for(
        self,
        *,
        example: Any,
        pred: Any,
        parsed: int | None,
        target: float,
        agreement: float,
        dimension: str,
        extra_feedback: str = "",
    ) -> str:
        example_id = str(getattr(example, "example_id", "") or getattr(example, "response_id", ""))
        with self._lock:
            cached = self._records.get(example_id)
        if cached is not None:
            return _format_feedback(cached)

        prompt = _build_aux_judge_prompt(
            example=example,
            pred=pred,
            parsed=parsed,
            target=target,
            agreement=agreement,
            dimension=dimension,
            extra_feedback=extra_feedback,
        )
        try:
            raw = self._chat_completion(prompt)
            feedback = raw.strip()
            status = "ok"
        except Exception as exc:
            raw = ""
            feedback = f"Auxiliary judge feedback unavailable: {type(exc).__name__}: {exc}"
            status = "error"

        record = AuxJudgeRecord(
            example_id=example_id,
            group_id=str(getattr(example, "group_id", "") or getattr(example, "context_id", "")),
            dataset=str(getattr(example, "dataset", "")),
            dimension=str(getattr(example, "dimension", "") or dimension),
            target=target,
            prediction=parsed,
            agreement=agreement,
            raw_response=raw,
            feedback=feedback,
            extra_feedback=extra_feedback,
            status=status,
        )
        with self._lock:
            self._records[example_id] = record
        return _format_feedback(record)

    def write_artifact(self, path: Path) -> int:
        rows = list(self._records.values())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
        return len(rows)

    def _chat_completion(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Aux judge response has no choices: {data}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError(f"Aux judge response has no message content: {data}")
        return content


def _build_aux_judge_prompt(
    *,
    example: Any,
    pred: Any,
    parsed: int | None,
    target: float,
    agreement: float,
    dimension: str,
    extra_feedback: str = "",
) -> str:
    source = str(getattr(example, "source_text", "") or getattr(example, "context", ""))
    candidate = str(getattr(example, "candidate_output", "") or getattr(example, "response", ""))
    rationale = str(getattr(pred, "rationale", ""))
    score = str(getattr(pred, "score", ""))
    extra_block = ""
    if extra_feedback.strip():
        extra_block = f"""
Additional weak proposer feedback already computed for this example:
{extra_feedback[:2400]}

Use the additional feedback only to infer a general rubric lesson. Do not copy token-level strings into the final prompt."""
    return f"""You are an auxiliary LLM-as-a-judge helping improve a G-EVAL prompt.

Do not produce a replacement score. Produce concise, generalizable feedback for a prompt proposer.

Dimension: {dimension}
Human mean score: {target:.2f}
Base judge parsed score: {parsed}
Normalized agreement: {agreement:.3f}

Source/context:
{source[:4000]}

Candidate output:
{candidate[:2000]}

Base judge rationale:
{rationale[:1200]}

Base judge raw score:
{score[:200]}
{extra_block}

Return:
1. One sentence explaining the likely judging error or confirming it is close.
2. One prompt-level lesson that generalizes beyond this example."""


def _format_feedback(record: AuxJudgeRecord) -> str:
    return (
        "Auxiliary 35B judge feedback "
        f"(status={record.status}; agreement={record.agreement:.3f}):\n{record.feedback}"
    )
