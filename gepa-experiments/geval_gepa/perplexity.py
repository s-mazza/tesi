"""Perplexity feedback helpers for GEPA optimization."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PerplexityResult:
    mean_nll: float
    perplexity: float
    token_count: int


class VllmPerplexityScorer:
    """Score candidate responses with vLLM prompt logprobs."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        tokenizer_model: str,
        hf_home: str,
        prompt_logprobs: int = 20,
        timeout_seconds: float = 120.0,
    ) -> None:
        from transformers import AutoTokenizer

        self.api_base = api_base.rstrip("/")
        self.model = model
        self.prompt_logprobs = prompt_logprobs
        self.timeout_seconds = timeout_seconds
        self.tokenizer = AutoTokenizer.from_pretrained(
            _resolve_tokenizer_path(tokenizer_model, Path(hf_home)),
            local_files_only=True,
            trust_remote_code=True,
        )
        self._cache: dict[tuple[str, str, str], PerplexityResult] = {}
        self._lock = threading.Lock()

    def score_example(self, example: Any) -> PerplexityResult:
        context = str(getattr(example, "context", ""))
        fact = str(getattr(example, "fact", ""))
        response = str(getattr(example, "response", ""))
        cache_key = (
            str(getattr(example, "context_id", "")),
            str(getattr(example, "response_id", "")),
            response,
        )
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self.score_response(context=context, fact=fact, response=response)
        with self._lock:
            self._cache[cache_key] = result
        return result

    def score_response(self, *, context: str, fact: str, response: str) -> PerplexityResult:
        prefix = format_perplexity_prefix(context=context, fact=fact)
        full_text = prefix + response
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        response_ids = full_ids[len(prefix_ids) :]
        if not response_ids:
            raise ValueError("Response tokenizes to zero tokens.")

        payload = {
            "model": self.model,
            "prompt": full_text,
            "max_tokens": 0,
            "temperature": 0,
            "prompt_logprobs": self.prompt_logprobs,
        }
        data = _post_json(f"{self.api_base}/completions", payload, timeout_seconds=self.timeout_seconds)
        prompt_logprobs = _extract_prompt_logprobs(data)
        if len(prompt_logprobs) < len(full_ids):
            raise ValueError(
                f"vLLM returned {len(prompt_logprobs)} prompt logprob entries for {len(full_ids)} prompt tokens."
            )

        nll = 0.0
        for index, token_id in enumerate(response_ids, start=len(prefix_ids)):
            token_logprob = _find_token_logprob(prompt_logprobs[index], token_id)
            if token_logprob is None:
                raise ValueError(
                    f"Token id {token_id} not found in prompt_logprobs at position {index}. "
                    f"Increase PERPLEXITY_PROMPT_LOGPROBS above {self.prompt_logprobs} or check vLLM logprob support."
                )
            nll += -token_logprob

        mean_nll = nll / len(response_ids)
        return PerplexityResult(
            mean_nll=mean_nll,
            perplexity=math.exp(mean_nll),
            token_count=len(response_ids),
        )


def format_perplexity_prefix(*, context: str, fact: str) -> str:
    return "\n".join(
        [
            "Dialogue history:",
            context.strip(),
            "",
            "Relevant fact:",
            fact.strip() or "_nofact",
            "",
            "Candidate response:",
            "",
        ]
    )


def format_perplexity_feedback(result: PerplexityResult) -> str:
    return (
        "Perplexity signals: "
        f"response_mean_nll={result.mean_nll:.4f}; "
        f"response_perplexity={result.perplexity:.4f}; "
        f"response_token_count={result.token_count}."
    )


def _resolve_tokenizer_path(model: str, hf_home: Path) -> str:
    model_cache = hf_home / "hub" / f"models--{model.replace('/', '--')}"
    snapshots_dir = model_cache / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return model


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM perplexity request failed with HTTP {exc.code}: {body}") from exc


def _extract_prompt_logprobs(data: dict[str, Any]) -> list[Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"vLLM response has no choices: {data}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError(f"vLLM choice is not an object: {choice}")
    prompt_logprobs = choice.get("prompt_logprobs")
    if prompt_logprobs is None and isinstance(choice.get("logprobs"), dict):
        prompt_logprobs = choice["logprobs"].get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list):
        raise ValueError(f"vLLM response has no prompt_logprobs list: {choice}")
    return prompt_logprobs


def _find_token_logprob(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        for key, value in entry.items():
            if _key_matches_token_id(key, token_id):
                return _coerce_logprob(value)
        if _key_matches_token_id(entry.get("token_id"), token_id):
            return _coerce_logprob(entry)
    if isinstance(entry, list):
        for item in entry:
            if isinstance(item, dict) and _key_matches_token_id(item.get("token_id"), token_id):
                return _coerce_logprob(item)
    return None


def _key_matches_token_id(key: Any, token_id: int) -> bool:
    if key == token_id:
        return True
    if isinstance(key, str):
        return key == str(token_id)
    return False


def _coerce_logprob(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        logprob = value.get("logprob")
        if isinstance(logprob, (int, float)):
            return float(logprob)
    if hasattr(value, "logprob"):
        logprob = getattr(value, "logprob")
        if isinstance(logprob, (int, float)):
            return float(logprob)
    return None
