"""Precompute NLA verbalization rows for GEPA feedback."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
WORD_RE = re.compile(r"\b[\w'-]{4,}\b")


@dataclass(frozen=True)
class TokenTarget:
    token_position: str
    token_text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ActivationRow:
    manifest_row: dict[str, Any]
    token_position: str
    token_text: str
    token_index: int
    activation_vector: list[float]
    model_id: str
    layer: int
    token_status: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            count += 1
    return count


def parse_explanation(text: str) -> tuple[str, str]:
    match = EXPLANATION_RE.search(text)
    if match:
        return match.group(1).strip(), "ok"
    stripped = text.strip()
    if stripped.startswith("<explanation>"):
        return stripped.removeprefix("<explanation>").strip(), "partial_tags"
    if stripped.endswith("</explanation>"):
        return stripped.removesuffix("</explanation>").strip(), "partial_tags"
    return stripped, "missing_tags" if stripped else "empty"


class SemanticTokenSelector:
    """Select multiple content-bearing prompt tokens instead of only the final token."""

    def __init__(self, *, max_tokens_per_example: int) -> None:
        if max_tokens_per_example < 1:
            raise ValueError("max_tokens_per_example must be >= 1")
        self.max_tokens_per_example = max_tokens_per_example

    def select(self, manifest_row: dict[str, Any], rendered_prompt: str) -> list[TokenTarget]:
        targets: list[TokenTarget] = []
        field_specs = (
            ("candidate", str(manifest_row.get("candidate_output") or "")),
            ("source", str(manifest_row.get("source_text") or "")),
            ("reference", str(manifest_row.get("fact") or manifest_row.get("reference") or "")),
        )
        budgets = _semantic_token_budgets(self.max_tokens_per_example)
        for field_name, field_text in field_specs:
            if len(targets) >= self.max_tokens_per_example:
                break
            field_budget = budgets.get(field_name, 0)
            if field_budget <= 0:
                continue
            if not field_text or field_text == "_nofact":
                continue
            base_offset = rendered_prompt.find(field_text)
            if base_offset < 0:
                continue
            for label, word, start, end in self._sample_words(field_text, limit=field_budget):
                targets.append(
                    TokenTarget(
                        token_position=f"{field_name}_{label}",
                        token_text=word,
                        char_start=base_offset + start,
                        char_end=base_offset + end,
                    )
                )
                if len(targets) >= self.max_tokens_per_example:
                    break

        if targets:
            return targets

        fallback_words = list(WORD_RE.finditer(rendered_prompt))
        for label, match in self._sample_matches(fallback_words, limit=self.max_tokens_per_example):
            targets.append(
                TokenTarget(
                    token_position=f"prompt_{label}",
                    token_text=match.group(0),
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
        return targets

    def _sample_words(self, text: str, *, limit: int) -> list[tuple[str, str, int, int]]:
        matches = list(WORD_RE.finditer(text))
        return [(label, match.group(0), match.start(), match.end()) for label, match in self._sample_matches(matches, limit=limit)]

    def _sample_matches(self, matches: list[re.Match[str]], *, limit: int) -> list[tuple[str, re.Match[str]]]:
        if not matches:
            return []
        indexes = [len(matches) // 2, len(matches) - 1, 0]
        if limit > 3:
            step = max(1, len(matches) // limit)
            indexes.extend(range(0, len(matches), step))
        selected: list[int] = []
        for index in indexes:
            if index not in selected:
                selected.append(index)
            if len(selected) >= limit:
                break
        labels = ("middle", "last", "first", "extra0", "extra1", "extra2", "extra3", "extra4")
        return [(labels[pos] if pos < len(labels) else f"extra{pos}", matches[index]) for pos, index in enumerate(selected)]


def _semantic_token_budgets(max_tokens_per_example: int) -> dict[str, int]:
    candidate = max(1, (max_tokens_per_example + 1) // 2)
    remaining = max_tokens_per_example - candidate
    source = 1 if remaining > 0 else 0
    reference = max(0, remaining - source)
    return {"candidate": candidate, "source": source, "reference": reference}


class QwenActivationExtractor:
    """Extract base-model residual stream vectors at selected prompt tokens."""

    def __init__(
        self,
        *,
        model_id: str,
        layer: int,
        dtype_name: str,
        device_map: str,
        trust_remote_code: bool,
        use_chat_template: bool,
        max_tokens_per_example: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.layer = layer
        self.use_chat_template = use_chat_template
        dtype = getattr(torch, dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.selector = SemanticTokenSelector(max_tokens_per_example=max_tokens_per_example)

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def extract(self, manifest_rows: list[dict[str, Any]], *, limit: int | None = None) -> list[ActivationRow]:
        rows: list[ActivationRow] = []
        for index, manifest_row in enumerate(manifest_rows[:limit], start=1):
            rows.extend(self.extract_one(manifest_row))
            print(f"Extracted NLA activations for {index}/{len(manifest_rows[:limit])} manifest rows", flush=True)
        return rows

    def extract_one(self, manifest_row: dict[str, Any]) -> list[ActivationRow]:
        prompt = str(manifest_row["prompt"])
        rendered_prompt = self._render_prompt(prompt)
        encoded = self.tokenizer(
            rendered_prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in encoded.items()}
        targets = self.selector.select(manifest_row, rendered_prompt)

        with self.torch.inference_mode():
            hidden_states = _hidden_states_only(self.model, **inputs)
        selected_rows: list[ActivationRow] = []
        for target in targets:
            token_index = _token_index_for_char_span(offsets, target.char_start, target.char_end)
            token_status = "ok" if token_index is not None else "no_token_overlap"
            if token_index is None:
                continue
            vector = hidden_states[self.layer][0, token_index].float().cpu().tolist()
            selected_rows.append(
                ActivationRow(
                    manifest_row=manifest_row,
                    token_position=target.token_position,
                    token_text=target.token_text,
                    token_index=token_index,
                    activation_vector=vector,
                    model_id=self.model_id,
                    layer=self.layer,
                    token_status=token_status,
                )
            )
        return selected_rows

    def _render_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def _hidden_states_only(model: Any, **inputs: Any) -> Any:
    backbone = getattr(model, "model", None)
    if backbone is not None:
        return backbone(output_hidden_states=True, use_cache=False, **inputs).hidden_states
    return model(output_hidden_states=True, use_cache=False, **inputs).hidden_states


def _token_index_for_char_span(offsets: list[list[int]], start: int, end: int) -> int | None:
    candidates = [
        index
        for index, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > start and tok_start < end and tok_end > tok_start
    ]
    if not candidates:
        return None
    return candidates[-1]


class TransformersNlaVerbalizer:
    """Run the NLA activation-verbalizer actor without an SGLang sidecar."""

    def __init__(
        self,
        *,
        checkpoint: str,
        nla_root: Path,
        dtype_name: str,
        device_map: str,
        trust_remote_code: bool,
        injection_scale: float | None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        checkpoint_path = resolve_checkpoint(checkpoint)
        sys.path.insert(0, str(nla_root))
        from nla_inference import NLAClient

        self.embed_client = NLAClient(
            checkpoint_path,
            sglang_url="http://127.0.0.1:9",
            injection_scale_override=injection_scale,
        )
        dtype = getattr(torch, dtype_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.torch = torch
        self.tokenizer = self.embed_client.tokenizer
        self.device = next(self.model.parameters()).device
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path

    def generate(self, activation: list[float], *, temperature: float, max_new_tokens: int) -> str:
        embeds_np, prompt_len = self.embed_client._build_embeds(
            self.torch.as_tensor(activation, dtype=self.torch.float32),
            None,
        )
        embeds = self.torch.from_numpy(embeds_np).unsqueeze(0).to(self.device, dtype=self.dtype)
        attention_mask = self.torch.ones(embeds.shape[:2], dtype=self.torch.long, device=self.device)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        generation_args: dict[str, Any] = {
            "inputs_embeds": embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature

        with self.torch.inference_mode():
            generated = self.model.generate(**generation_args)
        token_ids = generated[0].tolist()
        if len(token_ids) > prompt_len:
            token_ids = token_ids[prompt_len:]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def resolve_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint)
    if (path / "nla_meta.yaml").exists():
        return str(path)

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=checkpoint, local_files_only=True)


def stable_fake_vector(key: str, d_model: int) -> list[float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    values = []
    while len(values) < d_model:
        for byte in digest:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) >= d_model:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def fake_activation_rows(
    manifest_rows: list[dict[str, Any]],
    *,
    model_id: str,
    layer: int,
    max_tokens_per_example: int,
    d_model: int,
    limit: int | None,
) -> list[ActivationRow]:
    selector = SemanticTokenSelector(max_tokens_per_example=max_tokens_per_example)
    rows: list[ActivationRow] = []
    for item in manifest_rows[:limit]:
        prompt = str(item["prompt"])
        targets = selector.select(item, prompt)
        for index, target in enumerate(targets):
            key = f"{item['example_id']}::{target.token_position}"
            rows.append(
                ActivationRow(
                    manifest_row=item,
                    token_position=target.token_position,
                    token_text=target.token_text,
                    token_index=index,
                    activation_vector=stable_fake_vector(key, d_model),
                    model_id=model_id,
                    layer=layer,
                    token_status="fake",
                )
            )
    return rows


def iter_verbalization_rows(
    activation_rows: list[ActivationRow],
    *,
    checkpoint: str,
    backend: str,
    nla_root: Path,
    dry_run: bool,
    temperature: float,
    max_new_tokens: int,
    injection_scale: float | None,
    dtype_name: str,
    device_map: str,
    trust_remote_code: bool,
) -> Iterator[dict[str, Any]]:
    verbalizer = None
    if not dry_run:
        if backend != "transformers":
            raise ValueError(f"Unsupported NLA verbalization backend: {backend}")
        verbalizer = TransformersNlaVerbalizer(
            checkpoint=checkpoint,
            nla_root=nla_root,
            dtype_name=dtype_name,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            injection_scale=injection_scale,
        )

    for index, row in enumerate(activation_rows, start=1):
        if dry_run:
            raw_generation = (
                "<explanation>"
                f"DRY RUN ONLY: placeholder verbalization for {row.token_text!r} "
                f"at {row.token_position}."
                "</explanation>"
            )
        else:
            assert verbalizer is not None
            raw_generation = verbalizer.generate(
                row.activation_vector,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        explanation, parse_status = parse_explanation(raw_generation)
        manifest = row.manifest_row
        activation_stats = _activation_summary(row.activation_vector)
        yield {
            "dataset": manifest.get("dataset", ""),
            "dimension": manifest.get("dimension", ""),
            "example_id": manifest.get("example_id", ""),
            "base_example_id": manifest.get("base_example_id", ""),
            "group_id": manifest.get("group_id", ""),
            "human_score": manifest.get("human_score"),
            "shared_group_feedback": manifest.get("shared_group_feedback", False),
            "model_id": row.model_id,
            "layer": row.layer,
            "token_position": row.token_position,
            "token_text": row.token_text,
            "token_index": row.token_index,
            "token_status": row.token_status,
            "nla_model_id": checkpoint,
            "nla_backend": "dry_run" if dry_run else backend,
            "raw_generation": raw_generation,
            "verbalization": explanation,
            "explanation": explanation,
            "parse_status": parse_status,
            **activation_stats,
        }
        print(f"Verbalized NLA activation {index}/{len(activation_rows)}", flush=True)


def _activation_summary(vector: list[float]) -> dict[str, float | int]:
    """Persist compact vector diagnostics without storing raw activations."""

    if not vector:
        return {
            "activation_dim": 0,
            "activation_l2_norm": 0.0,
            "activation_mean": 0.0,
            "activation_std": 0.0,
            "activation_min": 0.0,
            "activation_max": 0.0,
            "activation_abs_mean": 0.0,
        }
    values = [float(item) for item in vector]
    dim = len(values)
    mean = sum(values) / dim
    variance = sum((item - mean) ** 2 for item in values) / dim
    return {
        "activation_dim": dim,
        "activation_l2_norm": math.sqrt(sum(item * item for item in values)),
        "activation_mean": mean,
        "activation_std": math.sqrt(variance),
        "activation_min": min(values),
        "activation_max": max(values),
        "activation_abs_mean": sum(abs(item) for item in values) / dim,
    }
