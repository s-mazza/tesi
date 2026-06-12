#!/usr/bin/env python3
"""Build NLA precompute files with isolated experimental token strategies."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geval_gepa.nla_precompute import (
    ActivationRow,
    TransformersNlaVerbalizer,
    _hidden_states_only,
    _token_index_for_char_span,
    iter_verbalization_rows,
    read_jsonl,
    stable_fake_vector,
    write_jsonl,
)


WORD_RE = re.compile(r"\b[\w'-]{4,}\b")
WEAK_TOKENS = {
    "from",
    "that",
    "this",
    "with",
    "there",
    "they",
    "their",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "about",
    "like",
    "just",
    "yeah",
    "hello",
    "said",
    "think",
    "recently",
}


@dataclass(frozen=True)
class ExperimentalTokenTarget:
    token_position: str
    token_text: str
    char_start: int
    char_end: int
    output_example_id: str | None = None
    shared_group_feedback: bool = False


STRATEGIES: dict[str, dict[str, Any]] = {
    "candidate_content_6": {
        "budgets": {"candidate": 6, "source": 0, "reference": 0},
        "avoid_first": True,
        "filter_weak": True,
    },
    "candidate_content_10": {
        "budgets": {"candidate": 10, "source": 0, "reference": 0},
        "avoid_first": True,
        "filter_weak": True,
    },
    "candidate_source_content_8": {
        "budgets": {"candidate": 6, "source": 2, "reference": 0},
        "avoid_first": True,
        "filter_weak": True,
    },
    "hybrid_context_dedup_6": {
        "budgets": {"candidate": 4, "source": 1, "reference": 1},
        "avoid_first": True,
        "filter_weak": True,
        "dedupe_context_fields": {"source", "reference"},
    },
    "hybrid_context_dedup_8": {
        "budgets": {"candidate": 6, "source": 1, "reference": 1},
        "avoid_first": True,
        "filter_weak": True,
        "dedupe_context_fields": {"source", "reference"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    parser.add_argument("--activation-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--nla-checkpoint", default="kitft/nla-qwen2.5-7b-L20-av")
    parser.add_argument("--nla-root", type=Path, default=Path("natural_language_autoencoders"))
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--activation-dtype", default="float16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--nla-dtype", default="float16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--injection-scale", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-d-model", type=int, default=3584)
    return parser.parse_args()


class ExperimentalTokenSelector:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.spec = STRATEGIES[strategy]
        self._seen_context_fields: set[tuple[str, str]] = set()

    def select(self, manifest_row: dict[str, Any], rendered_prompt: str) -> list[ExperimentalTokenTarget]:
        targets: list[ExperimentalTokenTarget] = []
        group_id = str(manifest_row.get("group_id") or "")
        fields = {
            "candidate": str(manifest_row.get("candidate_output") or ""),
            "source": str(manifest_row.get("source_text") or ""),
            "reference": str(manifest_row.get("fact") or manifest_row.get("reference") or ""),
        }
        budgets = self.spec["budgets"]
        for field_name in ("candidate", "source", "reference"):
            budget = int(budgets.get(field_name, 0))
            if budget <= 0:
                continue
            is_context_shared = field_name in set(self.spec.get("dedupe_context_fields", set()))
            if is_context_shared:
                context_key = (group_id, field_name)
                if context_key in self._seen_context_fields:
                    continue
            field_text = fields[field_name]
            if not field_text or field_text == "_nofact":
                continue
            base_offset = rendered_prompt.find(field_text)
            if base_offset < 0:
                continue
            if is_context_shared:
                self._seen_context_fields.add(context_key)
            for label, word, start, end in sample_words(
                field_text,
                budget,
                avoid_first=bool(self.spec.get("avoid_first")),
                filter_weak=bool(self.spec.get("filter_weak")),
            ):
                targets.append(
                    ExperimentalTokenTarget(
                        token_position=(
                            f"context_{field_name}_{label}" if is_context_shared else f"{field_name}_{label}"
                        ),
                        token_text=word,
                        char_start=base_offset + start,
                        char_end=base_offset + end,
                        output_example_id=f"__group__:{group_id}" if is_context_shared else None,
                        shared_group_feedback=is_context_shared,
                    )
                )
        return targets


def sample_words(text: str, limit: int, *, avoid_first: bool, filter_weak: bool) -> list[tuple[str, str, int, int]]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []
    indexes = [len(matches) // 2, len(matches) - 1]
    if not avoid_first:
        indexes.append(0)
    if limit > 3:
        step = max(1, len(matches) // limit)
        indexes.extend(range(0, len(matches), step))
    selected: list[int] = []
    for index in indexes:
        token = matches[index].group(0).lower()
        if avoid_first and index == 0 and len(matches) > 2:
            continue
        if filter_weak and token in WEAK_TOKENS:
            continue
        if index not in selected:
            selected.append(index)
        if len(selected) >= limit:
            break
    return [
        (label_for_index(index, matches), matches[index].group(0), matches[index].start(), matches[index].end())
        for index in selected
    ]


def label_for_index(index: int, matches: list[re.Match[str]]) -> str:
    if index == len(matches) // 2:
        return "middle"
    if index == len(matches) - 1:
        return "last"
    if index == 0:
        return "first"
    return f"content_{index}"


class ExperimentalQwenActivationExtractor:
    def __init__(
        self,
        *,
        model_id: str,
        layer: int,
        dtype_name: str,
        device_map: str,
        trust_remote_code: bool,
        use_chat_template: bool,
        strategy: str,
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
        self.selector = ExperimentalTokenSelector(strategy)

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def extract(self, manifest_rows: list[dict[str, Any]], *, limit: int | None = None) -> list[ActivationRow]:
        rows: list[ActivationRow] = []
        selected = manifest_rows[:limit]
        for index, manifest_row in enumerate(selected, start=1):
            rows.extend(self.extract_one(manifest_row))
            print(f"Extracted experimental NLA activations for {index}/{len(selected)} manifest rows", flush=True)
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
        output: list[ActivationRow] = []
        for target in targets:
            token_index = _token_index_for_char_span(offsets, target.char_start, target.char_end)
            if token_index is None:
                continue
            manifest_for_output = dict(manifest_row)
            if target.output_example_id:
                manifest_for_output["base_example_id"] = manifest_row.get("example_id", "")
                manifest_for_output["example_id"] = target.output_example_id
                manifest_for_output["shared_group_feedback"] = target.shared_group_feedback
            output.append(
                ActivationRow(
                    manifest_row=manifest_for_output,
                    token_position=f"experimental_{self.selector.strategy}_{target.token_position}",
                    token_text=target.token_text,
                    token_index=token_index,
                    activation_vector=hidden_states[self.layer][0, token_index].float().cpu().tolist(),
                    model_id=self.model_id,
                    layer=self.layer,
                    token_status="ok",
                )
            )
        return output

    def _render_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def fake_activation_rows(
    manifest_rows: list[dict[str, Any]],
    *,
    model_id: str,
    layer: int,
    strategy: str,
    d_model: int,
    limit: int | None,
) -> list[ActivationRow]:
    selector = ExperimentalTokenSelector(strategy)
    rows: list[ActivationRow] = []
    for item in manifest_rows[:limit]:
        for index, target in enumerate(selector.select(item, str(item["prompt"]))):
            key = f"{item['example_id']}::{strategy}::{target.token_position}"
            manifest_for_output = dict(item)
            if target.output_example_id:
                manifest_for_output["base_example_id"] = item.get("example_id", "")
                manifest_for_output["example_id"] = target.output_example_id
                manifest_for_output["shared_group_feedback"] = target.shared_group_feedback
            rows.append(
                ActivationRow(
                    manifest_row=manifest_for_output,
                    token_position=f"experimental_{strategy}_{target.token_position}",
                    token_text=target.token_text,
                    token_index=index,
                    activation_vector=stable_fake_vector(key, d_model),
                    model_id=model_id,
                    layer=layer,
                    token_status="fake",
                )
            )
    return rows


def main() -> int:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    if args.dry_run:
        activation_rows = fake_activation_rows(
            manifest_rows,
            model_id=args.activation_model,
            layer=args.layer,
            strategy=args.strategy,
            d_model=args.fake_d_model,
            limit=args.limit,
        )
    else:
        extractor = ExperimentalQwenActivationExtractor(
            model_id=args.activation_model,
            layer=args.layer,
            dtype_name=args.activation_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            use_chat_template=not args.no_chat_template,
            strategy=args.strategy,
        )
        try:
            activation_rows = extractor.extract(manifest_rows, limit=args.limit)
        finally:
            extractor.close()
            gc.collect()
    print(f"Prepared {len(activation_rows)} experimental NLA activation rows.", flush=True)
    count = write_jsonl(
        args.output,
        iter_verbalization_rows(
            activation_rows,
            checkpoint=args.nla_checkpoint,
            backend="transformers",
            nla_root=args.nla_root,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            injection_scale=args.injection_scale,
            dtype_name=args.nla_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        ),
    )
    print(f"Wrote {count} experimental NLA precomputed feedback rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
