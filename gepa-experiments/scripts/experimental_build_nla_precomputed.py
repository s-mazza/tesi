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
    "candidate_first_1": {
        "positions": {"candidate": ["first"]},
    },
    "candidate_middle_1": {
        "positions": {"candidate": ["middle"]},
    },
    "candidate_last_1": {
        "positions": {"candidate": ["last"]},
    },
    "candidate_fml_3": {
        "positions": {"candidate": ["first", "middle", "last"]},
    },
    "candidate_quintile_5": {
        "positions": {"candidate": ["first", "q25", "middle", "q75", "last"]},
        "filter_weak": True,
    },
    "candidate_even_8": {
        "positions": {"candidate": ["even8"]},
        "filter_weak": True,
    },
    "source_fml_3": {
        "positions": {"source": ["first", "middle", "last"]},
        "filter_weak": True,
    },
    "reference_fml_3": {
        "positions": {"reference": ["first", "middle", "last"]},
        "filter_weak": True,
    },
    "balanced_fml_9": {
        "positions": {
            "candidate": ["first", "middle", "last"],
            "source": ["first", "middle", "last"],
            "reference": ["first", "middle", "last"],
        },
        "filter_weak": True,
    },
    "prompt_tail_6": {
        "positions": {"prompt": ["tail6"]},
        "filter_weak": True,
    },
    "evaluation_tail_3": {
        "positions": {"evaluation": ["first", "middle", "last"]},
    },
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
        fields = field_texts_and_offsets(manifest_row, rendered_prompt)
        field_positions = self.spec.get("positions")
        if field_positions:
            field_items = [(field_name, list(positions)) for field_name, positions in field_positions.items()]
        else:
            budgets = self.spec["budgets"]
            field_items = [
                (field_name, int(budgets.get(field_name, 0)))
                for field_name in ("candidate", "source", "reference")
                if int(budgets.get(field_name, 0)) > 0
            ]
        for field_name, selection_spec in field_items:
            is_context_shared = field_name in set(self.spec.get("dedupe_context_fields", set()))
            if is_context_shared:
                context_key = (group_id, field_name)
                if context_key in self._seen_context_fields:
                    continue
            field_text, base_offset = fields.get(field_name, ("", -1))
            if not field_text or field_text == "_nofact":
                continue
            if base_offset < 0:
                continue
            if is_context_shared:
                self._seen_context_fields.add(context_key)
            positions = selection_spec if isinstance(selection_spec, list) else None
            budget = int(selection_spec) if not isinstance(selection_spec, list) else len(selection_spec)
            for label, word, start, end in sample_words(
                field_text,
                budget,
                avoid_first=bool(self.spec.get("avoid_first")),
                filter_weak=bool(self.spec.get("filter_weak")),
                positions=positions,
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


def field_texts_and_offsets(manifest_row: dict[str, Any], rendered_prompt: str) -> dict[str, tuple[str, int]]:
    candidate = str(manifest_row.get("candidate_output") or "")
    source = str(manifest_row.get("source_text") or "")
    reference = str(manifest_row.get("fact") or manifest_row.get("reference") or "")
    marker = "Evaluation form:"
    marker_start = rendered_prompt.find(marker)
    return {
        "candidate": (candidate, rendered_prompt.find(candidate) if candidate else -1),
        "source": (source, rendered_prompt.find(source) if source else -1),
        "reference": (reference, rendered_prompt.find(reference) if reference else -1),
        "prompt": (rendered_prompt, 0),
        "evaluation": (
            rendered_prompt[marker_start:] if marker_start >= 0 else "",
            marker_start,
        ),
    }


def sample_words(
    text: str,
    limit: int,
    *,
    avoid_first: bool,
    filter_weak: bool,
    positions: list[str] | None = None,
) -> list[tuple[str, str, int, int]]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []
    labeled_indexes = (
        position_labeled_indexes(matches, positions)
        if positions
        else default_labeled_indexes(matches, limit=limit, avoid_first=avoid_first)
    )
    selected: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, index in labeled_indexes:
        token = matches[index].group(0).lower()
        if avoid_first and index == 0 and len(matches) > 2:
            continue
        if filter_weak and token in WEAK_TOKENS:
            continue
        if index not in seen:
            selected.append((label, index))
            seen.add(index)
        if len(selected) >= limit and not positions:
            break
    return [
        (label, matches[index].group(0), matches[index].start(), matches[index].end())
        for label, index in selected
    ]


def default_labeled_indexes(
    matches: list[re.Match[str]],
    *,
    limit: int,
    avoid_first: bool,
) -> list[tuple[str, int]]:
    indexes = [len(matches) // 2, len(matches) - 1]
    if not avoid_first:
        indexes.append(0)
    if limit > 3:
        step = max(1, len(matches) // limit)
        indexes.extend(range(0, len(matches), step))
    return [(label_for_index(index, matches), index) for index in indexes]


def position_labeled_indexes(matches: list[re.Match[str]], positions: list[str] | None) -> list[tuple[str, int]]:
    output: list[tuple[str, int]] = []
    if not positions:
        return output
    for position in positions:
        if position.startswith("even"):
            count = int(position[len("even") :] or "0")
            output.extend((f"even{item + 1}_of_{count}", index) for item, index in enumerate(even_indexes(len(matches), count)))
        elif position.startswith("tail"):
            count = int(position[len("tail") :] or "0")
            tail = list(range(max(0, len(matches) - count), len(matches)))
            output.extend((f"tail{item + 1}_of_{count}", index) for item, index in enumerate(tail))
        else:
            output.append((position, index_for_named_position(len(matches), position)))
    return output


def even_indexes(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count == 1:
        return [length // 2]
    return sorted({round(item * (length - 1) / (count - 1)) for item in range(count)})


def index_for_named_position(length: int, position: str) -> int:
    if position == "first":
        return 0
    if position == "middle":
        return length // 2
    if position == "last":
        return length - 1
    if position == "q25":
        return round((length - 1) * 0.25)
    if position == "q75":
        return round((length - 1) * 0.75)
    raise ValueError(f"Unsupported token position: {position}")


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
