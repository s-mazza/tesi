#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        MODEL_ID,
        NLA_D_MODEL,
        NLA_LAYER,
        artifact_path,
        extract_score_text,
        read_jsonl,
    )
except ImportError:
    from common import (
        MODEL_ID,
        NLA_D_MODEL,
        NLA_LAYER,
        artifact_path,
        extract_score_text,
        read_jsonl,
    )


def stable_fake_vector(key: str, d_model: int) -> list[float]:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(d_model).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    vector *= 125.0
    return vector.tolist()


def _records_to_parquet(records: list[dict[str, Any]], output: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output)


def render_model_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def first_numeric_token_index(tokenizer: Any, token_ids: list[int]) -> int | None:
    for idx, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if extract_score_text(token_text):
            return idx
    return None


def build_fake_records(
    manifest_rows: list[dict[str, Any]],
    *,
    token_positions: list[str],
    d_model: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in manifest_rows[:limit]:
        for token_position in token_positions:
            key = f"{item['example_id']}::{token_position}"
            records.append(
                {
                    **item,
                    "model_id": MODEL_ID,
                    "layer": NLA_LAYER,
                    "token_position": token_position,
                    "token_status": "fake",
                    "model_output": "3",
                    "score_text": "3",
                    "activation_vector": stable_fake_vector(key, d_model),
                }
            )
    return records


def build_real_records(
    manifest_rows: list[dict[str, Any]],
    *,
    model_id: str,
    layer: int,
    token_positions: list[str],
    limit: int | None,
    max_new_tokens: int,
    dtype_name: str,
    device_map: str,
    use_chat_template: bool,
    trust_remote_code: bool,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    records: list[dict[str, Any]] = []

    for item in manifest_rows[:limit]:
        rendered_prompt = render_model_prompt(tokenizer, item["prompt"], use_chat_template)
        inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
        input_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            prompt_out = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
            )
            new_token_ids = generated[0, input_len:].tolist()
            model_output = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            full_out = model(
                input_ids=generated.to(device),
                attention_mask=torch.ones_like(generated).to(device),
                output_hidden_states=True,
                use_cache=False,
            )

        score_text = extract_score_text(model_output)
        if "prompt_final" in token_positions:
            records.append(
                {
                    **item,
                    "model_id": model_id,
                    "layer": layer,
                    "token_position": "prompt_final",
                    "token_status": "ok",
                    "model_output": model_output,
                    "score_text": score_text,
                    "activation_vector": prompt_out.hidden_states[layer][0, -1].float().cpu().tolist(),
                }
            )

        if "generated_score" in token_positions and new_token_ids:
            numeric_idx = first_numeric_token_index(tokenizer, new_token_ids)
            token_status = "ok" if numeric_idx is not None else "no_numeric_token"
            generated_idx = numeric_idx if numeric_idx is not None else 0
            seq_idx = input_len + generated_idx
            records.append(
                {
                    **item,
                    "model_id": model_id,
                    "layer": layer,
                    "token_position": "generated_score",
                    "token_status": token_status,
                    "model_output": model_output,
                    "score_text": score_text,
                    "activation_vector": full_out.hidden_states[layer][0, seq_idx].float().cpu().tolist(),
                }
            )

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Qwen residual-stream activations for SummEval prompts.")
    parser.add_argument("--manifest", type=Path, default=artifact_path("summeval", "task_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=artifact_path("summeval", "activations_qwen25_7b_instruct_L20.parquet"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--layer", type=int, default=NLA_LAYER)
    parser.add_argument("--token-positions", default="prompt_final,generated_score")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--fake", action="store_true", help="Write deterministic fake vectors for CPU schema tests.")
    parser.add_argument("--d-model", type=int, default=NLA_D_MODEL)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    token_positions = [item.strip() for item in args.token_positions.split(",") if item.strip()]
    bad_positions = sorted(set(token_positions) - {"prompt_final", "generated_score"})
    if bad_positions:
        raise ValueError(f"Unsupported token positions: {bad_positions}")

    if args.fake:
        records = build_fake_records(
            manifest_rows,
            token_positions=token_positions,
            d_model=args.d_model,
            limit=args.limit,
        )
    else:
        records = build_real_records(
            manifest_rows,
            model_id=args.model_id,
            layer=args.layer,
            token_positions=token_positions,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            dtype_name=args.dtype,
            device_map=args.device_map,
            use_chat_template=not args.no_chat_template,
            trust_remote_code=args.trust_remote_code,
        )

    _records_to_parquet(records, args.output)
    print(f"Wrote {len(records)} activation rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
