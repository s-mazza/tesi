#!/usr/bin/env python3
"""Approximate SIPIT recovery for learned soft-prompt embeddings.

This is a diagnostic handoff script, not a claim of exact SIPIT inversion. A
learned soft prompt is a continuous prefix and may not lie on the discrete token
embedding manifold. The script therefore reports both nearest-token baselines
and a bounded SIPIT-style recovery attempt against the hidden states induced by
the continuous prefix.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_sipit_paths(repo_root: Path) -> None:
    sipit_root = repo_root / "spit" / "SIPIT"
    random_prefix_dir = sipit_root / "scripts" / "random_prefix"
    sys.path.insert(0, str(sipit_root))
    sys.path.insert(0, str(random_prefix_dir))


REPO_ROOT = Path(__file__).resolve().parents[2]
add_sipit_paths(REPO_ROOT)

from sipit import RecoveryConfig, find_prompt, target_states  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding-file", default="soft_prompt_embeddings.pt")
    parser.add_argument(
        "--control-mode",
        choices=("soft_prompt", "init_prompt", "random_hard_tokens", "random_continuous"),
        default="soft_prompt",
        help="Use learned soft prompt embeddings or generate a control target.",
    )
    parser.add_argument(
        "--control-text",
        default="You are a careful, impartial evaluator. Rate the candidate output according to the rubric.",
    )
    parser.add_argument("--control-num-tokens", type=int, default=None)
    parser.add_argument("--layer-idx", type=int, default=-1)
    parser.add_argument("--precision", choices=["4", "16", "32"], default="4")
    parser.add_argument("--max-soft-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--projection-iters-base", type=int, default=50)
    parser.add_argument("--vocab-scale-factor", type=int, default=25000)
    parser.add_argument("--max-iters-per-token", type=int, default=500)
    parser.add_argument("--show-progress", action="store_false", dest="quiet_progress")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    model, tokenizer, layer_idx = load_model(args.model_name, args.precision, args.layer_idx)
    embedding_matrix = model.get_input_embeddings().weight.detach().float().cpu()  # type: ignore[union-attr]
    soft_prompt, control_meta = build_target_embeddings(args, tokenizer, embedding_matrix)
    if args.max_soft_tokens is not None:
        soft_prompt = soft_prompt[: args.max_soft_tokens]
    if soft_prompt.size(1) != embedding_matrix.size(1):
        raise ValueError(
            f"Soft prompt hidden size {soft_prompt.size(1)} does not match "
            f"model embedding size {embedding_matrix.size(1)}"
        )

    nearest_rows = nearest_tokens(soft_prompt, embedding_matrix, tokenizer, top_k=args.top_k)
    write_jsonl(args.output_dir / "nearest_tokens.jsonl", nearest_rows)

    model_embedding_dtype = model.get_input_embeddings().weight.dtype  # type: ignore[union-attr]
    soft_prompt = soft_prompt.to(device=model.device, dtype=model_embedding_dtype)  # type: ignore[union-attr]
    empty_prefix = soft_prompt.new_empty((0, soft_prompt.size(1)))
    targets = target_states(prefix=empty_prefix, suffix=soft_prompt, model=model, layer_idx=layer_idx)
    recovery_config = RecoveryConfig(
        step_size=args.step_size,
        projection_iters_base=args.projection_iters_base,
        vocab_scale_factor=args.vocab_scale_factor,
        max_iters_per_token=args.max_iters_per_token,
        quiet_progress=args.quiet_progress,
    )
    result = find_prompt(
        prefix=empty_prefix,
        target_hidden_states=targets,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        config=recovery_config,
    )
    recovered_text = tokenizer.decode(result.recovered_ids)
    payload = {
        "input_dir": str(args.input_dir),
        "model_name": args.model_name,
        "control_mode": args.control_mode,
        "control": control_meta,
        "layer_idx": layer_idx,
        "precision": args.precision,
        "num_soft_tokens": int(soft_prompt.size(0)),
        "hidden_size": int(soft_prompt.size(1)),
        "max_iters_per_token": args.max_iters_per_token,
        "elapsed_seconds": time.time() - started,
        "recovery_elapsed_seconds": result.elapsed,
        "recovered_token_ids": result.recovered_ids,
        "recovered_text": recovered_text,
        "timesteps": result.timesteps,
        "token_times": result.times,
        "verified": result.verified,
        "all_positions_verified": bool(result.verified) and all(result.verified),
        "nearest_token_ids": [row["top_tokens"][0]["token_id"] for row in nearest_rows],
        "nearest_text": tokenizer.decode([row["top_tokens"][0]["token_id"] for row in nearest_rows]),
        "nearest_mean_l2": float(np.mean([row["top_tokens"][0]["l2"] for row in nearest_rows])),
    }
    (args.output_dir / "sipit_recovery.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "summary.md", payload)
    return 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_soft_prompt(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        tensor = payload.get("soft_prompt_embeddings")
    else:
        tensor = payload
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Could not find a tensor soft prompt in {path}")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D soft prompt tensor, got shape {tuple(tensor.shape)}")
    return tensor.detach().float().cpu()


def build_target_embeddings(args: argparse.Namespace, tokenizer: Any, embedding_matrix: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.control_mode == "soft_prompt":
        tensor = load_soft_prompt(args.input_dir / args.embedding_file)
        return tensor, {"source": str(args.input_dir / args.embedding_file)}

    num_tokens = args.control_num_tokens or args.max_soft_tokens or 16
    if num_tokens <= 0:
        raise ValueError("--control-num-tokens must be positive")

    if args.control_mode == "init_prompt":
        token_ids = tokenizer.encode(args.control_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("--control-text produced no tokens")
        while len(token_ids) < num_tokens:
            token_ids.extend(token_ids)
        token_ids = token_ids[:num_tokens]
        return embedding_matrix[token_ids].clone(), {
            "text": args.control_text,
            "token_ids": token_ids,
            "token_text": tokenizer.decode(token_ids),
        }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    if args.control_mode == "random_hard_tokens":
        special_ids = {token_id for token_id in tokenizer.all_special_ids if token_id is not None}
        token_ids: list[int] = []
        vocab_size = int(embedding_matrix.size(0))
        while len(token_ids) < num_tokens:
            candidate = int(torch.randint(0, vocab_size, (1,), generator=generator).item())
            if candidate not in special_ids:
                token_ids.append(candidate)
        return embedding_matrix[token_ids].clone(), {
            "token_ids": token_ids,
            "token_text": tokenizer.decode(token_ids),
        }

    if args.control_mode == "random_continuous":
        reference_path = args.input_dir / args.embedding_file
        if reference_path.exists():
            reference = load_soft_prompt(reference_path)
            if args.max_soft_tokens is not None:
                reference = reference[: args.max_soft_tokens]
            num_tokens = min(num_tokens, int(reference.size(0)))
            random_vectors = torch.randn((num_tokens, reference.size(1)), generator=generator)
            reference_norms = reference[:num_tokens].norm(dim=1, keepdim=True).clamp_min(1e-6)
            random_vectors = F.normalize(random_vectors, dim=1) * reference_norms
            return random_vectors.float().cpu(), {
                "matched_reference": str(reference_path),
                "matched_norms": True,
            }
        random_vectors = torch.randn((num_tokens, embedding_matrix.size(1)), generator=generator)
        mean_norm = embedding_matrix.norm(dim=1).mean().clamp_min(1e-6)
        random_vectors = F.normalize(random_vectors, dim=1) * mean_norm
        return random_vectors.float().cpu(), {"matched_reference": "", "matched_norms": False}

    raise ValueError(f"Unsupported control mode: {args.control_mode}")


def load_model(model_name: str, precision: str, layer_idx: int) -> tuple[Any, Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"device_map": {"": 0}, "trust_remote_code": True}
    if precision == "4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )
        kwargs["torch_dtype"] = torch.float32
    elif precision == "16":
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        model.norm_bk = model.model.norm
        model.model.norm = nn.Identity()
    total_layers = model.config.num_hidden_layers
    resolved_layer = total_layers + layer_idx + 1 if layer_idx < 0 else layer_idx
    torch.set_grad_enabled(True)
    return model, tokenizer, resolved_layer


def nearest_tokens(prompt: torch.Tensor, embeddings: torch.Tensor, tokenizer: Any, *, top_k: int) -> list[dict[str, Any]]:
    prompt_cpu = prompt.detach().float().cpu()
    emb_cpu = embeddings.detach().float().cpu()
    distances = torch.cdist(prompt_cpu, emb_cpu)
    l2_values, l2_indexes = distances.topk(k=top_k, largest=False, dim=1)
    prompt_norm = F.normalize(prompt_cpu, dim=1)
    emb_norm = F.normalize(emb_cpu, dim=1)
    cosine = prompt_norm @ emb_norm.T
    rows = []
    for soft_index in range(prompt_cpu.size(0)):
        rows.append(
            {
                "soft_token_index": soft_index,
                "norm": float(prompt_cpu[soft_index].norm().item()),
                "top_tokens": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token_text": tokenizer.decode([int(token_id)]),
                        "l2": float(l2_values[soft_index, rank].item()),
                        "cosine": float(cosine[soft_index, int(token_id)].item()),
                    }
                    for rank, token_id in enumerate(l2_indexes[soft_index].tolist())
                ],
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Soft Prompt SIPIT Recovery",
        "",
        f"- input: `{payload['input_dir']}`",
        f"- model: `{payload['model_name']}`",
        f"- control mode: `{payload['control_mode']}`",
        f"- layer: `{payload['layer_idx']}`",
        f"- precision: `{payload['precision']}`",
        f"- soft tokens: `{payload['num_soft_tokens']}`",
        f"- max iters/token: `{payload['max_iters_per_token']}`",
        f"- elapsed seconds: `{payload['elapsed_seconds']:.2f}`",
        f"- all positions verified: `{payload['all_positions_verified']}`",
        f"- nearest text: `{payload['nearest_text']}`",
        f"- recovered text: `{payload['recovered_text']}`",
        "",
        "This is an approximate diagnostic for continuous soft prompts; exact verification is not expected unless the learned vectors fall on the discrete token embedding manifold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
