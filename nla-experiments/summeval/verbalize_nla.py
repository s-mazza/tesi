#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    from .common import (
        NLA_AV_MODEL_ID,
        artifact_path,
        cjk_fraction,
        extract_explanation as parse_explanation,
        write_jsonl,
    )
except ImportError:
    from common import (
        NLA_AV_MODEL_ID,
        artifact_path,
        cjk_fraction,
        extract_explanation as parse_explanation,
        write_jsonl,
    )


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def dry_run_generation(row: dict[str, Any]) -> tuple[str, str, str]:
    text = (
        "<explanation>"
        f"Dry run explanation for {row['dimension']} at {row['token_position']} "
        f"on {row['sample_bucket']}."
        "</explanation>"
    )
    explanation, parse_status = parse_explanation(text)
    return text, explanation, parse_status


def resolve_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint)
    if (path / "nla_meta.yaml").exists():
        return str(path)

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=checkpoint)


def load_nla_client(nla_root: Path, checkpoint: str, sglang_url: str, injection_scale: float | None):
    sys.path.insert(0, str(nla_root))
    from nla_inference import NLAClient

    checkpoint_path = resolve_checkpoint(checkpoint)
    return NLAClient(
        checkpoint_path,
        sglang_url=sglang_url,
        injection_scale_override=injection_scale,
    )


class TransformersNLAClient:
    """NLA actor inference through Hugging Face `generate(inputs_embeds=...)`.

    This reuses the upstream NLAClient embedding/injection path, but avoids a
    separate SGLang server for small thesis-scale runs.
    """

    def __init__(
        self,
        nla_root: Path,
        checkpoint: str,
        *,
        injection_scale: float | None,
        dtype_name: str,
        device_map: str,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        checkpoint_path = resolve_checkpoint(checkpoint)
        self.embed_client = load_nla_client(nla_root, checkpoint_path, "http://127.0.0.1:9", injection_scale)
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

    def generate(
        self,
        activation: list[float],
        *,
        extract_explanation: bool,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        embeds_np, prompt_len = self.embed_client._build_embeds(
            self.torch.as_tensor(activation, dtype=self.torch.float32),
            None,
        )
        embeds = self.torch.from_numpy(embeds_np).unsqueeze(0).to(self.device, dtype=self.dtype)
        attention_mask = self.torch.ones(embeds.shape[:2], dtype=self.torch.long, device=self.device)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        do_sample = temperature > 0
        generation_args = {
            "inputs_embeds": embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_args["temperature"] = temperature

        with self.torch.inference_mode():
            generated = self.model.generate(**generation_args)

        token_ids = generated[0].tolist()
        if len(token_ids) > prompt_len:
            token_ids = token_ids[prompt_len:]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if extract_explanation:
            explanation, _ = parse_explanation(text)
            return explanation
        return text


def iter_verbalizations(
    activation_rows: list[dict[str, Any]],
    *,
    checkpoint: str,
    sglang_url: str,
    nla_root: Path,
    backend: str,
    limit: int | None,
    dry_run: bool,
    temperature: float,
    max_new_tokens: int,
    injection_scale: float | None,
    dtype_name: str,
    device_map: str,
    trust_remote_code: bool,
) -> Iterator[dict[str, Any]]:
    rows = activation_rows[:limit]
    client = None
    if not dry_run:
        if backend == "sglang":
            client = load_nla_client(nla_root, checkpoint, sglang_url, injection_scale)
        elif backend == "transformers":
            client = TransformersNLAClient(
                nla_root,
                checkpoint,
                injection_scale=injection_scale,
                dtype_name=dtype_name,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    for index, row in enumerate(rows, start=1):
        if dry_run:
            raw_generation, explanation, parse_status = dry_run_generation(row)
        else:
            assert client is not None
            raw_generation = client.generate(
                row["activation_vector"],
                extract_explanation=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            explanation, parse_status = parse_explanation(raw_generation)

        failure_score = cjk_fraction(raw_generation)
        yield {
            **{key: value for key, value in row.items() if key != "activation_vector"},
            "nla_model_id": checkpoint,
            "nla_backend": "dry_run" if dry_run else backend,
            "sglang_url": sglang_url,
            "raw_generation": raw_generation,
            "explanation": explanation,
            "parse_status": parse_status,
            "injection_check_status": "cjk_like" if failure_score > 0.25 else "ok",
            "cjk_fraction": failure_score,
        }
        print(f"Verbalized {index}/{len(rows)} activation rows", flush=True)


def build_verbalizations(
    activation_rows: list[dict[str, Any]],
    *,
    checkpoint: str,
    sglang_url: str,
    nla_root: Path,
    backend: str,
    limit: int | None,
    dry_run: bool,
    temperature: float,
    max_new_tokens: int,
    injection_scale: float | None,
    dtype_name: str,
    device_map: str,
    trust_remote_code: bool,
) -> list[dict[str, Any]]:
    outputs = list(
        iter_verbalizations(
            activation_rows,
            checkpoint=checkpoint,
            sglang_url=sglang_url,
            nla_root=nla_root,
            backend=backend,
            limit=limit,
            dry_run=dry_run,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            injection_scale=injection_scale,
            dtype_name=dtype_name,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Verbalize saved activation vectors with the NLA AV model.")
    parser.add_argument("--activations", type=Path, default=artifact_path("summeval", "activations_qwen25_7b_instruct_L20.parquet"))
    parser.add_argument("--output", type=Path, default=artifact_path("summeval", "verbalizations.jsonl"))
    parser.add_argument("--checkpoint", default=NLA_AV_MODEL_ID)
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--nla-root", type=Path, default=Path("natural_language_autoencoders"))
    parser.add_argument("--backend", default="sglang", choices=("sglang", "transformers"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--injection-scale", type=float, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    activation_rows = read_parquet_rows(args.activations)
    outputs = iter_verbalizations(
        activation_rows,
        checkpoint=args.checkpoint,
        sglang_url=args.sglang_url,
        nla_root=args.nla_root,
        backend=args.backend,
        limit=args.limit,
        dry_run=args.dry_run,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        injection_scale=args.injection_scale,
        dtype_name=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    count = write_jsonl(args.output, outputs, flush=True)
    print(f"Wrote {count} verbalization rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
