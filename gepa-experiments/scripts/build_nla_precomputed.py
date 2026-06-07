#!/usr/bin/env python3
"""Build precomputed NLA verbalization feedback for GEPA train/validation rows."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from geval_gepa.nla_precompute import (
    QwenActivationExtractor,
    fake_activation_rows,
    iter_verbalization_rows,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--nla-checkpoint", default="kitft/nla-qwen2.5-7b-L20-av")
    parser.add_argument("--nla-root", type=Path, default=Path("natural_language_autoencoders"))
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--max-tokens-per-example", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--activation-dtype", default="float16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--nla-dtype", default="float16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--backend", default="transformers", choices=("transformers",))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--injection-scale", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-d-model", type=int, default=3584)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    if args.dry_run:
        activation_rows = fake_activation_rows(
            manifest_rows,
            model_id=args.activation_model,
            layer=args.layer,
            max_tokens_per_example=args.max_tokens_per_example,
            d_model=args.fake_d_model,
            limit=args.limit,
        )
    else:
        extractor = QwenActivationExtractor(
            model_id=args.activation_model,
            layer=args.layer,
            dtype_name=args.activation_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            use_chat_template=not args.no_chat_template,
            max_tokens_per_example=args.max_tokens_per_example,
        )
        try:
            activation_rows = extractor.extract(manifest_rows, limit=args.limit)
        finally:
            extractor.close()
            gc.collect()

    print(f"Prepared {len(activation_rows)} NLA activation rows.", flush=True)
    count = write_jsonl(
        args.output,
        iter_verbalization_rows(
            activation_rows,
            checkpoint=args.nla_checkpoint,
            backend=args.backend,
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
    print(f"Wrote {count} NLA precomputed feedback rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
