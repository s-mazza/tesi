#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from .common import (
        NLA_AV_MODEL_ID,
        artifact_path,
        cjk_fraction,
        extract_explanation,
        write_jsonl,
    )
except ImportError:
    from common import (
        NLA_AV_MODEL_ID,
        artifact_path,
        cjk_fraction,
        extract_explanation,
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
    explanation, parse_status = extract_explanation(text)
    return text, explanation, parse_status


def load_nla_client(nla_root: Path, checkpoint: str, sglang_url: str, injection_scale: float | None):
    sys.path.insert(0, str(nla_root))
    from nla_inference import NLAClient

    return NLAClient(
        checkpoint,
        sglang_url=sglang_url,
        injection_scale_override=injection_scale,
    )


def build_verbalizations(
    activation_rows: list[dict[str, Any]],
    *,
    checkpoint: str,
    sglang_url: str,
    nla_root: Path,
    limit: int | None,
    dry_run: bool,
    temperature: float,
    max_new_tokens: int,
    injection_scale: float | None,
) -> list[dict[str, Any]]:
    rows = activation_rows[:limit]
    client = None
    if not dry_run:
        client = load_nla_client(nla_root, checkpoint, sglang_url, injection_scale)

    outputs: list[dict[str, Any]] = []
    for row in rows:
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
            explanation, parse_status = extract_explanation(raw_generation)

        failure_score = cjk_fraction(raw_generation)
        outputs.append(
            {
                **{key: value for key, value in row.items() if key != "activation_vector"},
                "nla_model_id": checkpoint,
                "sglang_url": sglang_url,
                "raw_generation": raw_generation,
                "explanation": explanation,
                "parse_status": parse_status,
                "injection_check_status": "cjk_like" if failure_score > 0.25 else "ok",
                "cjk_fraction": failure_score,
            }
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Verbalize saved activation vectors with the NLA AV model.")
    parser.add_argument("--activations", type=Path, default=artifact_path("summeval", "activations_qwen25_7b_instruct_L20.parquet"))
    parser.add_argument("--output", type=Path, default=artifact_path("summeval", "verbalizations.jsonl"))
    parser.add_argument("--checkpoint", default=NLA_AV_MODEL_ID)
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--nla-root", type=Path, default=Path("natural_language_autoencoders"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--injection-scale", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    activation_rows = read_parquet_rows(args.activations)
    outputs = build_verbalizations(
        activation_rows,
        checkpoint=args.checkpoint,
        sglang_url=args.sglang_url,
        nla_root=args.nla_root,
        limit=args.limit,
        dry_run=args.dry_run,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        injection_scale=args.injection_scale,
    )
    count = write_jsonl(args.output, outputs)
    print(f"Wrote {count} verbalization rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
