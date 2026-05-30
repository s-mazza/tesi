"""Preflight checks for GEPA/G-EVAL cluster runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from .data import load_usr_examples, split_by_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", default="gepa-experiments/cache/tc_usr_data.json")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--nla-av-checkpoint", default="kitft/nla-qwen2.5-7b-L20-av")
    parser.add_argument("--hf-home", default="/llms")
    parser.add_argument("--train-contexts", type=int, default=10)
    parser.add_argument("--val-contexts", type=int, default=3)
    parser.add_argument("--test-contexts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")

    failed = [key for key, value in report.items() if key.startswith("ok_") and not value]
    if failed:
        raise SystemExit(f"Preflight failed: {', '.join(failed)}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    modules = ["torch", "vllm", "dspy", "gepa", "datasets"]
    module_ok = {name: importlib.util.find_spec(name) is not None for name in modules}

    torch_version = None
    torch_cuda = None
    vllm_version = None
    dspy_version = None
    if module_ok["torch"]:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    if module_ok["vllm"]:
        import vllm

        vllm_version = getattr(vllm, "__version__", "unknown")
    if module_ok["dspy"]:
        import dspy

        dspy_version = getattr(dspy, "__version__", "unknown")

    rows = load_usr_examples(args.data_source)
    split_sizes = [
        len(split)
        for split in split_by_context(
            rows,
            train_contexts=args.train_contexts,
            val_contexts=args.val_contexts,
            test_contexts=args.test_contexts,
            seed=args.seed,
        )
    ]

    hf_home = Path(args.hf_home)
    model_cache = hf_home / "hub" / f"models--{args.judge_model.replace('/', '--')}"
    nla_cache = hf_home / "hub" / f"models--{args.nla_av_checkpoint.replace('/', '--')}"

    return {
        "ok_modules": all(module_ok.values()),
        "modules": module_ok,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "vllm_version": vllm_version,
        "dspy_version": dspy_version,
        "flash_attn_available": importlib.util.find_spec("flash_attn") is not None,
        "ok_dataset": len(rows) == 360 and split_sizes == [60, 18, 24],
        "dataset_rows": len(rows),
        "split_sizes": split_sizes,
        "ok_judge_model_cache": model_cache.exists(),
        "judge_model_cache": str(model_cache),
        "ok_nla_av_cache": nla_cache.exists(),
        "nla_av_cache": str(nla_cache),
    }


if __name__ == "__main__":
    main()
