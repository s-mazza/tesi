"""Preflight checks for GEPA/G-EVAL cluster runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from .tasks import get_task, split_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", default="gepa-experiments/cache/tc_usr_data.json")
    parser.add_argument("--dataset", default="topical_chat")
    parser.add_argument("--dimension", default="")
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

    task = get_task(args.dataset)
    dimension = args.dimension or task.default_dimension
    rows = task.load(args.data_source, dimension)
    train_rows, val_rows, test_rows = split_examples(
        rows,
        train_groups=args.train_contexts,
        val_groups=args.val_contexts,
        test_groups=args.test_contexts,
        seed=args.seed,
    )
    split_sizes = [len(train_rows), len(val_rows), len(test_rows)]
    split_groups = [
        len({row.group_id for row in train_rows}),
        len({row.group_id for row in val_rows}),
        len({row.group_id for row in test_rows}),
    ]
    expected_split_groups = [args.train_contexts, args.val_contexts, args.test_contexts]
    split_group_sets = [
        {row.group_id for row in train_rows},
        {row.group_id for row in val_rows},
        {row.group_id for row in test_rows},
    ]
    context_overlaps = [
        sorted(split_group_sets[0] & split_group_sets[1]),
        sorted(split_group_sets[0] & split_group_sets[2]),
        sorted(split_group_sets[1] & split_group_sets[2]),
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
        "dataset": task.dataset,
        "dimension": dimension,
        "ok_dataset": len(rows) > 0,
        "ok_split_group_counts": split_groups == expected_split_groups,
        "ok_context_disjoint": not any(context_overlaps),
        "ok_response_ids_unique": len({row.example_id for row in rows}) == len(rows),
        "dataset_rows": len(rows),
        "split_sizes": {
            "gepa_train": split_sizes[0],
            "gepa_validation": split_sizes[1],
            "final_test": split_sizes[2],
        },
        "split_group_counts": {
            "gepa_train": split_groups[0],
            "gepa_validation": split_groups[1],
            "final_test": split_groups[2],
        },
        "expected_split_group_counts": {
            "gepa_train": expected_split_groups[0],
            "gepa_validation": expected_split_groups[1],
            "final_test": expected_split_groups[2],
        },
        "context_overlaps": context_overlaps,
        "ok_judge_model_cache": model_cache.exists(),
        "judge_model_cache": str(model_cache),
        "ok_nla_av_cache": nla_cache.exists(),
        "nla_av_cache": str(nla_cache),
    }


if __name__ == "__main__":
    main()
