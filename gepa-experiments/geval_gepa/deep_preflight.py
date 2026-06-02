"""Deep CPU-only readiness checks for the GEPA/G-EVAL cluster run."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .data import LABEL_SCALES, load_usr_examples, split_by_context
from .metrics import normalized_absolute_score, parse_discrete_score
from .prompts import ENGAGING_SEED_INSTRUCTIONS, metric_description
from .runner import create_metric_fn, get_gepa_class, make_dspy_examples, make_program


REQUIRED_CONFIG_KEYS = {
    "DATA_SOURCE",
    "LABEL",
    "TRAIN_CONTEXTS",
    "VAL_CONTEXTS",
    "TEST_CONTEXTS",
    "SEED",
    "JUDGE_MODEL",
    "NLA_AV_CHECKPOINT",
    "NLA_EXTRACTION_LAYER",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "NUM_THREADS",
    "OUTPUT_DIR",
}
BUDGET_KEYS = ("GEPA_AUTO", "MAX_FULL_EVALS", "MAX_METRIC_CALLS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default="gepa-experiments/config/geval_gepa_engaging_qwen25.env")
    parser.add_argument("--hf-home", default="/llms")
    parser.add_argument("--expected-vllm", default="0.10.2")
    parser.add_argument("--expected-torch-prefix", default="2.8.0")
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
        raise SystemExit(f"Deep preflight failed: {', '.join(failed)}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_env_file(Path(args.config_file))
    report: dict[str, Any] = {
        "config_file": args.config_file,
        "ok_config_keys": REQUIRED_CONFIG_KEYS.issubset(config),
        "missing_config_keys": sorted(REQUIRED_CONFIG_KEYS - set(config)),
        "ok_budget": sum(bool(config.get(key)) for key in BUDGET_KEYS) == 1,
        "budget_values": {key: config.get(key, "") for key in BUDGET_KEYS},
    }

    modules = _runtime_module_report()
    report.update(modules)
    report.update(_compiler_report())
    report["ok_version_pin"] = (
        str(modules.get("torch_version", "")).startswith(args.expected_torch_prefix)
        and modules.get("vllm_version") == args.expected_vllm
    )

    label = config.get("LABEL", "")
    data_source = config.get("DATA_SOURCE", "")
    train_contexts = _as_int(config.get("TRAIN_CONTEXTS"), "TRAIN_CONTEXTS")
    val_contexts = _as_int(config.get("VAL_CONTEXTS"), "VAL_CONTEXTS")
    test_contexts = _as_int(config.get("TEST_CONTEXTS"), "TEST_CONTEXTS")
    seed = _as_int(config.get("SEED"), "SEED")

    rows = load_usr_examples(data_source)
    train_rows, val_rows, test_rows = split_by_context(
        rows,
        train_contexts=train_contexts,
        val_contexts=val_contexts,
        test_contexts=test_contexts,
        seed=seed,
    )
    selected_rows = train_rows + val_rows + test_rows
    split_contexts = [_context_ids(split) for split in (train_rows, val_rows, test_rows)]
    context_overlaps = [
        sorted(split_contexts[0] & split_contexts[1]),
        sorted(split_contexts[0] & split_contexts[2]),
        sorted(split_contexts[1] & split_contexts[2]),
    ]
    report.update(
        {
            "ok_label": label == "Engaging" and label in LABEL_SCALES,
            "ok_dataset": len(rows) == 360,
            "dataset_rows": len(rows),
            "ok_split_sizes": [len(train_rows), len(val_rows), len(test_rows)] == [60, 18, 24],
            "split_sizes": [len(train_rows), len(val_rows), len(test_rows)],
            "ok_split_volume": 80 <= len(selected_rows) <= 120,
            "selected_rows": len(selected_rows),
            "ok_context_disjoint": not any(context_overlaps),
            "context_overlaps": context_overlaps,
            "ok_response_ids_unique": len({row.response_id for row in rows}) == len(rows),
        }
    )

    report.update(_metric_report(label))
    report.update(_prompt_program_report(label, train_rows))
    report.update(_model_cache_report(Path(args.hf_home), config))
    report.update(_vllm_architecture_report())
    return report


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _runtime_module_report() -> dict[str, Any]:
    modules = ["torch", "vllm", "dspy", "gepa", "datasets", "transformers"]
    module_ok = {name: importlib.util.find_spec(name) is not None for name in modules}
    report: dict[str, Any] = {"ok_modules": all(module_ok.values()), "modules": module_ok}

    if module_ok["torch"]:
        import torch

        report["torch_version"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
    if module_ok["vllm"]:
        import vllm

        report["vllm_version"] = getattr(vllm, "__version__", "unknown")
    if module_ok["dspy"]:
        import dspy

        report["dspy_version"] = getattr(dspy, "__version__", "unknown")
    if module_ok["gepa"]:
        try:
            report["gepa_version"] = importlib.metadata.version("gepa")
        except importlib.metadata.PackageNotFoundError:
            report["gepa_version"] = "unknown"
    report["flash_attn_available"] = importlib.util.find_spec("flash_attn") is not None
    return report


def _compiler_report() -> dict[str, Any]:
    cc = os.environ.get("CC", "gcc")
    cxx = os.environ.get("CXX", "g++")
    cc_path = shutil.which(cc)
    cxx_path = shutil.which(cxx)
    return {
        "ok_c_compiler": cc_path is not None and cxx_path is not None,
        "cc": cc,
        "cc_path": cc_path,
        "cxx": cxx,
        "cxx_path": cxx_path,
    }


def _metric_report(label: str) -> dict[str, Any]:
    min_score, max_score = LABEL_SCALES[label]
    parsed_from_text = parse_discrete_score("Rationale...\nScore: 3", min_score=min_score, max_score=max_score)
    parsed_invalid = parse_discrete_score("Score: 5", min_score=min_score, max_score=max_score)
    perfect = normalized_absolute_score(2, 2.0, min_score=min_score, max_score=max_score)
    off_by_one = normalized_absolute_score(1, 2.0, min_score=min_score, max_score=max_score)
    return {
        "ok_metric_parser": parsed_from_text == 3 and parsed_invalid is None,
        "ok_metric_scale": perfect == 1.0 and 0.0 <= off_by_one < perfect,
        "metric_description": metric_description(label),
    }


def _prompt_program_report(label: str, train_rows: list[Any]) -> dict[str, Any]:
    program = make_program(ENGAGING_SEED_INSTRUCTIONS)
    examples = make_dspy_examples(train_rows[:2], label)
    metric_fn = create_metric_fn(label)
    good_feedback = metric_fn(examples[0], type("Pred", (), {"score": "3"})())
    bad_feedback = metric_fn(examples[0], type("Pred", (), {"score": "bad"})())
    GEPA = get_gepa_class()
    gepa_signature = inspect.signature(GEPA)
    return {
        "ok_prompt_mentions_scale": "1 to 3" in ENGAGING_SEED_INSTRUCTIONS and "Engagingness" in ENGAGING_SEED_INSTRUCTIONS,
        "ok_program_constructs": program is not None and len(examples) == 2,
        "ok_metric_feedback": hasattr(good_feedback, "score") and hasattr(bad_feedback, "feedback"),
        "ok_gepa_api": "metric" in gepa_signature.parameters,
        "gepa_signature": str(gepa_signature),
    }


def _model_cache_report(hf_home: Path, config: dict[str, str]) -> dict[str, Any]:
    judge_model = config.get("JUDGE_MODEL", "")
    nla_checkpoint = config.get("NLA_AV_CHECKPOINT", "")
    judge_cache = hf_home / "hub" / f"models--{judge_model.replace('/', '--')}"
    nla_cache = hf_home / "hub" / f"models--{nla_checkpoint.replace('/', '--')}"
    report: dict[str, Any] = {
        "ok_judge_model_cache": judge_cache.exists(),
        "judge_model_cache": str(judge_cache),
        "ok_nla_av_cache": nla_cache.exists(),
        "nla_av_cache": str(nla_cache),
        "ok_nla_model_family": "qwen2.5-7b" in nla_checkpoint.lower() and "Qwen2.5-7B" in judge_model,
        "ok_nla_layer": _as_int(config.get("NLA_EXTRACTION_LAYER"), "NLA_EXTRACTION_LAYER") == 20,
    }
    if importlib.util.find_spec("transformers") is None or not judge_cache.exists():
        report["ok_transformers_local_config"] = False
        return report

    from transformers import AutoConfig, AutoTokenizer

    hf_path = _latest_snapshot(judge_cache)
    model_config = AutoConfig.from_pretrained(hf_path, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, local_files_only=True, trust_remote_code=True)
    architectures = getattr(model_config, "architectures", []) or []
    has_vllm_tokenizer_attrs = hasattr(tokenizer, "all_special_tokens_extended")
    report.update(
        {
            "judge_model_type": getattr(model_config, "model_type", ""),
            "judge_architectures": architectures,
            "ok_transformers_local_config": (
                getattr(model_config, "model_type", "") == "qwen2" and "Qwen2ForCausalLM" in architectures
            ),
            "ok_tokenizer_local": tokenizer is not None,
            "ok_tokenizer_vllm_attrs": has_vllm_tokenizer_attrs,
            "tokenizer_class": tokenizer.__class__.__name__,
        }
    )
    return report


def _vllm_architecture_report() -> dict[str, Any]:
    try:
        importlib.import_module("vllm.model_executor.models.qwen2")
    except Exception as exc:  # pragma: no cover - depends on container runtime
        return {"ok_vllm_qwen2_import": False, "vllm_qwen2_import_error": f"{type(exc).__name__}: {exc}"}
    return {"ok_vllm_qwen2_import": True, "vllm_qwen2_import_error": ""}


def _latest_snapshot(model_cache: Path) -> Path:
    snapshots_dir = model_cache / "snapshots"
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise ValueError(f"No Hugging Face snapshots found under {snapshots_dir}")
    return snapshots[-1]


def _context_ids(rows: list[Any]) -> set[str]:
    return {row.context_id for row in rows}


def _as_int(value: str | None, key: str) -> int:
    if value is None or value == "":
        raise ValueError(f"Missing integer config value {key}")
    return int(value)


if __name__ == "__main__":
    main()
