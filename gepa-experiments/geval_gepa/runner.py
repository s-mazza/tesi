"""Run GEPA prompt optimization for G-EVAL-style benchmark judges."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .data import DEFAULT_USR_URL, LABEL_SCALES
from .metrics import compute_regression_metrics, normalized_absolute_score, parse_discrete_score
from .nla_feedback import NlaFeedbackProvider
from .perplexity import VllmPerplexityScorer, format_perplexity_feedback
from .prompts import ENGAGING_SEED_INSTRUCTIONS, metric_description, seed_instructions
from .proposers import make_instruction_proposer
from .tasks import EvalExample, get_task, split_examples, write_split_manifest
from .trajectory import export_prompt_trajectory, write_fallback_gepa_viz_run


def make_program(instructions: str) -> Any:
    import dspy

    class JudgeSignature(dspy.Signature):
        source_text: str = dspy.InputField(desc="Source document or dialogue context to evaluate against.")
        fact: str = dspy.InputField(desc='Relevant reference/fact, or "_nofact" when none is provided.')
        candidate_output: str = dspy.InputField(desc="Candidate response or summary to evaluate.")
        rationale: str = dspy.OutputField(desc="Brief explanation of the score.")
        score: str = dspy.OutputField(desc="Exactly one integer in the requested score scale.")

    class GevalJudge(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.judge = dspy.ChainOfThought(JudgeSignature.with_instructions(instructions))

        def forward(self, source_text: str, fact: str, candidate_output: str, **kwargs: Any) -> Any:
            del kwargs
            return self.judge(source_text=source_text, fact=fact, candidate_output=candidate_output)

    return GevalJudge()


def make_dspy_examples(rows: list[EvalExample]) -> list[Any]:
    import dspy

    examples = []
    for row in rows:
        examples.append(
            dspy.Example(
                source_text=row.source_text,
                context=row.context,
                fact=row.fact,
                response=row.candidate_output,
                candidate_output=row.candidate_output,
                reference=row.reference,
                human_score=row.human_score,
                context_id=row.group_id,
                response_id=row.example_id,
                group_id=row.group_id,
                example_id=row.example_id,
                model=row.system_id,
                dataset=row.dataset,
                dimension=row.dimension,
                min_score=row.min_score,
                max_score=row.max_score,
            ).with_inputs("source_text", "fact", "candidate_output")
        )
    return examples


def create_metric_fn(
    label: str,
    perplexity_scorer: Any | None = None,
    *,
    min_score: int | None = None,
    max_score: int | None = None,
    nla_feedback_provider: Any | None = None,
) -> Callable[..., Any]:
    import dspy

    if min_score is None or max_score is None:
        min_score, max_score = LABEL_SCALES[label]
    metric_label = label

    def metric_fn(example: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None) -> Any:
        del trace, pred_name, pred_trace
        parsed = parse_discrete_score(pred, min_score=min_score, max_score=max_score)
        target = float(example.human_score)

        if parsed is None:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"FORMAT ERROR: score must be exactly one integer from {min_score} to {max_score}. "
                    f"Human mean for {metric_label} is {target:.2f}."
                ),
            )

        score = normalized_absolute_score(parsed, target, min_score=min_score, max_score=max_score)
        delta = parsed - target
        direction = _error_direction(delta)
        feedback = [
            f"Human mean {metric_label} score: {target:.2f}; predicted score: {parsed}; normalized agreement: {score:.3f}.",
            f"Metric definition: {metric_description(metric_label)}",
            (
                "Abstract error summary: "
                f"target_bucket={_score_bucket(target, min_score=min_score, max_score=max_score)}; "
                f"judge_bucket={_score_bucket(parsed, min_score=min_score, max_score=max_score)}; "
                f"error_direction={direction}; "
                f"agreement_bucket={_agreement_bucket(score)}."
            ),
            f"Rubric signals: {_abstract_rubric_signals(example)}",
        ]
        if perplexity_scorer is not None:
            try:
                perplexity = perplexity_scorer.score_example(example)
            except Exception as exc:
                context_id = getattr(example, "context_id", "unknown_context")
                response_id = getattr(example, "response_id", "unknown_response")
                raise RuntimeError(
                    f"Perplexity feedback failed for {context_id}/{response_id}: {type(exc).__name__}: {exc}"
                ) from exc
            feedback.append(format_perplexity_feedback(perplexity))
        if nla_feedback_provider is not None:
            feedback.append(nla_feedback_provider.feedback_for(example))
        if abs(delta) >= 1.0:
            feedback.append(
                f"The output was {direction}. Revise the judging instructions to better distinguish weak outputs "
                "from outputs that satisfy the target dimension."
            )
        else:
            feedback.append("The predicted score is close to the aggregated human annotation.")

        return dspy.Prediction(score=score, feedback="\n".join(feedback))

    return metric_fn


def _score_bucket(score: float, *, min_score: int, max_score: int) -> str:
    if max_score == min_score:
        return "single"
    normalized = (score - min_score) / (max_score - min_score)
    if normalized <= 0.25:
        return "low"
    if normalized >= 0.75:
        return "high"
    return "middle"


def _agreement_bucket(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _error_direction(delta: float) -> str:
    if abs(delta) < 0.5:
        return "close"
    return "overrated" if delta > 0 else "underrated"


def _abstract_rubric_signals(example: Any) -> str:
    response = str(getattr(example, "response", "") or getattr(example, "candidate_output", ""))
    context = str(getattr(example, "context", "") or getattr(example, "source_text", ""))
    fact = str(getattr(example, "fact", ""))
    response_words = _word_count(response)
    context_turns = sum(1 for line in context.splitlines() if line.strip())
    generic_ack = _looks_like_generic_acknowledgement(response)
    signals = {
        "candidate_length": _length_bucket(response_words),
        "context_depth": _length_bucket(context_turns, short=4, medium=10),
        "asks_question": _yes_no("?" in response),
        "fact_available": _yes_no(fact.strip() != "_nofact"),
        "generic_acknowledgement_only": _yes_no(generic_ack),
        "has_concrete_surface_detail": _yes_no(_has_concrete_surface_detail(response)),
    }
    return "; ".join(f"{key}={value}" for key, value in signals.items())


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _length_bucket(count: int, *, short: int = 8, medium: int = 24) -> str:
    if count <= short:
        return "short"
    if count <= medium:
        return "medium"
    return "long"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _looks_like_generic_acknowledgement(text: str) -> bool:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if len(words) > 10:
        return False
    generic_terms = {
        "ok",
        "okay",
        "yes",
        "yeah",
        "sure",
        "cool",
        "nice",
        "great",
        "interesting",
        "thanks",
        "wow",
        "haha",
    }
    return bool(words) and sum(word in generic_terms for word in words) >= max(1, len(words) - 2)


def _has_concrete_surface_detail(text: str) -> bool:
    tokens = re.findall(r"\b[\w'-]+\b", text)
    if any(token[:1].isupper() and token.lower() not in {"i"} for token in tokens):
        return True
    if any(char.isdigit() for char in text):
        return True
    return _word_count(text) >= 12


def make_openai_lm(
    *,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> Any:
    import dspy

    return dspy.LM(
        model=f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def configure_lms(args: argparse.Namespace) -> tuple[Any, Any]:
    import dspy

    judge_lm = make_openai_lm(
        model=args.judge_model,
        api_base=args.api_base,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if hasattr(dspy, "configure"):
        dspy.configure(lm=judge_lm)
    else:
        dspy.settings.configure(lm=judge_lm)

    proposer_lm = judge_lm
    if args.proposer_api_base:
        proposer_key = args.proposer_api_key or os.getenv(args.proposer_api_key_env, "EMPTY")
        proposer_lm = make_openai_lm(
            model=args.proposer_model,
            api_base=args.proposer_api_base,
            api_key=proposer_key,
            max_tokens=args.proposer_max_tokens,
            temperature=args.proposer_temperature,
        )

    return judge_lm, proposer_lm


def get_gepa_class() -> Any:
    import dspy

    if hasattr(dspy, "GEPA"):
        return dspy.GEPA
    try:
        from dspy.teleprompt import GEPA

        return GEPA
    except ImportError as exc:
        raise ImportError("GEPA is unavailable. Install compatible dspy/gepa packages.") from exc


def evaluate_program(program: Any, rows: list[EvalExample], output_path: Path) -> dict[str, Any]:
    predictions: list[float] = []
    targets: list[float] = []
    agreement_scores: list[float] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            pred = program(source_text=row.source_text, fact=row.fact, candidate_output=row.candidate_output)
            min_score, max_score = row.min_score, row.max_score
            parsed = parse_discrete_score(pred, min_score=min_score, max_score=max_score)
            prediction = float(parsed) if parsed is not None else float("nan")
            target = row.human_score
            parse_status = "ok" if parsed is not None else "unparsed"
            if parsed is not None:
                predictions.append(prediction)
                targets.append(target)
                agreement_scores.append(normalized_absolute_score(prediction, target, min_score=min_score, max_score=max_score))
            else:
                agreement_scores.append(0.0)
            handle.write(
                json.dumps(
                    {
                        "dataset": row.dataset,
                        "dimension": row.dimension,
                        "group_id": row.group_id,
                        "example_id": row.example_id,
                        "model": row.system_id,
                        "target": target,
                        "prediction": parsed,
                        "parse_status": parse_status,
                        "raw_score": str(getattr(pred, "score", "")),
                        "raw_rationale": str(getattr(pred, "rationale", "")),
                        "min_score": row.min_score,
                        "max_score": row.max_score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(rows)
    parsed_count = len(predictions)
    coverage = parsed_count / total if total else 0.0
    agreement = sum(agreement_scores) / total if total else 0.0
    common_metrics: dict[str, Any] = {
        "total": total,
        "parsed": parsed_count,
        "coverage": coverage,
        "agreement": agreement,
    }
    if predictions:
        return {**common_metrics, **compute_regression_metrics(predictions, targets).as_dict()}
    return {**common_metrics, "n": 0, "pearson": 0.0, "spearman": 0.0, "mae": float("nan")}


def extract_instructions(program: Any) -> str:
    candidates = [
        ("judge", "predict", "signature"),
        ("judge", "signature"),
        ("predict", "signature"),
        ("signature",),
    ]
    for path in candidates:
        obj = program
        try:
            for attr in path:
                obj = getattr(obj, attr)
            instructions = getattr(obj, "instructions", None)
            if instructions:
                return str(instructions)
        except AttributeError:
            continue
    return "Could not extract optimized instructions from this DSPy version."


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", default=DEFAULT_USR_URL)
    parser.add_argument("--dataset", default="topical_chat")
    parser.add_argument("--dimension", default="")
    parser.add_argument("--label", default="Engaging", choices=sorted(LABEL_SCALES), help="Legacy Topical-Chat label.")
    parser.add_argument("--train-contexts", type=int, default=10)
    parser.add_argument("--val-contexts", type=int, default=3)
    parser.add_argument("--test-contexts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="gepa-experiments/results/geval_gepa_engaging_qwen25")

    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--nla-av-checkpoint", default="kitft/nla-qwen2.5-7b-L20-av")
    parser.add_argument("--nla-extraction-layer", type=int, default=20)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--proposer-model", default="local-llamacpp")
    parser.add_argument("--proposer-api-base", default="")
    parser.add_argument("--proposer-api-key", default="")
    parser.add_argument("--proposer-api-key-env", default="LLAMA_API_KEY")
    parser.add_argument("--proposer-temperature", type=float, default=0.7)
    parser.add_argument("--proposer-max-tokens", type=int, default=4096)
    parser.add_argument("--perplexity-feedback", action="store_true")
    parser.add_argument("--perplexity-hf-home", default="/llms")
    parser.add_argument("--perplexity-prompt-logprobs", type=int, default=20)
    parser.add_argument("--perplexity-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--nla-feedback", action="store_true")
    parser.add_argument("--nla-backend", default="precomputed", choices=("precomputed", "dry_run"))
    parser.add_argument("--nla-precomputed-path", default="")
    parser.add_argument("--nla-max-tokens-per-example", type=int, default=6)

    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--gepa-auto", choices=["light", "medium", "heavy"])
    budget.add_argument("--max-full-evals", type=int)
    budget.add_argument("--max-metric-calls", type=int)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--instruction-proposer",
        default="default",
        choices=("default", "generalizing"),
        help="GEPA proposer mode. 'generalizing' avoids copying validation feedback into the prompt.",
    )
    parser.add_argument("--skip-gepa", action="store_true", help="Only evaluate the seed prompt.")
    return parser.parse_args()


def resolve_task_from_args(args: argparse.Namespace) -> tuple[Any, str, str]:
    dataset = args.dataset
    dimension = args.dimension
    if dataset == "topical_chat" and not dimension:
        dimension = _legacy_label_to_dimension(args.label)
    task = get_task(dataset)
    dimension = task.default_dimension if not dimension else dimension
    return task, task.dataset, dimension


def _legacy_label_to_dimension(label: str) -> str:
    mapping = {
        "Natural": "naturalness",
        "Maintains Context": "coherence",
        "Engaging": "engagingness",
        "Uses Knowledge": "groundedness",
        "Understandable": "naturalness",
        "Overall": "engagingness",
    }
    return mapping.get(label, label.lower().replace(" ", "_"))


def _same_scale(rows: list[EvalExample]) -> tuple[int, int]:
    scales = {(row.min_score, row.max_score) for row in rows}
    if len(scales) != 1:
        raise ValueError(f"Expected one score scale per run, found: {sorted(scales)}")
    return next(iter(scales))


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    monotonic_started = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    task, dataset, dimension = resolve_task_from_args(args)
    rows = task.load(args.data_source, dimension)
    min_score, max_score = _same_scale(rows)
    seed_prompt = seed_instructions(
        dataset=dataset,
        dimension=dimension,
        min_score=min_score,
        max_score=max_score,
    )
    train_rows, val_rows, test_rows = split_examples(
        rows,
        train_groups=args.train_contexts,
        val_groups=args.val_contexts,
        test_groups=args.test_contexts,
        seed=args.seed,
    )
    write_split_manifest(output_dir / f"split_manifest_{timestamp}.json", train_rows, val_rows, test_rows)
    print(
        "Split rows: "
        f"gepa_train={len(train_rows)}, gepa_validation={len(val_rows)}, final_test={len(test_rows)}. "
        "GEPA compile uses only gepa_train/gepa_validation; final_test is evaluated only after optimization.",
        flush=True,
    )

    judge_lm, proposer_lm = configure_lms(args)
    perplexity_scorer = None
    nla_feedback_provider = None
    if args.perplexity_feedback:
        perplexity_scorer = VllmPerplexityScorer(
            api_base=args.api_base,
            model=args.judge_model,
            tokenizer_model=args.judge_model,
            hf_home=args.perplexity_hf_home,
            prompt_logprobs=args.perplexity_prompt_logprobs,
            timeout_seconds=args.perplexity_timeout_seconds,
        )
        examples_for_feedback = make_dspy_examples(train_rows + val_rows)
        print(
            f"Precomputing response-only perplexity feedback for {len(examples_for_feedback)} GEPA train/validation rows.",
            flush=True,
        )
        for index, example in enumerate(examples_for_feedback, start=1):
            perplexity_scorer.score_example(example)
            if index % 25 == 0 or index == len(examples_for_feedback):
                print(f"Perplexity feedback cache: {index}/{len(examples_for_feedback)} rows scored.", flush=True)
    if args.nla_feedback:
        nla_feedback_provider = NlaFeedbackProvider(
            checkpoint=args.nla_av_checkpoint,
            layer=args.nla_extraction_layer,
            backend=args.nla_backend,
            max_tokens_per_example=args.nla_max_tokens_per_example,
            precomputed_path=args.nla_precomputed_path,
        )

    seed_program = make_program(seed_prompt)
    optimized_program = seed_program
    optimized_instructions = seed_prompt
    gepa_viz_path = output_dir / f"gepa_viz_run_{timestamp}.json"
    prompt_trajectory_path = output_dir / f"prompt_trajectory_{timestamp}.jsonl"

    if not args.skip_gepa:
        GEPA = get_gepa_class()
        metric_fn = create_metric_fn(
            dimension,
            perplexity_scorer=perplexity_scorer,
            min_score=min_score,
            max_score=max_score,
            nla_feedback_provider=nla_feedback_provider,
        )
        gepa_kwargs: dict[str, Any] = {
            "metric": metric_fn,
            "num_threads": args.num_threads,
            "track_stats": True,
            "track_best_outputs": True,
            "add_format_failure_as_feedback": True,
            "reflection_lm": proposer_lm,
        }
        proposer = make_instruction_proposer(args.instruction_proposer, fallback_instruction=seed_prompt)
        if proposer is not None:
            gepa_kwargs["instruction_proposer"] = proposer
        if args.gepa_auto:
            gepa_kwargs["auto"] = args.gepa_auto
        elif args.max_full_evals is not None:
            gepa_kwargs["max_full_evals"] = args.max_full_evals
        elif args.max_metric_calls is not None:
            gepa_kwargs["max_metric_calls"] = args.max_metric_calls

        trainset = make_dspy_examples(train_rows)
        valset = make_dspy_examples(val_rows)
        callback_cm, callback = _make_gepa_viz_callback(
            gepa_viz_path=gepa_viz_path,
            trainset=trainset,
            valset=valset,
        )
        if callback is not None:
            gepa_kwargs["gepa_kwargs"] = {"callbacks": [callback]}
        with callback_cm:
            optimizer = GEPA(**gepa_kwargs)
            try:
                optimized_program = optimizer.compile(student=seed_program, trainset=trainset, valset=valset)
            except TypeError:
                optimized_program = optimizer.compile(seed_program, trainset=trainset, valset=valset)
        optimized_instructions = extract_instructions(optimized_program)
    else:
        trainset = make_dspy_examples(train_rows)
        valset = make_dspy_examples(val_rows)

    baseline_metrics = evaluate_program(seed_program, test_rows, output_dir / f"baseline_predictions_{timestamp}.jsonl")
    summary_rows = [{"program": "baseline", **baseline_metrics}]
    if not args.skip_gepa:
        optimized_metrics = evaluate_program(
            optimized_program,
            test_rows,
            output_dir / f"optimized_predictions_{timestamp}.jsonl",
        )
        summary_rows.append({"program": "optimized", **optimized_metrics})

    write_summary(output_dir / f"metrics_{timestamp}.csv", summary_rows)
    (output_dir / f"optimized_prompt_{timestamp}.txt").write_text(optimized_instructions, encoding="utf-8")
    (output_dir / f"seed_prompt_{timestamp}.txt").write_text(seed_prompt, encoding="utf-8")
    optimized_score = summary_rows[-1].get("agreement") if len(summary_rows) > 1 else None
    if not gepa_viz_path.exists():
        write_fallback_gepa_viz_run(
            gepa_viz_path,
            trainset=trainset,
            valset=valset,
            seed_prompt=seed_prompt,
            optimized_prompt=optimized_instructions,
            optimized_score=float(optimized_score) if isinstance(optimized_score, (int, float)) else None,
        )
    trajectory_count = export_prompt_trajectory(gepa_viz_path, prompt_trajectory_path)
    nla_artifact_path = output_dir / f"nla_verbalizations_{timestamp}.jsonl"
    nla_artifact_count = 0
    if nla_feedback_provider is not None:
        nla_artifact_count = nla_feedback_provider.write_artifact(nla_artifact_path)
    finished_at = datetime.now(timezone.utc)
    (output_dir / f"run_config_{timestamp}.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "dimension": dimension,
                "legacy_label": args.label,
                "seed": args.seed,
                "train_groups": args.train_contexts,
                "val_groups": args.val_contexts,
                "test_groups": args.test_contexts,
                "score_scale": {"min": min_score, "max": max_score},
                "judge_model": args.judge_model,
                "judge_api_base": args.api_base,
                "judge_temperature": args.temperature,
                "nla_av_checkpoint": args.nla_av_checkpoint,
                "nla_extraction_layer": args.nla_extraction_layer,
                "nla_feedback": args.nla_feedback,
                "nla_backend": args.nla_backend if args.nla_feedback else "",
                "nla_precomputed_path": args.nla_precomputed_path if args.nla_feedback else "",
                "nla_max_tokens_per_example": args.nla_max_tokens_per_example if args.nla_feedback else 0,
                "max_tokens": args.max_tokens,
                "proposer_model": args.proposer_model if args.proposer_api_base else args.judge_model,
                "proposer_api_base": args.proposer_api_base if args.proposer_api_base else args.api_base,
                "proposer_temperature": (
                    args.proposer_temperature if args.proposer_api_base else args.temperature
                ),
                "proposer_max_tokens": (
                    args.proposer_max_tokens if args.proposer_api_base else args.max_tokens
                ),
                "proposer_is_separate_lm": bool(args.proposer_api_base),
                "instruction_proposer": args.instruction_proposer,
                "perplexity_feedback": args.perplexity_feedback,
                "perplexity_model": args.judge_model if args.perplexity_feedback else "",
                "perplexity_scope": (
                    "response_only_conditioned_on_context_fact" if args.perplexity_feedback else ""
                ),
                "perplexity_numeric_fields": (
                    ["response_mean_nll", "response_perplexity", "response_token_count"]
                    if args.perplexity_feedback
                    else []
                ),
                "perplexity_prompt_logprobs": args.perplexity_prompt_logprobs if args.perplexity_feedback else 0,
                "split_semantics": {
                    "gepa_train": "Used by GEPA during prompt search.",
                    "gepa_validation": "Used by GEPA for candidate prompt validation/selection.",
                    "final_test": "Never passed to GEPA; evaluated only after the final prompt is selected.",
                },
                "rows": {
                    "gepa_train": len(train_rows),
                    "gepa_validation": len(val_rows),
                    "final_test": len(test_rows),
                },
                "groups": {
                    "gepa_train": len({row.group_id for row in train_rows}),
                    "gepa_validation": len({row.group_id for row in val_rows}),
                    "final_test": len({row.group_id for row in test_rows}),
                },
                "artifacts": {
                    "gepa_viz_run": str(gepa_viz_path),
                    "prompt_trajectory": str(prompt_trajectory_path),
                    "prompt_trajectory_candidates": trajectory_count,
                    "nla_verbalizations": str(nla_artifact_path) if args.nla_feedback else "",
                    "nla_verbalization_rows": nla_artifact_count,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / f"runtime_manifest_{timestamp}.json").write_text(
        json.dumps(
            {
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "elapsed_seconds": round(time.monotonic() - monotonic_started, 3),
                "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
                "slurm_job_name": os.getenv("SLURM_JOB_NAME", ""),
                "slurm_nodelist": os.getenv("SLURM_NODELIST", ""),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "output_dir": str(output_dir),
                "timestamp": timestamp,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _make_gepa_viz_callback(*, gepa_viz_path: Path, trainset: list[Any], valset: list[Any]) -> tuple[Any, Any | None]:
    try:
        from gepa_viz import GepaVizCallback
    except Exception as exc:
        print(f"gepa-viz callback unavailable; writing fallback trajectory only: {type(exc).__name__}: {exc}", flush=True)
        return nullcontext(), None

    callback = GepaVizCallback(
        valset=valset,
        trainset=trainset,
        live=False,
        path=str(gepa_viz_path),
    )
    return callback, callback


if __name__ == "__main__":
    main()
