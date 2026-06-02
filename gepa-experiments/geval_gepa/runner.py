"""Run GEPA prompt optimization for a G-EVAL-style Topical-Chat judge."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .data import DEFAULT_USR_URL, LABEL_SCALES, UsrResponseExample, load_usr_examples, split_by_context
from .metrics import compute_regression_metrics, normalized_absolute_score, parse_discrete_score
from .prompts import ENGAGING_SEED_INSTRUCTIONS, metric_description
from .proposers import make_instruction_proposer


def make_program(instructions: str) -> Any:
    import dspy

    class JudgeSignature(dspy.Signature):
        context: str = dspy.InputField(desc="Dialogue history before the candidate response.")
        fact: str = dspy.InputField(desc='Relevant knowledge sentence, or "_nofact" when no fact is provided.')
        response: str = dspy.InputField(desc="Candidate next response to evaluate.")
        rationale: str = dspy.OutputField(desc="Brief explanation of the Engagingness score.")
        score: str = dspy.OutputField(desc="Exactly one integer from 1, 2, or 3.")

    class TopicalChatJudge(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.judge = dspy.ChainOfThought(JudgeSignature.with_instructions(instructions))

        def forward(self, context: str, fact: str, response: str) -> Any:
            return self.judge(context=context, fact=fact, response=response)

    return TopicalChatJudge()


def make_dspy_examples(rows: list[UsrResponseExample], label: str) -> list[Any]:
    import dspy

    examples = []
    for row in rows:
        examples.append(
            dspy.Example(
                context=row.context,
                fact=row.fact,
                response=row.response,
                human_score=row.human_score(label),
                context_id=row.context_id,
                response_id=row.response_id,
                model=row.model,
            ).with_inputs("context", "fact", "response")
        )
    return examples


def create_metric_fn(label: str) -> Callable[..., Any]:
    import dspy

    min_score, max_score = LABEL_SCALES[label]

    def metric_fn(example: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None) -> Any:
        del trace, pred_name, pred_trace
        parsed = parse_discrete_score(pred, min_score=min_score, max_score=max_score)
        target = float(example.human_score)

        if parsed is None:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"FORMAT ERROR: score must be exactly one integer from {min_score} to {max_score}. "
                    f"Human mean for {label} is {target:.2f}."
                ),
            )

        score = normalized_absolute_score(parsed, target, min_score=min_score, max_score=max_score)
        delta = parsed - target
        feedback = [
            f"Human mean {label} score: {target:.2f}; predicted score: {parsed}; normalized agreement: {score:.3f}.",
            f"Metric definition: {metric_description(label)}",
        ]
        if abs(delta) >= 1.0:
            direction = "overrated" if delta > 0 else "underrated"
            feedback.append(
                f"The response was {direction}. Revise the judging instructions to better distinguish generic replies "
                "from responses that add specific, conversation-advancing content."
            )
        else:
            feedback.append("The predicted score is close to the aggregated human annotation.")

        return dspy.Prediction(score=score, feedback="\n".join(feedback))

    return metric_fn


def configure_lm(args: argparse.Namespace) -> Any:
    import dspy

    lm = dspy.LM(
        model=f"openai/{args.judge_model}",
        api_base=args.api_base,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if hasattr(dspy, "configure"):
        dspy.configure(lm=lm)
    else:
        dspy.settings.configure(lm=lm)
    return lm


def get_gepa_class() -> Any:
    import dspy

    if hasattr(dspy, "GEPA"):
        return dspy.GEPA
    try:
        from dspy.teleprompt import GEPA

        return GEPA
    except ImportError as exc:
        raise ImportError("GEPA is unavailable. Install compatible dspy/gepa packages.") from exc


def evaluate_program(program: Any, rows: list[UsrResponseExample], label: str, output_path: Path) -> dict[str, Any]:
    min_score, max_score = LABEL_SCALES[label]
    predictions: list[float] = []
    targets: list[float] = []
    agreement_scores: list[float] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            pred = program(context=row.context, fact=row.fact, response=row.response)
            parsed = parse_discrete_score(pred, min_score=min_score, max_score=max_score)
            prediction = float(parsed) if parsed is not None else float("nan")
            target = row.human_score(label)
            if parsed is not None:
                predictions.append(prediction)
                targets.append(target)
                agreement_scores.append(normalized_absolute_score(prediction, target, min_score=min_score, max_score=max_score))
            else:
                agreement_scores.append(0.0)
            handle.write(
                json.dumps(
                    {
                        "context_id": row.context_id,
                        "response_id": row.response_id,
                        "model": row.model,
                        "label": label,
                        "target": target,
                        "prediction": parsed,
                        "raw_score": str(getattr(pred, "score", "")),
                        "raw_rationale": str(getattr(pred, "rationale", "")),
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
    parser.add_argument("--label", default="Engaging", choices=sorted(LABEL_SCALES))
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


def main() -> None:
    args = parse_args()
    if args.label != "Engaging":
        raise ValueError("Only the Engaging prompt is implemented in this v1.")

    output_dir = Path(args.output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = load_usr_examples(args.data_source)
    train_rows, val_rows, test_rows = split_by_context(
        rows,
        train_contexts=args.train_contexts,
        val_contexts=args.val_contexts,
        test_contexts=args.test_contexts,
        seed=args.seed,
    )
    print(
        "Split rows: "
        f"gepa_train={len(train_rows)}, gepa_validation={len(val_rows)}, final_test={len(test_rows)}. "
        "GEPA compile uses only gepa_train/gepa_validation; final_test is evaluated only after optimization.",
        flush=True,
    )

    lm = configure_lm(args)
    seed_program = make_program(ENGAGING_SEED_INSTRUCTIONS)
    optimized_program = seed_program
    optimized_instructions = ENGAGING_SEED_INSTRUCTIONS

    if not args.skip_gepa:
        GEPA = get_gepa_class()
        metric_fn = create_metric_fn(args.label)
        gepa_kwargs: dict[str, Any] = {
            "metric": metric_fn,
            "num_threads": args.num_threads,
            "track_stats": True,
            "track_best_outputs": True,
            "add_format_failure_as_feedback": True,
            "reflection_lm": lm,
        }
        proposer = make_instruction_proposer(args.instruction_proposer, fallback_instruction=ENGAGING_SEED_INSTRUCTIONS)
        if proposer is not None:
            gepa_kwargs["instruction_proposer"] = proposer
        if args.gepa_auto:
            gepa_kwargs["auto"] = args.gepa_auto
        elif args.max_full_evals is not None:
            gepa_kwargs["max_full_evals"] = args.max_full_evals
        elif args.max_metric_calls is not None:
            gepa_kwargs["max_metric_calls"] = args.max_metric_calls

        optimizer = GEPA(**gepa_kwargs)
        trainset = make_dspy_examples(train_rows, args.label)
        valset = make_dspy_examples(val_rows, args.label)
        try:
            optimized_program = optimizer.compile(student=seed_program, trainset=trainset, valset=valset)
        except TypeError:
            optimized_program = optimizer.compile(seed_program, trainset=trainset, valset=valset)
        optimized_instructions = extract_instructions(optimized_program)

    baseline_metrics = evaluate_program(seed_program, test_rows, args.label, output_dir / f"baseline_predictions_{timestamp}.jsonl")
    summary_rows = [{"program": "baseline", **baseline_metrics}]
    if not args.skip_gepa:
        optimized_metrics = evaluate_program(
            optimized_program,
            test_rows,
            args.label,
            output_dir / f"optimized_predictions_{timestamp}.jsonl",
        )
        summary_rows.append({"program": "optimized", **optimized_metrics})

    write_summary(output_dir / f"metrics_{timestamp}.csv", summary_rows)
    (output_dir / f"optimized_prompt_{timestamp}.txt").write_text(optimized_instructions, encoding="utf-8")
    (output_dir / f"run_config_{timestamp}.json").write_text(
        json.dumps(
            {
                "label": args.label,
                "seed": args.seed,
                "train_contexts": args.train_contexts,
                "val_contexts": args.val_contexts,
                "test_contexts": args.test_contexts,
                "judge_model": args.judge_model,
                "nla_av_checkpoint": args.nla_av_checkpoint,
                "nla_extraction_layer": args.nla_extraction_layer,
                "max_tokens": args.max_tokens,
                "instruction_proposer": args.instruction_proposer,
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
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
