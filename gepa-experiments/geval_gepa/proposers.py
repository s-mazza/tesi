"""Instruction proposers for GEPA prompt optimization."""

from __future__ import annotations

import re
from typing import Any


FORBIDDEN_PROMPT_SUBSTRINGS = (
    "human mean",
    "predicted score",
    "normalized agreement",
    "metric definition",
    "feedback:",
    "generated_outputs",
    "generated output",
    "inputs:",
    "example ",
    "context_",
    "response_",
    "perplexity",
    "mean_nll",
    "response_token_count",
)

FORBIDDEN_REFLECTION_SUBSTRINGS = (
    "human mean",
    "predicted score",
    "normalized agreement",
    "metric definition",
    "feedback:",
    "generated_outputs",
    "generated output",
    "inputs:",
    "example ",
    "context_",
)

DIMENSION_KEYWORDS = {
    "naturalness": ("naturalness", "natural"),
    "coherence": ("coherence", "coherent"),
    "engagingness": ("engagingness", "engaging"),
    "groundedness": ("groundedness", "grounded", "uses knowledge"),
    "fluency": ("fluency", "fluent"),
    "consistency": ("consistency", "consistent", "factually consistent"),
    "relevance": ("relevance", "relevant"),
}
DIMENSION_CONFLICT_TERMS = {
    "naturalness": ("naturalness",),
    "coherence": ("coherence",),
    "engagingness": ("engagingness",),
    "groundedness": ("groundedness",),
    "fluency": ("fluency",),
    "consistency": ("consistency",),
    "relevance": ("relevance",),
}


class ProposerUnavailableError(RuntimeError):
    """Raised when the requested proposer cannot be constructed."""


def sanitize_proposed_instruction(instruction: str, fallback: str, expected_dimension: str | None = None) -> str:
    """Remove optimizer-feedback artifacts that should never become judge policy."""

    cleaned_lines = []
    for line in instruction.splitlines():
        normalized = line.strip().lower()
        if not normalized:
            cleaned_lines.append(line)
            continue
        if any(token in normalized for token in FORBIDDEN_PROMPT_SUBSTRINGS):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if len(cleaned) < 80 or "score" not in cleaned.lower():
        return fallback
    if expected_dimension and _mentions_wrong_dimension(cleaned, expected_dimension):
        return fallback
    return cleaned


def sanitize_reflection_feedback(feedback: Any) -> str:
    """Keep actionable feedback while dropping labels/scores that can leak into prompts."""

    cleaned_lines = []
    for line in str(feedback).splitlines():
        normalized = line.strip().lower()
        if not normalized:
            continue
        if any(token in normalized for token in FORBIDDEN_REFLECTION_SUBSTRINGS):
            continue
        cleaned_lines.append(line.strip())
    return "\n".join(cleaned_lines) or "No generic feedback available."


def make_instruction_proposer(mode: str, fallback_instruction: str, *, dataset: str, dimension: str) -> Any | None:
    """Create the requested GEPA instruction proposer."""

    if mode == "default":
        return None
    if mode != "generalizing":
        raise ValueError(f"Unknown instruction proposer mode: {mode}")

    try:
        import dspy
        from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample
        from gepa.core.adapter import ProposalFn
    except Exception as exc:  # pragma: no cover - depends on cluster env
        raise ProposerUnavailableError(f"Could not import GEPA proposer APIs: {exc}") from exc

    class ProposeGeneralJudgeInstruction(dspy.Signature):
        """
        You are an expert prompt engineer improving instructions for a G-EVAL-style judge.

        The judge rates generated outputs for one requested evaluation dimension.
        Review the current instruction plus recent inputs, outputs, and feedback.

        CRITICAL RULES:
        1. Turn feedback into generic judging rules for the requested evaluation dimension only.
        2. Do not copy any conversation topic, fact, response text, entity, context id, example id, human score,
           predicted score, metric value, or feedback label into the new instruction.
        3. Do not mention training examples, validation examples, normalized agreement, Pearson, Spearman, MAE,
           or optimizer feedback.
        4. Keep the instruction broadly applicable to unseen examples from the named dataset.
        5. Preserve the required output contract: a brief rationale and a final line formatted as
           Score: <allowed score>.
        """

        dataset = dspy.InputField(desc="Dataset being evaluated.")
        evaluation_dimension = dspy.InputField(desc="The only dimension the judge is allowed to evaluate.")
        current_instruction = dspy.InputField(desc="The instruction currently being used.")
        reflection_data = dspy.InputField(desc="Recent inputs, outputs, and feedback from GEPA.")
        new_instruction = dspy.OutputField(desc="A generic, non-overfit judge instruction for the requested dimension.")

    class GeneralizingJudgeProposer(ProposalFn):
        """GEPA proposer that converts example feedback into generic rubric changes."""

        def __init__(self) -> None:
            self.instruction_improver = dspy.ChainOfThought(ProposeGeneralJudgeInstruction)

        def __call__(
            self,
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[ReflectiveExample]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            updated_components = {}
            for component_name in components_to_update:
                if component_name not in candidate or component_name not in reflective_dataset:
                    continue

                current_instruction = candidate[component_name]
                reflection_text = _format_reflection_examples(reflective_dataset[component_name])
                result = self.instruction_improver(
                    dataset=dataset,
                    evaluation_dimension=dimension,
                    current_instruction=current_instruction,
                    reflection_data=reflection_text,
                )
                proposed = str(getattr(result, "new_instruction", ""))
                updated_components[component_name] = sanitize_proposed_instruction(
                    proposed,
                    fallback=current_instruction or fallback_instruction,
                    expected_dimension=dimension,
                )

            return updated_components

    return GeneralizingJudgeProposer()


def _mentions_wrong_dimension(instruction: str, expected_dimension: str) -> bool:
    normalized = instruction.lower()
    expected = expected_dimension.strip().lower().replace("-", "_")
    other_terms = [
        term
        for dimension, terms in DIMENSION_CONFLICT_TERMS.items()
        if dimension != expected
        for term in terms
    ]
    return any(term in normalized for term in other_terms)


def _format_reflection_examples(examples: list[Any]) -> str:
    chunks = []
    for index, example in enumerate(examples, start=1):
        inputs = _mapping_get(example, "Inputs", {})
        output = _mapping_get(example, "Generated_Outputs", "")
        feedback = _mapping_get(example, "Feedback", "No feedback")
        chunks.append(
            "\n".join(
                [
                    f"--- Reflection {index} ---",
                    f"Input summary: {_summarize_inputs(inputs)}",
                    f"Judge output summary: {_summarize_judge_output(output)}",
                    f"Generic feedback: {sanitize_reflection_feedback(feedback)}",
                ]
            )
        )
    return "\n\n".join(chunks)


def _summarize_inputs(inputs: Any) -> str:
    if not hasattr(inputs, "items"):
        return "inputs are not a mapping"
    parts = []
    for key, value in inputs.items():
        text = str(value)
        if key in {"context", "fact", "response"}:
            parts.append(f"{key}_chars={len(text)}")
        else:
            parts.append(f"{key}=redacted")
    return ", ".join(parts) or "no inputs"


def _summarize_judge_output(output: Any) -> str:
    raw_output = str(output)
    score_text = str(_mapping_get(output, "score", raw_output))
    rationale_text = str(_mapping_get(output, "rationale", ""))
    parsed_score = _parse_score(score_text) or _parse_score(raw_output)
    fields = {
        "score_parse": "ok" if parsed_score is not None else "missing_or_ambiguous",
        "score_bucket": _score_bucket(parsed_score),
        "output_length": _length_bucket(_word_count(raw_output)),
        "rationale_length": _length_bucket(_word_count(rationale_text)) if rationale_text else "unknown",
    }
    return "; ".join(f"{key}={value}" for key, value in fields.items())


def _parse_score(text: str) -> int | None:
    matches = [int(match) for match in re.findall(r"(?<!\d)([1-3])(?!\d)", text)]
    if len(matches) == 1:
        return matches[0]
    return None


def _score_bucket(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= 1:
        return "low"
    if score >= 3:
        return "high"
    return "middle"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _length_bucket(count: int) -> str:
    if count <= 8:
        return "short"
    if count <= 40:
        return "medium"
    return "long"


def _mapping_get(value: Any, key: str, default: Any) -> Any:
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)
