"""Instruction proposers for GEPA prompt optimization."""

from __future__ import annotations

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
)


class ProposerUnavailableError(RuntimeError):
    """Raised when the requested proposer cannot be constructed."""


def sanitize_proposed_instruction(instruction: str, fallback: str) -> str:
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
    return cleaned


def sanitize_reflection_feedback(feedback: Any) -> str:
    """Keep actionable feedback while dropping labels/scores that can leak into prompts."""

    cleaned_lines = []
    for line in str(feedback).splitlines():
        normalized = line.strip().lower()
        if not normalized:
            continue
        if any(token in normalized for token in FORBIDDEN_PROMPT_SUBSTRINGS):
            continue
        cleaned_lines.append(line.strip())
    return "\n".join(cleaned_lines) or "No generic feedback available."


def make_instruction_proposer(mode: str, fallback_instruction: str) -> Any | None:
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
        You are an expert prompt engineer improving instructions for a dialogue-response judge.

        The judge rates a candidate response's Engagingness on a 1 to 3 scale.
        Review the current instruction plus recent inputs, outputs, and feedback.

        CRITICAL RULES:
        1. Turn feedback into generic judging rules for Engagingness only.
        2. Do not copy any conversation topic, fact, response text, entity, context id, example id, human score,
           predicted score, metric value, or feedback label into the new instruction.
        3. Do not mention training examples, validation examples, normalized agreement, Pearson, Spearman, MAE,
           or optimizer feedback.
        4. Keep the instruction broadly applicable to unseen Topical-Chat responses.
        5. Preserve the required output contract: a brief rationale and a final line formatted as
           Score: <1, 2, or 3>.
        """

        current_instruction = dspy.InputField(desc="The instruction currently being used.")
        reflection_data = dspy.InputField(desc="Recent inputs, outputs, and feedback from GEPA.")
        new_instruction = dspy.OutputField(desc="A generic, non-overfit Engagingness judge instruction.")

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
                    current_instruction=current_instruction,
                    reflection_data=reflection_text,
                )
                proposed = str(getattr(result, "new_instruction", ""))
                updated_components[component_name] = sanitize_proposed_instruction(
                    proposed,
                    fallback=current_instruction or fallback_instruction,
                )

            return updated_components

    return GeneralizingJudgeProposer()


def _format_reflection_examples(examples: list[Any]) -> str:
    chunks = []
    for index, example in enumerate(examples, start=1):
        inputs = _mapping_get(example, "Inputs", {})
        feedback = _mapping_get(example, "Feedback", "No feedback")
        chunks.append(
            "\n".join(
                [
                    f"--- Reflection {index} ---",
                    f"Input summary: {_summarize_inputs(inputs)}",
                    "Generated output text omitted to avoid copying example-specific wording.",
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


def _mapping_get(value: Any, key: str, default: Any) -> Any:
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)
