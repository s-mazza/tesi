"""Prompt templates for G-EVAL-style response and summary judging."""

ENGAGING_SEED_INSTRUCTIONS = """You are evaluating one potential next response in a dialogue.

Your task is to rate the response on Engagingness using a 1 to 3 scale.

Evaluation Criteria:
Engagingness (1-3): Is the response dull or interesting?
- 1 means dull: the response is generic, repetitive, evasive, or unlikely to continue the conversation.
- 2 means somewhat interesting: the response is acceptable and may continue the conversation, but it is not especially vivid, specific, or curiosity-provoking.
- 3 means interesting: the response is very interesting, asks or answers something specific, adds a relevant thought, or presents an interesting fact.

Evaluation Steps:
1. Read the conversation history to understand the current topic and conversational direction.
2. Read the provided fact. If the fact is "_nofact", judge whether the response is engaging without requiring external knowledge use.
3. Read the candidate response and decide whether it would keep a human conversational partner interested.
4. Assign exactly one score from 1, 2, or 3.

Return a brief rationale and then a final line in this exact format:
Score: <1, 2, or 3>

Do not output scores outside this scale. Never output 0, 4, or 5 for Engagingness."""


DIMENSION_DESCRIPTIONS = {
    "naturalness": "Naturalness: whether the response sounds fluent, natural, and human-like.",
    "coherence": "Coherence: whether the output is logically organized and maintains context.",
    "engagingness": "Engagingness: whether the response captures interest and moves the conversation forward.",
    "groundedness": "Groundedness: whether the response uses provided knowledge or source facts appropriately.",
    "fluency": "Fluency: whether the summary is well-written and grammatical.",
    "consistency": "Consistency: whether the output is factually consistent with the source.",
    "relevance": "Relevance: whether the summary captures important source content without irrelevant material.",
}


def seed_instructions(*, dataset: str, dimension: str, min_score: int, max_score: int) -> str:
    if dataset == "topical_chat" and dimension == "engagingness" and min_score == 1 and max_score == 3:
        return ENGAGING_SEED_INSTRUCTIONS

    description = metric_description(dimension)
    source_label = "conversation history" if dataset == "topical_chat" else "source document"
    candidate_label = "candidate response" if dataset == "topical_chat" else "candidate summary"
    return f"""You are evaluating one generated output for a G-EVAL-style benchmark.

Your task is to rate the {candidate_label} on {dimension} using a {min_score} to {max_score} scale.

Evaluation Criteria:
{description}
- {min_score} means the output performs very poorly on this dimension.
- Middle scores mean the output is partially acceptable but has important weaknesses.
- {max_score} means the output performs very well on this dimension.

Evaluation Steps:
1. Read the {source_label} carefully.
2. Read the reference or background fact if it is provided. If it is "_nofact", do not require extra grounding beyond the source.
3. Read the {candidate_label}.
4. Compare the output against the requested dimension only.
5. Assign exactly one score from {min_score} to {max_score}.

Return a brief rationale and then a final line in this exact format:
Score: <{min_score} to {max_score}>

Do not output scores outside this scale."""


def metric_description(label: str) -> str:
    key = label.strip().lower().replace("-", "_")
    if label == "Engaging":
        return "Engagingness on a 1-3 scale, mapped to the USR 'Engaging' annotations."
    if key in DIMENSION_DESCRIPTIONS:
        return DIMENSION_DESCRIPTIONS[key]
    raise ValueError(f"No prompt description is defined for label/dimension {label!r}")
