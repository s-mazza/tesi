from __future__ import annotations

import unittest

from geval_gepa.data import load_usr_examples, split_by_context
from geval_gepa.metrics import compute_regression_metrics, normalized_absolute_score, parse_discrete_score
from geval_gepa.proposers import _format_reflection_examples, sanitize_proposed_instruction, sanitize_reflection_feedback
from geval_gepa.runner import _abstract_rubric_signals


FIXTURE = [
    {
        "context": "hello\nwhat topic should we discuss?",
        "fact": "the fact says jazz is popular.",
        "annotators": ["a", "b", "c"],
        "responses": [
            {
                "response": "jazz sounds like a good topic.",
                "model": "Original Ground Truth",
                "Understandable": [1, 1, 1],
                "Natural": [3, 3, 2],
                "Maintains Context": [3, 2, 2],
                "Engaging": [3, 2, 1],
                "Uses Knowledge": [1, 0, 1],
                "Overall": [5, 4, 3],
            },
            {
                "response": "yes.",
                "model": "Argmax Decoding",
                "Understandable": [1, 1, 0],
                "Natural": [2, 2, 1],
                "Maintains Context": [1, 1, 1],
                "Engaging": [1, 1, 1],
                "Uses Knowledge": [0, 0, 0],
                "Overall": [2, 2, 1],
            },
        ],
    },
    {
        "context": "do you like football?",
        "fact": "_nofact",
        "annotators": ["a", "b", "c"],
        "responses": [
            {
                "response": "football can be exciting in the playoffs.",
                "model": "New Human Generated",
                "Understandable": [1, 1, 1],
                "Natural": [3, 3, 3],
                "Maintains Context": [3, 3, 3],
                "Engaging": [3, 3, 3],
                "Uses Knowledge": [1, 1, 1],
                "Overall": [5, 5, 5],
            }
        ],
    },
]


class DataAndMetricsTest(unittest.TestCase):
    def test_loads_response_level_examples(self) -> None:
        examples = load_usr_examples_from_fixture()

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0].context_id, "context_000")
        self.assertAlmostEqual(examples[0].human_score("Engaging"), 2.0)
        self.assertAlmostEqual(examples[0].human_score("Uses Knowledge"), 2 / 3)

    def test_split_keeps_contexts_disjoint(self) -> None:
        examples = load_usr_examples_from_fixture()
        train, val, test = split_by_context(
            examples,
            train_contexts=1,
            val_contexts=1,
            test_contexts=0,
            seed=42,
        )

        train_ids = {row.context_id for row in train}
        val_ids = {row.context_id for row in val}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(len(train) + len(val) + len(test), 3)

    def test_parse_discrete_score(self) -> None:
        self.assertEqual(parse_discrete_score("Score: 3", min_score=1, max_score=3), 3)
        self.assertEqual(parse_discrete_score("2", min_score=1, max_score=3), 2)
        self.assertIsNone(parse_discrete_score("Score: 4", min_score=1, max_score=3))
        self.assertIsNone(parse_discrete_score("Score: 0", min_score=1, max_score=3))
        self.assertIsNone(parse_discrete_score("Scores 1 and 3", min_score=1, max_score=3))

    def test_normalized_score_and_correlations(self) -> None:
        self.assertEqual(normalized_absolute_score(3, 3, min_score=1, max_score=3), 1.0)
        self.assertEqual(normalized_absolute_score(1, 3, min_score=1, max_score=3), 0.0)

        metrics = compute_regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(metrics.n, 3)
        self.assertAlmostEqual(metrics.pearson, 1.0)
        self.assertAlmostEqual(metrics.spearman, 1.0)
        self.assertAlmostEqual(metrics.mae, 0.0)

    def test_sanitizes_optimizer_feedback_from_proposed_instruction(self) -> None:
        fallback = "Rate Engagingness from 1 to 3 and return Score: <1, 2, or 3>."
        proposed = "\n".join(
            [
                "Rate the candidate response for Engagingness on a 1 to 3 scale.",
                "Human mean Engaging score: 2.33; predicted score: 2.",
                "Use specific, conversation-advancing content to separate 2 from 3.",
                "Return a final line formatted as Score: <1, 2, or 3>.",
            ]
        )

        cleaned = sanitize_proposed_instruction(proposed, fallback=fallback)

        self.assertIn("conversation-advancing", cleaned)
        self.assertNotIn("Human mean", cleaned)
        self.assertNotIn("predicted score", cleaned)

    def test_sanitizes_reflection_feedback_for_proposer(self) -> None:
        feedback = "\n".join(
            [
                "Human mean Engaging score: 2.33; predicted score: 2; normalized agreement: 0.83.",
                "Metric definition: Engagingness on a 1-3 scale.",
                "Abstract error summary: target_bucket=high; judge_bucket=middle; error_direction=underrated; agreement_bucket=medium.",
                "Rubric signals: candidate_length=medium; asks_question=yes; generic_acknowledgement_only=no.",
                "The response was underrated. Revise the judging instructions to reward specific follow-up questions.",
            ]
        )

        cleaned = sanitize_reflection_feedback(feedback)

        self.assertIn("underrated", cleaned)
        self.assertIn("target_bucket=high", cleaned)
        self.assertIn("asks_question=yes", cleaned)
        self.assertNotIn("Human mean", cleaned)
        self.assertNotIn("Metric definition", cleaned)

    def test_reflection_examples_use_abstract_trace_summaries(self) -> None:
        reflection_text = _format_reflection_examples(
            [
                {
                    "Inputs": {
                        "context": "Alice and Bob discuss astronomy.",
                        "fact": "A comet was visible in 2020.",
                        "response": "That is interesting. Do you watch meteor showers?",
                    },
                    "Generated_Outputs": {
                        "rationale": "The reply asks a relevant question.",
                        "score": "Score: 3",
                    },
                    "Feedback": (
                        "Abstract error summary: target_bucket=high; judge_bucket=high; "
                        "error_direction=close; agreement_bucket=high."
                    ),
                }
            ]
        )

        self.assertIn("context_chars=", reflection_text)
        self.assertIn("Judge output summary:", reflection_text)
        self.assertIn("score_bucket=high", reflection_text)
        self.assertIn("target_bucket=high", reflection_text)
        self.assertNotIn("Alice", reflection_text)
        self.assertNotIn("meteor showers", reflection_text)

    def test_abstract_rubric_signals_do_not_copy_response_text(self) -> None:
        examples = load_usr_examples_from_fixture()
        signals = _abstract_rubric_signals(examples[0])

        self.assertIn("candidate_length=", signals)
        self.assertIn("fact_available=yes", signals)
        self.assertNotIn("jazz", signals)


def load_usr_examples_from_fixture():
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fixture.json"
        path.write_text(json.dumps(FIXTURE), encoding="utf-8")
        return load_usr_examples(path)


if __name__ == "__main__":
    unittest.main()
