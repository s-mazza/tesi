from __future__ import annotations

import json
import unittest
from pathlib import Path
from argparse import Namespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from geval_gepa.aux_judge import AuxJudgeFeedbackProvider
from geval_gepa.data import load_usr_examples, split_by_context
from geval_gepa.metrics import compute_regression_metrics, normalized_absolute_score, parse_discrete_score
from geval_gepa.nla_feedback import NlaFeedbackProvider
from geval_gepa.nla_precompute import SemanticTokenSelector, fake_activation_rows, iter_verbalization_rows, parse_explanation
from geval_gepa.perplexity import PerplexityResult
from geval_gepa.preflight import build_report as build_preflight_report
from geval_gepa.proposers import _format_reflection_examples, sanitize_proposed_instruction, sanitize_reflection_feedback
from geval_gepa.runner import (
    _abstract_rubric_signals,
    _validate_aux_judge_feedback_success,
    _validate_nla_feedback_ready,
    configure_lms,
    create_metric_fn,
    evaluate_program,
)
from geval_gepa.tasks import get_task, split_examples
from geval_gepa.trajectory import export_prompt_trajectory

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aggregate_results import build_rows  # noqa: E402
from diagnose_nla_run import build_report as build_nla_diagnostic_report  # noqa: E402
from experimental_build_nla_precomputed import ExperimentalTokenSelector  # noqa: E402
from export_nla_manifest import build_manifest  # noqa: E402


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

    def test_task_registry_loads_topical_chat_dimensions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tc.json"
            path.write_text(json_dump(FIXTURE), encoding="utf-8")

            task = get_task("Topical-Chat")
            examples = task.load(path, "groundedness")

        self.assertEqual(task.task_type, "dialogue")
        self.assertEqual(examples[0].dimension, "groundedness")
        self.assertEqual(examples[0].min_score, 0)
        self.assertEqual(examples[0].max_score, 1)
        self.assertAlmostEqual(examples[0].human_score, 2 / 3)

    def test_task_split_keeps_groups_disjoint(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tc.json"
            path.write_text(json_dump(FIXTURE), encoding="utf-8")
            examples = get_task("usr").load(path, "engagingness")

        train, val, test = split_examples(examples, train_groups=1, val_groups=1, test_groups=0, seed=7)
        self.assertFalse({row.group_id for row in train} & {row.group_id for row in val})
        self.assertEqual(len(train) + len(val) + len(test), len(examples))

    def test_task_registry_loads_summeval_fixture(self) -> None:
        records = [
            {
                "doc_id": "doc1",
                "source_article": "Article text",
                "candidate_summary": "Candidate summary",
                "reference_summary": "Reference summary",
                "system_id": "sysA",
                "coherence": [4, 5],
            }
        ]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summeval.json"
            path.write_text(json_dump(records), encoding="utf-8")
            examples = get_task("summeval").load(path, "coherence")

        self.assertEqual(examples[0].dataset, "summeval")
        self.assertEqual(examples[0].group_id, "doc1")
        self.assertAlmostEqual(examples[0].human_score, 4.5)
        self.assertEqual(examples[0].min_score, 1)
        self.assertEqual(examples[0].max_score, 5)

    def test_task_registry_loads_summeval_nested_scores(self) -> None:
        records = [
            {
                "doc_id": "doc1",
                "source": "Article text",
                "system_output": "Candidate summary",
                "reference": "Reference summary",
                "system_id": "sysA",
                "scores": {"consistency": 2.5, "coherence": 3.0},
            }
        ]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summeval.json"
            path.write_text(json_dump(records), encoding="utf-8")
            examples = get_task("summeval").load(path, "consistency")

        self.assertAlmostEqual(examples[0].human_score, 2.5)

    def test_task_registry_loads_qags_fixture(self) -> None:
        records = [
            {
                "doc_id": "doc1",
                "source": "Article text",
                "summary": "Generated summary",
                "system": "sysA",
                "consistency": 4,
            }
        ]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qags_cnn.json"
            path.write_text(json_dump(records), encoding="utf-8")
            examples = get_task("qags-cnn").load(path, "consistency")

        self.assertEqual(examples[0].dataset, "qags_cnn")
        self.assertEqual(examples[0].dimension, "consistency")
        self.assertAlmostEqual(examples[0].human_score, 4.0)

    def test_task_registry_loads_qags_mturk_jsonl_fixture(self) -> None:
        records = [
            {
                "article": "Article text",
                "summary_sentences": [
                    {
                        "sentence": "Sentence one.",
                        "responses": [
                            {"worker_id": 0, "response": "yes"},
                            {"worker_id": 1, "response": "no"},
                        ],
                    },
                    {
                        "sentence": "Sentence two.",
                        "responses": [{"worker_id": 2, "response": "yes"}],
                    },
                ],
            }
        ]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qags_cnn.jsonl"
            path.write_text("\n".join(json_dump(record) for record in records) + "\n", encoding="utf-8")
            examples = get_task("qags-cnn").load(path, "consistency")

        self.assertEqual(examples[0].candidate_output, "Sentence one. Sentence two.")
        self.assertAlmostEqual(examples[0].human_score, 1.0 + 4.0 * (2 / 3))

    def test_light_preflight_uses_task_registry_for_summeval(self) -> None:
        records = [
            {
                "doc_id": "doc1",
                "source": "Article text",
                "system_output": "Candidate summary",
                "reference": "Reference summary",
                "scores": {"consistency": 2.5},
            }
        ]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summeval.json"
            path.write_text(json_dump(records), encoding="utf-8")
            report = build_preflight_report(
                Namespace(
                    data_source=str(path),
                    dataset="summeval",
                    dimension="consistency",
                    judge_model="Qwen/Qwen2.5-7B-Instruct",
                    nla_av_checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                    hf_home=tmpdir,
                    train_contexts=1,
                    val_contexts=0,
                    test_contexts=0,
                    seed=42,
                )
            )

        self.assertEqual(report["dataset"], "summeval")
        self.assertTrue(report["ok_dataset"])
        self.assertTrue(report["ok_split_group_counts"])

    def test_light_preflight_uses_task_registry_for_qags_jsonl(self) -> None:
        record = {
            "article": "Article text",
            "summary_sentences": [
                {
                    "sentence": "Sentence one.",
                    "responses": [{"worker_id": 0, "response": "yes"}],
                }
            ],
        }
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qags.jsonl"
            path.write_text(json_dump(record) + "\n", encoding="utf-8")
            report = build_preflight_report(
                Namespace(
                    data_source=str(path),
                    dataset="qags_cnn",
                    dimension="consistency",
                    judge_model="Qwen/Qwen2.5-7B-Instruct",
                    nla_av_checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                    hf_home=tmpdir,
                    train_contexts=1,
                    val_contexts=0,
                    test_contexts=0,
                    seed=42,
                )
            )

        self.assertEqual(report["dataset"], "qags_cnn")
        self.assertTrue(report["ok_dataset"])
        self.assertTrue(report["ok_split_group_counts"])

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
        self.assertAlmostEqual(metrics.kendall_tau, 1.0)
        self.assertAlmostEqual(metrics.mae, 0.0)

    def test_kendall_tau_b_handles_ties(self) -> None:
        metrics = compute_regression_metrics([1, 1, 2, 3], [1, 2, 2, 3])

        self.assertEqual(metrics.n, 4)
        self.assertAlmostEqual(metrics.kendall_tau, 0.8)

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

    def test_sanitizes_proposed_instruction_with_wrong_dimension(self) -> None:
        fallback = "Rate the candidate summary on consistency from 1 to 3 and return Score: <1, 2, or 3>."
        proposed = "\n".join(
            [
                "Rate the candidate summary on engagingness using a 1 to 3 scale.",
                "Reward vivid summaries that capture reader interest.",
                "Return a final line formatted as Score: <1, 2, or 3>.",
            ]
        )

        cleaned = sanitize_proposed_instruction(proposed, fallback=fallback, expected_dimension="consistency")

        self.assertEqual(cleaned, fallback)

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

    def test_perplexity_feedback_reaches_reflection_but_not_prompt(self) -> None:
        feedback = "\n".join(
            [
                "Perplexity signals: response_mean_nll=2.7310; response_perplexity=15.3500; response_token_count=18.",
                "Rubric signals: candidate_length=medium; asks_question=yes.",
            ]
        )

        cleaned_feedback = sanitize_reflection_feedback(feedback)

        self.assertIn("response_mean_nll=2.7310", cleaned_feedback)
        self.assertIn("response_perplexity=15.3500", cleaned_feedback)

        cleaned_prompt = sanitize_proposed_instruction(
            "\n".join(
                [
                    "Rate Engagingness from 1 to 3.",
                    "Use response_perplexity=15.3500 to decide the score.",
                    "Return a final line formatted as Score: <1, 2, or 3>.",
                ]
            ),
            fallback="Rate Engagingness from 1 to 3 and return Score: <1, 2, or 3>.",
        )

        self.assertNotIn("response_perplexity", cleaned_prompt)

    def test_metric_feedback_can_include_numeric_perplexity(self) -> None:
        class FakeScorer:
            def score_example(self, example):
                return PerplexityResult(mean_nll=2.731, perplexity=15.35, token_count=18)

        class FakeDspy:
            class Prediction:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

        example = load_usr_examples_from_fixture()[0]
        with patch.dict("sys.modules", {"dspy": FakeDspy}):
            metric_fn = create_metric_fn("Engaging", perplexity_scorer=FakeScorer())
            result = metric_fn(
                type(
                    "Example",
                    (),
                    {
                        "human_score": example.human_score("Engaging"),
                        "context": example.context,
                        "fact": example.fact,
                        "response": example.response,
                        "context_id": example.context_id,
                        "response_id": example.response_id,
                    },
                )(),
                type("Pred", (), {"score": "2"})(),
            )

        self.assertIn("response_mean_nll=2.7310", result.feedback)
        self.assertIn("response_perplexity=15.3500", result.feedback)
        self.assertIn("response_token_count=18", result.feedback)

    def test_metric_feedback_can_include_precomputed_nla_verbalization(self) -> None:
        example = load_usr_examples_from_fixture()[0]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nla.jsonl"
            path.write_text(
                json_dump(
                    {
                        "example_id": example.response_id,
                        "token_position": "candidate_3",
                        "token_text": "jazz",
                        "explanation": "The activation focuses on a concrete music topic.",
                        "parse_status": "ok",
                        "layer": 20,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider = NlaFeedbackProvider(
                checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                layer=20,
                backend="precomputed",
                max_tokens_per_example=3,
                precomputed_path=str(path),
            )

        class FakeDspy:
            class Prediction:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

        with patch.dict("sys.modules", {"dspy": FakeDspy}):
            metric_fn = create_metric_fn(
                "Engaging",
                min_score=1,
                max_score=3,
                nla_feedback_provider=provider,
            )
            result = metric_fn(
                type(
                    "Example",
                    (),
                    {
                        "human_score": example.human_score("Engaging"),
                        "context": example.context,
                        "fact": example.fact,
                        "response": example.response,
                        "context_id": example.context_id,
                        "response_id": example.response_id,
                        "example_id": example.response_id,
                    },
                )(),
                type("Pred", (), {"score": "2"})(),
            )

        self.assertIn("NLA multi-token verbalizations", result.feedback)
        self.assertIn("concrete music topic", result.feedback)

    def test_metric_feedback_can_include_group_shared_nla_verbalization(self) -> None:
        example = load_usr_examples_from_fixture()[0]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nla.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json_dump(
                            {
                                "example_id": example.response_id,
                                "group_id": example.context_id,
                                "token_position": "candidate_middle",
                                "token_text": "jazz",
                                "explanation": "The candidate activation focuses on a concrete music topic.",
                                "parse_status": "ok",
                                "token_status": "ok",
                                "layer": 20,
                            }
                        ),
                        json_dump(
                            {
                                "example_id": f"__group__:{example.context_id}",
                                "group_id": example.context_id,
                                "token_position": "experimental_hybrid_context_dedup_6_context_source_middle",
                                "token_text": "discuss",
                                "explanation": "The shared context activation preserves topic grounding.",
                                "parse_status": "ok",
                                "token_status": "ok",
                                "layer": 20,
                                "shared_group_feedback": True,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            provider = NlaFeedbackProvider(
                checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                layer=20,
                backend="precomputed",
                max_tokens_per_example=3,
                precomputed_path=str(path),
            )

        feedback = provider.feedback_for(
            type(
                "Example",
                (),
                {
                    "example_id": example.response_id,
                    "group_id": example.context_id,
                    "dataset": "topical_chat",
                    "dimension": "engagingness",
                },
            )()
        )

        self.assertIn("concrete music topic", feedback)
        self.assertIn("shared context activation", feedback)

    def test_nla_precomputed_validation_rejects_missing_examples(self) -> None:
        example = load_usr_examples_from_fixture()[0]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nla.jsonl"
            path.write_text("", encoding="utf-8")
            provider = NlaFeedbackProvider(
                checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                layer=20,
                backend="precomputed",
                max_tokens_per_example=3,
                precomputed_path=str(path),
            )

        dspy_example = type("Example", (), {"example_id": example.response_id})()
        with self.assertRaisesRegex(ValueError, "coverage is below threshold"):
            _validate_nla_feedback_ready(provider, [dspy_example], min_coverage=0.95)

    def test_precomputed_nla_missing_rows_do_not_fall_back_to_dry_run(self) -> None:
        example = load_usr_examples_from_fixture()[0]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nla.jsonl"
            artifact_path = Path(tmpdir) / "emitted.jsonl"
            path.write_text("", encoding="utf-8")
            provider = NlaFeedbackProvider(
                checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                layer=20,
                backend="precomputed",
                max_tokens_per_example=3,
                precomputed_path=str(path),
            )

            feedback = provider.feedback_for(type("Example", (), {"example_id": example.response_id})())
            emitted_count = provider.write_artifact(artifact_path)

        self.assertIn("NLA verbalizations unavailable", feedback)
        self.assertEqual(emitted_count, 0)
        self.assertFalse(artifact_path.exists())

    def test_nla_precomputed_validation_accepts_useful_rows(self) -> None:
        example = load_usr_examples_from_fixture()[0]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nla.jsonl"
            path.write_text(
                json_dump(
                    {
                        "example_id": example.response_id,
                        "token_position": "candidate_3",
                        "token_text": "jazz",
                        "explanation": "The activation focuses on a concrete music topic.",
                        "parse_status": "ok",
                        "token_status": "ok",
                        "layer": 20,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider = NlaFeedbackProvider(
                checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                layer=20,
                backend="precomputed",
                max_tokens_per_example=3,
                precomputed_path=str(path),
            )

        dspy_example = type("Example", (), {"example_id": example.response_id})()
        _validate_nla_feedback_ready(provider, [dspy_example], min_coverage=0.95)

    def test_nla_precomputed_validation_accepts_missing_tag_text(self) -> None:
        example = load_usr_examples_from_fixture()[0]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nla.jsonl"
            path.write_text(
                json_dump(
                    {
                        "example_id": example.response_id,
                        "token_position": "candidate_3",
                        "token_text": "jazz",
                        "explanation": "The activation focuses on a concrete music topic.",
                        "parse_status": "missing_tags",
                        "token_status": "ok",
                        "layer": 20,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider = NlaFeedbackProvider(
                checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                layer=20,
                backend="precomputed",
                max_tokens_per_example=3,
                precomputed_path=str(path),
            )

        dspy_example = type("Example", (), {"example_id": example.response_id})()
        _validate_nla_feedback_ready(provider, [dspy_example], min_coverage=0.95)

    def test_parse_explanation_accepts_closing_tag_only(self) -> None:
        explanation, status = parse_explanation("semantic content.\n</explanation>")
        self.assertEqual(explanation, "semantic content.")
        self.assertEqual(status, "partial_tags")

    def test_aux_judge_feedback_does_not_change_metric_score(self) -> None:
        class FakeAuxJudge:
            def feedback_for(self, **kwargs):
                self.kwargs = kwargs
                return "Auxiliary 35B judge feedback: reward topic-specific details."

        class FakeDspy:
            class Prediction:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

        example = load_usr_examples_from_fixture()[0]
        provider = FakeAuxJudge()
        with patch.dict("sys.modules", {"dspy": FakeDspy}):
            metric_fn = create_metric_fn(
                "Engaging",
                min_score=1,
                max_score=3,
                aux_judge_provider=provider,
            )
            result = metric_fn(
                type(
                    "Example",
                    (),
                    {
                        "human_score": example.human_score("Engaging"),
                        "context": example.context,
                        "fact": example.fact,
                        "response": example.response,
                        "context_id": example.context_id,
                        "response_id": example.response_id,
                    },
                )(),
                type("Pred", (), {"score": "2", "rationale": "It is acceptable."})(),
            )

        self.assertAlmostEqual(result.score, 1.0)
        self.assertIn("Auxiliary 35B judge feedback", result.feedback)
        self.assertEqual(provider.kwargs["parsed"], 2)

    def test_aux_judge_receives_nla_feedback_when_available(self) -> None:
        class FakeNlaProvider:
            def feedback_for(self, example):
                return "NLA multi-token verbalizations: candidate token='jazz': concrete music-topic signal."

        class FakeAuxJudge:
            def feedback_for(self, **kwargs):
                self.kwargs = kwargs
                return "Auxiliary 35B judge feedback: turn NLA into a rubric-level lesson."

        class FakeDspy:
            class Prediction:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

        example = load_usr_examples_from_fixture()[0]
        aux_provider = FakeAuxJudge()
        with patch.dict("sys.modules", {"dspy": FakeDspy}):
            metric_fn = create_metric_fn(
                "Engaging",
                min_score=1,
                max_score=3,
                nla_feedback_provider=FakeNlaProvider(),
                aux_judge_provider=aux_provider,
            )
            result = metric_fn(
                type(
                    "Example",
                    (),
                    {
                        "human_score": example.human_score("Engaging"),
                        "context": example.context,
                        "fact": example.fact,
                        "response": example.response,
                        "context_id": example.context_id,
                        "response_id": example.response_id,
                        "example_id": example.response_id,
                    },
                )(),
                type("Pred", (), {"score": "2", "rationale": "It is acceptable."})(),
            )

        self.assertIn("NLA multi-token verbalizations", result.feedback)
        self.assertIn("Auxiliary 35B judge feedback", result.feedback)
        self.assertIn("concrete music-topic signal", aux_provider.kwargs["extra_feedback"])

    def test_aux_judge_empty_response_is_error_and_keeps_response_json(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": "   ",
                                    "reasoning_content": "The endpoint returned reasoning but no final feedback.",
                                },
                            }
                        ]
                    }
                ).encode("utf-8")

        example = load_usr_examples_from_fixture()[0]
        provider = AuxJudgeFeedbackProvider(
            api_base="http://127.0.0.1:1/v1",
            model="local-llamacpp",
            api_key="EMPTY",
        )
        dspy_example = type(
            "Example",
            (),
            {
                "human_score": example.human_score("Engaging"),
                "context": example.context,
                "fact": example.fact,
                "response": example.response,
                "context_id": example.context_id,
                "response_id": example.response_id,
                "example_id": example.response_id,
            },
        )()

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            feedback = provider.feedback_for(
                example=dspy_example,
                pred=type("Pred", (), {"score": "2", "rationale": "It is acceptable."})(),
                parsed=2,
                target=example.human_score("Engaging"),
                agreement=1.0,
                dimension="Engaging",
            )

        self.assertIn("status=error", feedback)
        self.assertIn("message content is empty", feedback)
        with TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "aux.jsonl"
            self.assertEqual(provider.write_artifact(artifact), 1)
            row = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["raw_response"], "")
        self.assertIn("reasoning_content", row["response_json"])

    def test_aux_judge_success_validation_rejects_low_coverage(self) -> None:
        class FakeAuxJudgeProvider:
            def status_counts(self):
                return {"ok": 2, "error": 1}

        with TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "aux.jsonl"
            with self.assertRaises(ValueError) as ctx:
                _validate_aux_judge_feedback_success(
                    FakeAuxJudgeProvider(),
                    min_success_rate=0.95,
                    artifact_path=artifact,
                )

        self.assertIn("success rate is below threshold", str(ctx.exception))
        self.assertIn("'error': 1", str(ctx.exception))

    def test_configure_lms_keeps_judge_global_and_returns_separate_proposer(self) -> None:
        configured = {}

        class FakeLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeDspy:
            LM = FakeLM

            @staticmethod
            def configure(**kwargs):
                configured.update(kwargs)

        args = Namespace(
            judge_model="Qwen/Qwen2.5-7B-Instruct",
            api_base="http://127.0.0.1:8000/v1",
            max_tokens=512,
            temperature=0.0,
            proposer_model="local-llamacpp",
            proposer_api_base="http://127.0.0.1:8080/v1",
            proposer_api_key="local-llamacpp-key",
            proposer_api_key_env="LLAMA_API_KEY",
            proposer_max_tokens=4096,
            proposer_temperature=0.7,
        )

        with patch.dict("sys.modules", {"dspy": FakeDspy}):
            judge_lm, proposer_lm = configure_lms(args)

        self.assertIs(configured["lm"], judge_lm)
        self.assertIsNot(judge_lm, proposer_lm)
        self.assertEqual(judge_lm.kwargs["model"], "openai/Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(judge_lm.kwargs["api_base"], "http://127.0.0.1:8000/v1")
        self.assertEqual(proposer_lm.kwargs["model"], "openai/local-llamacpp")
        self.assertEqual(proposer_lm.kwargs["api_base"], "http://127.0.0.1:8080/v1")
        self.assertEqual(proposer_lm.kwargs["api_key"], "local-llamacpp-key")

    def test_exports_prompt_trajectory_from_gepa_viz_schema(self) -> None:
        payload = {
            "examples": [{"source_text": "x"}],
            "candidates": {
                "0": {"prompt": "Seed prompt", "parent": None, "score": 0.5, "predictions": [], "minibatch": None},
                "1": {
                    "prompt": "Better prompt",
                    "parent": "0",
                    "score": 0.75,
                    "predictions": [{"score": 1.0}],
                    "minibatch": [{"feedback": "Reward concrete details."}],
                },
            },
        }
        with TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "run.json"
            out_path = Path(tmpdir) / "trajectory.jsonl"
            run_path.write_text(json_dump(payload), encoding="utf-8")
            count = export_prompt_trajectory(run_path, out_path)
            rows = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]

        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 2)
        self.assertIn('"candidate_id": "1"', rows[1])
        self.assertIn("Reward concrete details", rows[1])
        self.assertIn("Better prompt", rows[1])

    def test_aggregate_results_computes_metric_improvements(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "metrics_20260101T000000Z.csv").write_text(
                "\n".join(
                    [
                        "program,total,parsed,coverage,agreement,n,pearson,spearman,kendall_tau,mae",
                        "baseline,10,10,1.0,0.5,10,0.2,0.3,0.1,0.6",
                        "optimized,10,10,1.0,0.75,10,0.4,0.5,0.4,0.3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "run_config_20260101T000000Z.json").write_text(
                json_dump(
                    {
                        "dataset": "summeval",
                        "dimension": "consistency",
                        "perplexity_feedback": True,
                        "nla_feedback": True,
                        "rows": {"gepa_train": 8, "gepa_validation": 4, "final_test": 10},
                    }
                ),
                encoding="utf-8",
            )

            rows = build_rows(root)

        optimized = [row for row in rows if row["program"] == "optimized"][0]
        self.assertEqual(optimized["dataset"], "summeval")
        self.assertAlmostEqual(optimized["agreement_improvement"], 0.25)
        self.assertAlmostEqual(optimized["kendall_tau_improvement"], 0.3)
        self.assertAlmostEqual(optimized["mae_improvement"], 0.3)

    def test_diagnose_nla_run_compares_control_and_nla_errors(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            control = root / "control"
            nla = root / "nla"
            control.mkdir()
            nla.mkdir()
            for run_dir, nla_enabled in ((control, False), (nla, True)):
                (run_dir / "run_config_20260101T000000Z.json").write_text(
                    json_dump(
                        {
                            "dataset": "topical_chat",
                            "dimension": "engagingness",
                            "seed": 42,
                            "train_groups": 40,
                            "val_groups": 10,
                            "test_groups": 10,
                            "judge_model": "Qwen/Qwen2.5-7B-Instruct",
                            "proposer_model": "local-llamacpp",
                            "instruction_proposer": "generalizing",
                            "perplexity_feedback": True,
                            "nla_feedback": nla_enabled,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "metrics_20260101T000000Z.csv").write_text(
                    "\n".join(
                        [
                            "program,total,parsed,coverage,agreement,n,pearson,spearman,kendall_tau,mae",
                            "baseline,1,1,1.0,0.5,1,0.0,0.0,0.0,1.0",
                            "optimized,1,1,1.0,0.5,1,0.0,0.0,0.0,1.0",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (control / "optimized_predictions_20260101T000000Z.jsonl").write_text(
                json_dump({"example_id": "ex1", "group_id": "g1", "target": 3, "prediction": 1, "parse_status": "ok"})
                + "\n",
                encoding="utf-8",
            )
            (nla / "optimized_predictions_20260101T000000Z.jsonl").write_text(
                json_dump({"example_id": "ex1", "group_id": "g1", "target": 3, "prediction": 2, "parse_status": "ok"})
                + "\n",
                encoding="utf-8",
            )
            (nla / "nla_verbalizations_20260101T000000Z.jsonl").write_text(
                json_dump(
                    {
                        "example_id": "ex1",
                        "token_position": "candidate_first",
                        "token_status": "ok",
                        "parse_status": "ok",
                        "verbalization": "This activation highlights a specific candidate detail.",
                        "activation_dim": 3584,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_nla_diagnostic_report(control, nla)

        self.assertEqual(report["prediction_summary"]["joined_examples"], 1)
        self.assertEqual(report["prediction_summary"]["nla_improved_examples"], 1)
        self.assertEqual(report["nla_quality"]["rows"], 1)
        self.assertEqual(report["nla_quality"]["covered_examples"], 1)
        self.assertEqual(report["nla_quality"]["token_category"], {"candidate": 1})
        self.assertEqual(report["nla_quality"]["activation_stats_rows"], 1)

    def test_export_nla_manifest_contains_prompt_and_ids(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tc.json"
            path.write_text(json_dump(FIXTURE), encoding="utf-8")
            args = Namespace(
                dataset="topical_chat",
                dimension="engagingness",
                data_source=str(path),
                split="validation",
                train_groups=1,
                val_groups=1,
                test_groups=0,
                seed=42,
                limit=None,
                token_policy="semantic_multi",
            )
            rows = build_manifest(args)

        self.assertTrue(rows)
        self.assertIn("prompt", rows[0])
        self.assertIn("example_id", rows[0])
        self.assertEqual(rows[0]["token_policy"], "semantic_multi")
        self.assertIn("Candidate output:", rows[0]["prompt"])

    def test_export_nla_manifest_gepa_split_contains_train_and_validation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tc.json"
            path.write_text(json_dump(FIXTURE), encoding="utf-8")
            args = Namespace(
                dataset="topical_chat",
                dimension="engagingness",
                data_source=str(path),
                split="gepa",
                train_groups=1,
                val_groups=1,
                test_groups=0,
                seed=42,
                limit=None,
                token_policy="semantic_multi",
            )
            rows = build_manifest(args)

        self.assertEqual(len(rows), 3)

    def test_nla_precompute_selects_multiple_semantic_tokens(self) -> None:
        row = {
            "source_text": "Alice discusses astronomy with concrete comet details.",
            "candidate_output": "The candidate asks about meteor showers and stars tonight.",
            "fact": "_nofact",
        }
        prompt = "\n".join([row["source_text"], row["candidate_output"]])
        targets = SemanticTokenSelector(max_tokens_per_example=3).select(row, prompt)

        self.assertGreaterEqual(len(targets), 2)
        self.assertGreaterEqual(sum(target.token_position.startswith("candidate_") for target in targets), 2)
        self.assertTrue(any(target.token_position.startswith("source_") for target in targets))
        self.assertFalse(any(target.token_position == "candidate_first" for target in targets))
        self.assertIn("meteor", [target.token_text for target in targets])

    def test_nla_precompute_dry_run_writes_runner_compatible_rows(self) -> None:
        manifest = [
            {
                "dataset": "topical_chat",
                "dimension": "engagingness",
                "example_id": "ex1",
                "group_id": "ctx1",
                "human_score": 2.0,
                "source_text": "Alice discusses astronomy with concrete comet details.",
                "candidate_output": "The candidate asks about meteor showers and stars.",
                "fact": "_nofact",
                "prompt": "Alice discusses astronomy with concrete comet details.\nThe candidate asks about meteor showers and stars.",
            }
        ]
        activations = fake_activation_rows(
            manifest,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            layer=20,
            max_tokens_per_example=2,
            d_model=8,
            limit=None,
        )
        rows = list(
            iter_verbalization_rows(
                activations,
                checkpoint="kitft/nla-qwen2.5-7b-L20-av",
                backend="transformers",
                nla_root=Path("natural_language_autoencoders"),
                dry_run=True,
                temperature=0.0,
                max_new_tokens=8,
                injection_scale=None,
                dtype_name="float16",
                device_map="auto",
                trust_remote_code=True,
            )
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["example_id"], "ex1")
        self.assertIn("verbalization", rows[0])
        self.assertEqual(rows[0]["parse_status"], "ok")
        self.assertEqual(rows[0]["activation_dim"], 8)
        self.assertGreater(rows[0]["activation_l2_norm"], 0.0)
        self.assertIn("activation_std", rows[0])

    def test_experimental_hybrid_strategy_deduplicates_context_rows(self) -> None:
        selector = ExperimentalTokenSelector("hybrid_context_dedup_6")
        first = {
            "example_id": "ctx1_response_00",
            "group_id": "ctx1",
            "source_text": "Alice discusses astronomy with detailed comet observations.",
            "candidate_output": "The candidate asks about meteor showers and stars tonight.",
            "fact": "A comet was visible from Earth for several nights.",
            "prompt": "",
        }
        second = {
            **first,
            "example_id": "ctx1_response_01",
            "candidate_output": "Another response mentions telescopes and night skies.",
        }
        first_prompt = "\n".join([first["source_text"], first["fact"], first["candidate_output"]])
        second_prompt = "\n".join([second["source_text"], second["fact"], second["candidate_output"]])

        first_targets = selector.select(first, first_prompt)
        second_targets = selector.select(second, second_prompt)

        self.assertTrue(any(target.shared_group_feedback for target in first_targets))
        self.assertFalse(any(target.shared_group_feedback for target in second_targets))
        self.assertTrue(any(target.token_position.startswith("candidate_") for target in second_targets))

    def test_experimental_position_strategies_select_expected_fields(self) -> None:
        row = {
            "example_id": "ctx1_response_00",
            "group_id": "ctx1",
            "source_text": "Alice discusses astronomy with detailed comet observations.",
            "candidate_output": "The candidate asks about meteor showers and stars tonight.",
            "fact": "A comet was visible from Earth for several nights.",
            "prompt": "",
        }
        prompt = "\n".join(
            [
                row["source_text"],
                row["fact"],
                row["candidate_output"],
                "Evaluation form:",
                "Rationale:",
                "Score:",
            ]
        )

        candidate_targets = ExperimentalTokenSelector("candidate_fml_3").select(row, prompt)
        self.assertEqual(
            [target.token_position for target in candidate_targets],
            ["candidate_first", "candidate_middle", "candidate_last"],
        )

        prompt_targets = ExperimentalTokenSelector("prompt_tail_6").select(row, prompt)
        self.assertTrue(prompt_targets)
        self.assertTrue(all(target.token_position.startswith("prompt_tail") for target in prompt_targets))

    def test_soft_prompt_module_imports_without_gpu_dependencies(self) -> None:
        import importlib.util

        module_path = Path(__file__).resolve().parents[1] / "soft_prompting" / "train_soft_judge.py"
        spec = importlib.util.spec_from_file_location("train_soft_judge", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader if spec else None)

    def test_evaluate_program_predictions_include_debug_text(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tc.json"
            path.write_text(json_dump(FIXTURE), encoding="utf-8")
            row = get_task("usr").load(path, "engagingness")[0]
            output = Path(tmpdir) / "predictions.jsonl"

            class Program:
                def __call__(self, **kwargs):
                    del kwargs
                    return type("Pred", (), {"score": "2", "rationale": "Acceptable response."})()

            evaluate_program(Program(), [row], output)
            payload = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(payload["source_text"], row.source_text)
        self.assertEqual(payload["fact"], row.fact)
        self.assertEqual(payload["candidate_output"], row.candidate_output)

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


def json_dump(value):
    import json

    return json.dumps(value)


if __name__ == "__main__":
    unittest.main()
