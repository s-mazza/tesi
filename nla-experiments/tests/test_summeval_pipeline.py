from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nla-experiments"))

from summeval.common import extract_explanation
from summeval.extract_qwen_activations import build_fake_records
from summeval.prepare_summeval import build_manifest


def sample_rows() -> list[dict]:
    base = {
        "source": "Article text about a football match and a late substitution.",
        "reference": "A player was substituted late in a football match.",
        "system_output": "The summary says a player entered the match late.",
    }
    rows = []
    for idx, score in enumerate([1.0, 1.5, 4.5, 5.0]):
        rows.append(
            {
                **base,
                "doc_id": f"doc-{idx}",
                "system_id": f"M{idx}",
                "scores": {
                    "coherence": score,
                    "consistency": score,
                    "fluency": min(score, 3.0),
                    "relevance": score,
                    "overall": score,
                },
            }
        )
    return rows


class SummEvalPipelineTest(unittest.TestCase):
    def test_manifest_is_stratified_and_dimension_expanded(self) -> None:
        manifest = build_manifest(sample_rows(), max_examples=2, source_sha256="abc")
        self.assertEqual(len(manifest), 8)
        self.assertEqual({row["dimension"] for row in manifest}, {"coherence", "consistency", "fluency", "relevance"})
        self.assertEqual({row["sample_bucket"] for row in manifest}, {"low_consistency", "high_consistency"})
        for row in manifest:
            self.assertIn("prompt_hash", row)
            self.assertIn("Source Text", row["prompt"]) if row["dimension"] != "fluency" else self.assertIn("Summary", row["prompt"])

    def test_fake_activation_shape(self) -> None:
        manifest = build_manifest(sample_rows(), max_examples=2, source_sha256="abc")
        records = build_fake_records(
            manifest,
            token_positions=["prompt_final", "generated_score"],
            d_model=16,
            limit=1,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[0]["activation_vector"]), 16)
        self.assertEqual(records[0]["token_status"], "fake")

    def test_explanation_parser(self) -> None:
        explanation, status = extract_explanation("<explanation>fact tracking</explanation>")
        self.assertEqual(explanation, "fact tracking")
        self.assertEqual(status, "ok")
        explanation, status = extract_explanation("plain text")
        self.assertEqual(explanation, "plain text")
        self.assertEqual(status, "missing_tags")
        explanation, status = extract_explanation("<explanation>partial decode")
        self.assertEqual(explanation, "partial decode")
        self.assertEqual(status, "partial_tags")


if __name__ == "__main__":
    unittest.main()
