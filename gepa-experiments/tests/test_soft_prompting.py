from __future__ import annotations

import unittest

import torch

from soft_prompting.train_soft_judge import nearest_tokens


class DummyTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"tok_{token_ids[0]}"


class SoftPromptingTest(unittest.TestCase):
    def test_nearest_tokens_rank_by_l2_and_include_cosine(self) -> None:
        prompt = torch.tensor([[1.0, 0.1]])
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [10.0, 1.0],
                [-1.0, 0.0],
            ]
        )

        rows = nearest_tokens(prompt, embeddings, DummyTokenizer(), top_k=2)

        self.assertEqual(rows[0]["ranking_metric"], "l2")
        self.assertEqual(rows[0]["top_tokens"][0]["token_id"], 0)
        self.assertEqual(rows[0]["top_tokens"][1]["token_id"], 2)
        self.assertIn("l2", rows[0]["top_tokens"][0])
        self.assertIn("cosine", rows[0]["top_tokens"][0])
        self.assertIsInstance(rows[0]["top_tokens"][0]["cosine"], float)


if __name__ == "__main__":
    unittest.main()
