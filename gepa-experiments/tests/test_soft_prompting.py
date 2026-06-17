from __future__ import annotations

import unittest

import torch

from soft_prompting.train_soft_judge import (
    SoftJudgeExperiment,
    SoftPromptConfig,
    nearest_tokens,
    soft_prompt_config_metadata,
)


class DummyTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"tok_{token_ids[0]}"


class SoftPromptingTest(unittest.TestCase):
    def make_config(self, init: str) -> SoftPromptConfig:
        return SoftPromptConfig(
            model_name="model",
            dataset="topical_chat",
            dimension="engagingness",
            data_source="data.json",
            train_groups=1,
            val_groups=1,
            test_groups=1,
            seed=42,
            num_virtual_tokens=16,
            soft_prompt_init=init,
            soft_init_text="seed text",
            max_seq_len=128,
            max_new_tokens=4,
            train_batch_size=1,
            gradient_accumulation_steps=1,
            eval_batch_size=1,
            learning_rate=0.1,
            epochs=1.0,
            warmup_ratio=0.0,
            max_train_examples=None,
            max_eval_examples=None,
            load_in_4bit=True,
        )

    def test_config_metadata_marks_inactive_init_text(self) -> None:
        payload = soft_prompt_config_metadata(self.make_config("random"))

        self.assertFalse(payload["soft_init_text_active"])
        self.assertEqual(payload["soft_init_text"], "")

    def test_config_metadata_keeps_active_init_text(self) -> None:
        payload = soft_prompt_config_metadata(self.make_config("text"))

        self.assertTrue(payload["soft_init_text_active"])
        self.assertEqual(payload["soft_init_text"], "seed text")

    def test_experiment_keeps_training_methods(self) -> None:
        self.assertTrue(callable(getattr(SoftJudgeExperiment, "load_model")))
        self.assertTrue(callable(getattr(SoftJudgeExperiment, "make_peft_model")))
        self.assertTrue(callable(getattr(SoftJudgeExperiment, "write_soft_prompt_artifacts")))

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
