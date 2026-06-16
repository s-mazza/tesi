#!/usr/bin/env python3
"""Train a soft-prompt judge on the same G-EVAL task used by GEPA.

The backbone model stays frozen. The only trainable parameters are PEFT prompt
tuning embeddings. Outputs are saved so SIPIT/random-prefix style experiments
can later try to verbalize or recover the learned continuous prefix.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from geval_gepa.metrics import compute_regression_metrics, normalized_absolute_score, parse_discrete_score
from geval_gepa.prompts import seed_instructions
from geval_gepa.tasks import EvalExample, get_task, split_examples


@dataclass(frozen=True)
class SoftPromptConfig:
    model_name: str
    dataset: str
    dimension: str
    data_source: str
    train_groups: int
    val_groups: int
    test_groups: int
    seed: int
    num_virtual_tokens: int
    soft_init_text: str
    max_seq_len: int
    max_new_tokens: int
    train_batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int
    learning_rate: float
    epochs: float
    warmup_ratio: float
    max_train_examples: int | None
    max_eval_examples: int | None
    load_in_4bit: bool


class TokenizedTrainDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class SoftPromptTrainer(Trainer):
    """Compute loss only on target score tokens to avoid full-prompt loss noise."""

    def __init__(self, *args: Any, virtual_tokens: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.virtual_tokens = virtual_tokens

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        prefix = torch.full(
            (labels.size(0), self.virtual_tokens),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        labels = torch.cat([prefix, labels], dim=1)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        loss = F.cross_entropy(shift_logits[mask].float(), shift_labels[mask])
        return (loss, outputs) if return_outputs else loss


class SoftJudgeExperiment:
    def __init__(self, config: SoftPromptConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("Soft-prompt training requires a CUDA GPU.")

    def run(self) -> None:
        seed_everything(self.config.seed)
        train_rows, val_rows, test_rows = self.load_rows()
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        min_score = train_rows[0].min_score
        max_score = train_rows[0].max_score
        model = self.load_model()
        peft_model = self.make_peft_model(model)
        train_items = tokenize_train_rows(
            train_rows,
            tokenizer=tokenizer,
            config=self.config,
            min_score=min_score,
            max_score=max_score,
        )
        if not train_items:
            raise RuntimeError("No train examples survived max_seq_len filtering.")
        trainer = SoftPromptTrainer(
            model=peft_model,
            virtual_tokens=self.config.num_virtual_tokens,
            args=TrainingArguments(
                output_dir=str(self.output_dir / "trainer"),
                per_device_train_batch_size=self.config.train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.epochs,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type="cosine",
                max_grad_norm=1.0,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False,
            ),
            train_dataset=TokenizedTrainDataset(train_items),
            data_collator=make_collator(tokenizer.pad_token_id),
        )
        trainer.train()
        peft_model.eval()
        adapter_dir = self.output_dir / "adapter"
        peft_model.save_pretrained(adapter_dir)
        self.write_soft_prompt_artifacts(peft_model, tokenizer)
        metrics = {
            "config": asdict(self.config),
            "splits": {
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "test_rows": len(test_rows),
                "tokenized_train_rows": len(train_items),
            },
            "validation_baseline": evaluate_rows(
                peft_model,
                tokenizer,
                val_rows,
                config=self.config,
                use_soft_prompt=False,
                output_path=self.output_dir / "predictions_validation_baseline.jsonl",
            ),
            "validation_soft_prompt": evaluate_rows(
                peft_model,
                tokenizer,
                val_rows,
                config=self.config,
                use_soft_prompt=True,
                output_path=self.output_dir / "predictions_validation_soft_prompt.jsonl",
            ),
            "test_baseline": evaluate_rows(
                peft_model,
                tokenizer,
                test_rows,
                config=self.config,
                use_soft_prompt=False,
                output_path=self.output_dir / "predictions_test_baseline.jsonl",
            ),
            "test_soft_prompt": evaluate_rows(
                peft_model,
                tokenizer,
                test_rows,
                config=self.config,
                use_soft_prompt=True,
                output_path=self.output_dir / "predictions_test_soft_prompt.jsonl",
            ),
        }
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def load_rows(self) -> tuple[list[EvalExample], list[EvalExample], list[EvalExample]]:
        task = get_task(self.config.dataset)
        rows = task.load(self.config.data_source or None, self.config.dimension or task.default_dimension)
        train, val, test = split_examples(
            rows,
            train_groups=self.config.train_groups,
            val_groups=self.config.val_groups,
            test_groups=self.config.test_groups,
            seed=self.config.seed,
        )
        if self.config.max_train_examples is not None:
            train = train[: self.config.max_train_examples]
        if self.config.max_eval_examples is not None:
            val = val[: self.config.max_eval_examples]
            test = test[: self.config.max_eval_examples]
        return train, val, test

    def load_model(self) -> Any:
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16,
            "device_map": {"": 0},
            "trust_remote_code": True,
        }
        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover - depends on runtime image.
                raise RuntimeError("load_in_4bit requires bitsandbytes-compatible transformers install.") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        return model

    def make_peft_model(self, model: Any) -> Any:
        try:
            from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
        except Exception as exc:  # pragma: no cover - depends on runtime image.
            raise RuntimeError("Soft-prompt training requires peft. Use the soft-prompt image.") from exc
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=self.config.soft_init_text,
            num_virtual_tokens=self.config.num_virtual_tokens,
            tokenizer_name_or_path=self.config.model_name,
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def write_soft_prompt_artifacts(self, peft_model: Any, tokenizer: Any) -> None:
        with torch.no_grad():
            prompt = peft_model.get_prompt(batch_size=1).detach().float().cpu()[0]
            embeddings = peft_model.get_base_model().get_input_embeddings().weight.detach().float().cpu()
            nearest = nearest_tokens(prompt, embeddings, tokenizer, top_k=10)
        torch.save({"soft_prompt_embeddings": prompt}, self.output_dir / "soft_prompt_embeddings.pt")
        write_jsonl(self.output_dir / "nearest_tokens.jsonl", nearest)
        manifest = {
            "model_name": self.config.model_name,
            "num_virtual_tokens": self.config.num_virtual_tokens,
            "hidden_size": int(prompt.shape[-1]),
            "embedding_artifact": str(self.output_dir / "soft_prompt_embeddings.pt"),
            "nearest_tokens_artifact": str(self.output_dir / "nearest_tokens.jsonl"),
            "adapter_dir": str(self.output_dir / "adapter"),
            "sipit_use": "Use soft_prompt_embeddings as a continuous prefix for SIPIT/random-prefix inversion experiments.",
        }
        (self.output_dir / "sipit_soft_prompt_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(row: EvalExample) -> str:
    instructions = seed_instructions(
        dataset=row.dataset,
        dimension=row.dimension,
        min_score=row.min_score,
        max_score=row.max_score,
    )
    return "\n\n".join(
        [
            instructions,
            "Source/context:",
            row.source_text,
            "Reference/fact:",
            row.fact or row.reference or "_nofact",
            "Candidate output:",
            row.candidate_output,
            "Return only the final score line in this format:",
            f"Score: <{row.min_score} to {row.max_score}>",
        ]
    )


def rounded_gold(row: EvalExample) -> int:
    return min(max(int(round(row.human_score)), row.min_score), row.max_score)


def encode_chat(tokenizer: Any, prompt: str, target: str | None) -> Any:
    user_msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(user_msg, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if target is None:
        return prompt_ids
    full_msg = user_msg + [{"role": "assistant", "content": target}]
    full_text = tokenizer.apply_chat_template(full_msg, add_generation_prompt=False, tokenize=False)
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids, full_ids[len(prompt_ids) :]
    return prompt_ids, tokenizer(target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]


def tokenize_train_rows(
    rows: list[EvalExample],
    *,
    tokenizer: Any,
    config: SoftPromptConfig,
    min_score: int,
    max_score: int,
) -> list[dict[str, Any]]:
    del min_score, max_score
    output = []
    skipped = 0
    for row in rows:
        prompt_ids, target_ids = encode_chat(tokenizer, build_prompt(row), f"Score: {rounded_gold(row)}")
        if config.num_virtual_tokens + len(prompt_ids) + len(target_ids) > config.max_seq_len:
            skipped += 1
            continue
        output.append({"prompt_ids": prompt_ids, "target_ids": target_ids})
    print(f"Tokenized {len(output)} train rows; skipped {skipped} over max_seq_len.", flush=True)
    return output


def make_collator(pad_id: int) -> Any:
    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        seqs = [feature["prompt_ids"] + feature["target_ids"] for feature in features]
        prompt_lengths = [len(feature["prompt_ids"]) for feature in features]
        target_lengths = [len(feature["target_ids"]) for feature in features]
        batch_size = len(features)
        max_len = max(len(seq) for seq in seqs)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for index, seq in enumerate(seqs):
            input_ids[index, : len(seq)] = torch.tensor(seq)
            attention_mask[index, : len(seq)] = 1
            start = prompt_lengths[index]
            end = start + target_lengths[index]
            labels[index, start:end] = torch.tensor(features[index]["target_ids"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


@torch.no_grad()
def generate_batch(peft_model: Any, tokenizer: Any, prompt_ids: list[list[int]], *, use_soft_prompt: bool, max_new_tokens: int) -> list[str]:
    peft_model.config.use_cache = True
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(item) for item in prompt_ids)
    device = next(peft_model.parameters()).device
    input_ids = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long, device=device)
    for index, item in enumerate(prompt_ids):
        input_ids[index, max_len - len(item) :] = torch.tensor(item, device=device)
        attention_mask[index, max_len - len(item) :] = 1
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if use_soft_prompt:
        generated = peft_model.generate(**kwargs)
    else:
        generated = peft_model.get_base_model().generate(**kwargs)
    new_tokens = generated[:, input_ids.shape[1] :]
    peft_model.config.use_cache = False
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def evaluate_rows(
    peft_model: Any,
    tokenizer: Any,
    rows: list[EvalExample],
    *,
    config: SoftPromptConfig,
    use_soft_prompt: bool,
    output_path: Path,
) -> dict[str, Any]:
    items = [(row, encode_chat(tokenizer, build_prompt(row), None)) for row in rows]
    predictions = []
    targets = []
    parsed = 0
    output_rows = []
    for start in range(0, len(items), config.eval_batch_size):
        batch = items[start : start + config.eval_batch_size]
        texts = generate_batch(
            peft_model,
            tokenizer,
            [item[1] for item in batch],
            use_soft_prompt=use_soft_prompt,
            max_new_tokens=config.max_new_tokens,
        )
        for (row, _), text in zip(batch, texts):
            score = parse_discrete_score(text, min_score=row.min_score, max_score=row.max_score)
            target = row.human_score
            if score is not None:
                parsed += 1
                predictions.append(float(score))
                targets.append(float(target))
            output_rows.append(
                {
                    "example_id": row.example_id,
                    "group_id": row.group_id,
                    "human_score": target,
                    "rounded_gold": rounded_gold(row),
                    "predicted_score": score,
                    "raw_generation": text,
                }
            )
    write_jsonl(output_path, output_rows)
    if predictions:
        metrics = compute_regression_metrics(predictions, targets).as_dict()
        metrics["agreement"] = sum(
            normalized_absolute_score(pred, target, min_score=rows[0].min_score, max_score=rows[0].max_score)
            for pred, target in zip(predictions, targets)
        ) / len(predictions)
    else:
        metrics = {"n": 0, "pearson": 0.0, "spearman": 0.0, "kendall_tau": 0.0, "mae": 0.0, "agreement": 0.0}
    metrics["parse_coverage"] = parsed / len(rows) if rows else 0.0
    return metrics


def nearest_tokens(prompt: torch.Tensor, embeddings: torch.Tensor, tokenizer: Any, *, top_k: int) -> list[dict[str, Any]]:
    prompt_norm = F.normalize(prompt, dim=1)
    embedding_norm = F.normalize(embeddings, dim=1)
    scores = prompt_norm @ embedding_norm.T
    values, indexes = scores.topk(k=top_k, dim=1)
    rows = []
    for soft_index in range(prompt.shape[0]):
        rows.append(
            {
                "soft_token_index": soft_index,
                "top_tokens": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token_text": tokenizer.decode([int(token_id)]),
                        "cosine": float(values[soft_index, rank]),
                    }
                    for rank, token_id in enumerate(indexes[soft_index].tolist())
                ],
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="topical_chat")
    parser.add_argument("--dimension", default="engagingness")
    parser.add_argument("--data-source", default="gepa-experiments/cache/tc_usr_data.json")
    parser.add_argument("--train-groups", type=int, default=40)
    parser.add_argument("--val-groups", type=int, default=10)
    parser.add_argument("--test-groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--soft-init-text", default="You are a careful, impartial evaluator. Rate the candidate output according to the rubric.")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SoftPromptConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        dimension=args.dimension,
        data_source=args.data_source,
        train_groups=args.train_groups,
        val_groups=args.val_groups,
        test_groups=args.test_groups,
        seed=args.seed,
        num_virtual_tokens=args.num_virtual_tokens,
        soft_init_text=args.soft_init_text,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        load_in_4bit=not args.no_4bit,
    )
    experiment = SoftJudgeExperiment(config, args.output_dir)
    experiment.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
