# train_sft.py (your training script)

from __future__ import annotations

import os
from functools import partial
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import CONFIG


def _parse_gsm8k_answer(*, answer_field: str) -> tuple[str, str]:
    """Parse GSM8K 'answer' field into (reasoning, final_answer).

    GSM8K commonly stores the rationale plus a final answer after a '####' marker.

    Args:
        answer_field: Raw GSM8K answer string.

    Returns:
        A tuple (reasoning, final_answer). If parsing fails, final_answer is "".
    """
    if "####" in answer_field:
        parts: list[str] = answer_field.split("####", maxsplit=1)
        reasoning: str = parts[0].strip()
        final_answer: str = parts[1].strip()
        return (reasoning, final_answer)

    reasoning_fallback: str = answer_field.strip()
    final_fallback: str = ""
    return (reasoning_fallback, final_fallback)


def formatting_prompts_func(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    cfg: CONFIG,
) -> str:
    """Format GSM8K examples into a Qwen2.5 chat-template supervised sample.

    This emits a ChatML-style conversation with:
      system -> user -> assistant

    The assistant message includes:
      <think>reasoning</think>
      final_answer

    Args:
        example: Dataset example containing at least "question" and "answer".
        tokenizer: HF tokenizer implementing apply_chat_template().
        cfg: Shared SFT configuration (includes system prompt).

    Returns:
        A single training string.
    """
    question: str = str(example["question"])
    answer_field: str = str(example["answer"])
    reasoning, final_answer = _parse_gsm8k_answer(answer_field=answer_field)

    assistant_text: str = f"<think>{reasoning}</think>\n{final_answer}".strip()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_text},
    ]

    text: str = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    """Create a BitsAndBytes quantization config.

    Args:
        use_4bit: Whether to enable 4-bit NF4 quantization.

    Returns:
        A BitsAndBytesConfig if enabled, else None.
    """
    if not use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        # Minimal fix: align compute dtype with BF16 training to avoid FP16 GradScaler.
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def train() -> None:
    """Run LoRA SFT on GSM8K with a Qwen2.5 Instruct model.

    Minimal fix:
    - Train in BF16 (fp16=False, bf16=True) so Accelerate does not use GradScaler.
    - Align model dtype and 4-bit compute dtype to BF16.
    - This avoids the crash: "_amp_foreach_non_finite_check_and_unscale_cuda"
      not implemented for 'BFloat16' (triggered by GradScaler unscale on BF16 grads).
    """
    cfg: CONFIG = CONFIG()

    # Ensure Accelerate does not implicitly choose FP16 (which enables GradScaler).
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    # 1. Load tokenizer
    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build quantization config
    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    # 3. Load model (important: use `dtype` per your Transformers version)
    model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # 4. Configure LoRA
    peft_config: LoraConfig = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    # 5. Load dataset
    raw: Any = load_dataset(path=cfg.dataset_name, name=cfg.dataset_config)
    dataset: Dataset = raw["train"]

    # 6. Training configuration
    sft_args: SFTConfig = SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=int(cfg.epochs),
        per_device_train_batch_size=int(cfg.batch_size_questions),
        gradient_accumulation_steps=2,
        learning_rate=float(cfg.lr),
        # Minimal fix: disable fp16 to avoid GradScaler, enable bf16 instead.
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
        max_length=int(cfg.max_seq_len),
        packing=False,
    )

    formatting_func: Any = partial(formatting_prompts_func, tokenizer=tokenizer, cfg=cfg)

    # 7. Initialize trainer
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_args,
        formatting_func=formatting_func,
    )

    # 8. Train and save artifacts
    trainer.train()
    trainer.save_model(output_dir=str(cfg.output_dir))
    tokenizer.save_pretrained(save_directory=str(cfg.output_dir))


if __name__ == "__main__":
    train()
