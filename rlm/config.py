# config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CONFIG:
    """Shared configuration for SFT training and checks.

    Args:
        model_name: HF id of the base model.
        dataset_name: HF dataset name.
        dataset_config: Dataset config/subset name passed to load_dataset(..., name=...).
        output_dir: Directory where the LoRA adapter and tokenizer are saved.
        use_4bit: Whether to load the base model in 4-bit quantized mode.
        max_seq_len: Maximum sequence length for SFT.
        system_prompt: System prompt used for chat-template formatting.
        max_new_tokens: Maximum new tokens to generate in check script.

        epochs: Number of training epochs for SFT.
        lr: Learning rate for SFT training.
        batch_size_questions: Per-device training batch size (questions/examples).
        grpo_group_size: Group size for GRPO-style training (unused in these scripts).
        temperature: Sampling temperature for generation (unused when do_sample=False).
        top_p: Nucleus sampling probability mass for generation (unused when do_sample=False).
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    output_dir: Path = Path("./weights/sft_lora")

    use_4bit: bool = True
    max_seq_len: int = 1024

    system_prompt: str = (
        "You are a helpful math tutor. Solve the problem step by step. "
        "Put your reasoning inside <think>...</think>. "
        "Then provide the final answer clearly and concisely."
    )

    # Generation hyperparameters (check script)
    max_new_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None

    # Training hyperparameters (SFT)
    epochs: int = 1
    lr: float = 2e-4
    batch_size_questions: int = 4

    # Present for completeness (not used by these scripts)
    grpo_group_size: int | None = None
