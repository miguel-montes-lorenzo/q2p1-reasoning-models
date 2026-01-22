# config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME: str = "gsm8k"
DATASET_CONFIG: str = "main"

SYSTEM_PROMPT: str = (
        "You are a helpful math tutor. Solve the problem step by step. "
        "Put your reasoning inside <think>...</think>. "
        "Then provide the final answer clearly and concisely."
    )

USE_4_BIT: bool = True
MAX_SEQ_LEN: int = 1024

MAX_NEW_TOKENS: int = 256
TEMPERATURE: float = 0.8
TOP_P: float = 0.95


@dataclass(frozen=True)
class SFT_CONFIG:
    """Configuration container for Supervised Fine-Tuning (SFT).

    This class centralizes all parameters required for SFT training,
    checkpointing, and evaluation/check scripts. It is intended to be
    instantiated once and passed to training and inference utilities
    to ensure consistent behavior across scripts.

    Attributes:
        model_name: Hugging Face model identifier of the base causal LM.
        dataset_name: Hugging Face dataset name used for SFT.
        dataset_config: Dataset configuration or subset name passed to
            `load_dataset(..., name=dataset_config)`.

        use_4bit: Whether to load the base model using 4-bit NF4 quantization.
        max_seq_len: Maximum total sequence length (prompt + completion).

        system_prompt: System prompt prepended when formatting inputs using
            a chat template.

        max_new_tokens: Maximum number of tokens generated during inference
            in check/evaluation scripts.
        temperature: Sampling temperature for generation. If None, generation
            defaults to deterministic decoding.
        top_p: Nucleus sampling probability mass. Only used when sampling
            is enabled.

        epochs: Number of training epochs for SFT.
        lr: Learning rate used during SFT optimization.
        batch_size_questions: Per-device batch size measured in questions
            (training examples).

        loogging_interval: Number of steps between metric logging.
        checkpoint_directory: Directory where LoRA adapters and tokenizer
            checkpoints are saved.
        checkpoint_interval: Number of steps between checkpoint saves.
        keep_last_checkpoints: Maximum number of recent checkpoints to retain.
    """

    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    dataset_config: str = DATASET_CONFIG

    use_4bit: bool = USE_4_BIT
    max_seq_len: int = MAX_SEQ_LEN

    system_prompt: str = SYSTEM_PROMPT

    # Generation hyperparameters (check script)
    do_sample: bool = False
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float | None = TEMPERATURE if do_sample else None
    top_p: float | None = TOP_P if do_sample else None

    # Training hyperparameters (SFT)
    epochs: int = 1
    lr: float = 2e-4
    batch_size_questions: int = 4

    # Training Management
    loogging_interval: int = 10
    checkpoint_directory: Path = Path("./weights/sft_lora")
    checkpoint_interval: int = 200
    keep_last_checkpoints: int = 2


class GRPO_CONFIG:
    """Configuration container for GRPO-based policy optimization.

    This class groups all hyperparameters and runtime settings required
    for Group Relative Policy Optimization (GRPO) training on GSM8K-style
    tasks. It mirrors `SFT_CONFIG` where applicable, but includes
    additional parameters specific to group-based policy-gradient
    training.

    Attributes:
        model_name: Hugging Face model identifier of the base causal LM.
        dataset_name: Hugging Face dataset name used for GRPO training.
        dataset_config: Dataset configuration or subset name passed to
            `load_dataset(..., name=dataset_config)`.

        use_4bit: Whether to load the base model using 4-bit NF4 quantization.
        max_seq_len: Maximum total sequence length (prompt + generation).

        system_prompt: System prompt prepended when formatting inputs using
            a chat template.

        max_new_tokens: Maximum number of tokens generated per sampled
            completion during training.
        temperature: Sampling temperature used during stochastic generation.
            Must be greater than zero for GRPO to be effective.
        top_p: Nucleus sampling probability mass used during generation.

        epochs: Number of training epochs for GRPO.
        lr: Learning rate used for policy updates.
        batch_size_questions: Number of questions processed per optimization
            step.
        group_size: Number of sampled completions per question used to
            compute group-relative advantages.
        max_train_examples: Optional cap on the number of training examples
            for faster or debug runs.
        grad_accum_steps: Number of gradient accumulation steps.
        clip_grad_norm: Maximum gradient norm used for clipping.

        loogging_interval: Number of steps between metric logging.
        checkpoint_directory: Directory where LoRA adapter checkpoints are
            saved.
        checkpoint_interval: Number of steps between checkpoint saves.
        keep_last_checkpoints: Maximum number of recent checkpoints to retain.
    """

    # model_name: str = MODEL_NAME
    # dataset_name: str = DATASET_NAME
    # dataset_config: str = DATASET_CONFIG

    # use_4bit: bool = USE_4_BIT
    # max_seq_len: int = MAX_SEQ_LEN

    # system_prompt: str = SYSTEM_PROMPT

    # # Generation hyperparameters (check script)
    # do_sample: bool = True
    # max_new_tokens: int = MAX_NEW_TOKENS
    # temperature: float | None = TEMPERATURE if do_sample else None
    # top_p: float | None = TOP_P if do_sample else None

    # # Training hyperparameters (GRPO)
    # epochs: int = 1
    # lr: float = 2e-4
    # batch_size_questions: int = 1  # 4
    # group_size: int = 4
    # max_train_examples: int | None = 2_000
    # grad_accum_steps: int = 1
    # clip_grad_norm: float = 1.0

    # # Training Management
    # loogging_interval: int = 25
    # checkpoint_directory: Path = Path("./weights/final_rlm_lora")
    # checkpoint_interval: int = 200
    # keep_last_checkpoints: int = 2



    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    dataset_config: str = DATASET_CONFIG

    use_4bit: bool = USE_4_BIT
    max_seq_len: int = MAX_SEQ_LEN

    system_prompt: str = SYSTEM_PROMPT

    do_sample: bool = True
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE
    top_p: float | None = TOP_P

    # GRPO / RL hyperparameters
    epochs: int = 1
    lr: float = 1e-5
    batch_size_questions: int = 1
    group_size: int = 8
    max_train_examples: int | None = 2_000
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    beta_kl: float = 0.02

    # LoRA hyperparameters (optional override from training script)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    logging_interval: int = 25
    checkpoint_directory: Path = Path("./weights/final_rlm_lora")
    checkpoint_interval: int = 200
    keep_last_checkpoints: int = 2