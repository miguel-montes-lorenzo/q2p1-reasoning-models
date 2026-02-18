# tool_use/train_sft.py

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from tool_use.custom.config import REPO_DIR
from tool_use.custom.config import SFT_CONFIG as CONFIG
from tool_use.custom.tool_handler import (
    ensure_response_contains_answer,
    insert_tool_desciptions_in_system_propt,
    parse_and_execute_tool_call,
)
from tool_use.custom.tools import TOOL_DICT
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from utils.paths import check_cwd


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
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _is_safe_calculator_expression(*, expression: str) -> bool:
    """Check whether an expression is safe for the calculator tool.

    Args:
        expression: Candidate expression string.

    Returns:
        True if expression uses only allowed characters, else False.
    """
    allowed: set[str] = set("0123456789+-*/(). ")
    return set(expression).issubset(allowed) and len(expression.strip()) > 0


def formatting_prompts_func(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    cfg: CONFIG,
) -> str:
    """Format GSM8K examples into a tool-use chat-template supervised sample.

    This creates a 2-step interaction that demonstrates tool usage:
      system -> user -> assistant(tool call) -> user(tool evaluation) -> assistant(answer)

    The tool evaluation step is produced by:
      parse_and_execute_tool_call(...), which returns exactly what must be appended
      to the previous prompt to create the next prompt (<think>...</think> and
      <tools>...</tools>).

    Finally, ensure_response_contains_answer(...) enforces that the last assistant
    response contains an <answer>...</answer> block.

    IMPORTANT:
      This script now uses the NEW tool-call format:
        @calculator(expression="..."):ID
      (and the handler is the single source of truth for validation/execution).

    Args:
        example: Dataset example containing at least "question" and "answer".
        tokenizer: HF tokenizer implementing apply_chat_template().
        cfg: Shared SFT configuration (includes system prompt and max_calls).

    Returns:
        A single training string.
    """
    question: str = str(example["question"])
    answer_field: str = str(example["answer"])
    reasoning, final_answer = _parse_gsm8k_answer(answer_field=answer_field)

    wrapped_question: str = f"<question>{question}</question>"

    # Step 1: assistant proposes a tool call (typically no <answer>, so the handler continues).
    #
    # We pick a deterministic id for the *training* example. This is fine in SFT,
    # because ids are local to the sample transcript. The global-uniqueness constraint
    # is enforced at runtime across iterations, not across dataset rows.
    if _is_safe_calculator_expression(expression=final_answer):
        first_assistant: str = (
            "<think>"
            f"{reasoning}\n"
            "I will compute the final numeric value using the calculator tool. "
            f'The result is @calculator(expression="{final_answer}"):1.'
            "</think>"
        )
    else:
        first_assistant = f"<think>{reasoning}</think>"

    (
        should_continue,
        prompt_appendix,
        _raw_full,
        _formatted_full,
        _parsed_answer_only,
    ) = parse_and_execute_tool_call(
        model_output=first_assistant,
        tool_dict=TOOL_DICT,
        max_calls=int(cfg.max_calls),
        used_tool_ids=set(),
        global_outputs={},
    )

    # Step 2: assistant produces the final answer (must contain <answer>).
    #
    # If we used the calculator, it's nicer for the model to learn referencing:
    #   <answer>@1</answer>
    # Otherwise it can just output the final answer text.
    if _is_safe_calculator_expression(expression=final_answer):
        second_assistant: str = f"<think>{reasoning}</think>\n<answer>@1</answer>"
    else:
        second_assistant = (
            f"<think>{reasoning}</think>\n<answer>{final_answer}</answer>"
        )
    second_assistant = ensure_response_contains_answer(full_prompt=second_assistant)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": wrapped_question},
    ]

    if should_continue:
        messages.extend([
            {"role": "assistant", "content": first_assistant},
            {"role": "user", "content": prompt_appendix},
            {"role": "assistant", "content": second_assistant},
        ])
    else:
        # If no tools were executed / handler did not ask to continue, train a simple 1-turn.
        messages.append({"role": "assistant", "content": second_assistant})

    text: str = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def train() -> None:
    """Run LoRA SFT on GSM8K with tool-use prompt + calculator description."""
    cfg_base: CONFIG = CONFIG()

    descriptions: dict[str, str] = {
        tool_name: str(tool_meta["description"])
        for tool_name, tool_meta in TOOL_DICT.items()
    }
    tool_augmented_system_prompt: str = insert_tool_desciptions_in_system_propt(
        descriptions=descriptions
    )
    cfg: CONFIG = replace(cfg_base, system_prompt=tool_augmented_system_prompt)

    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

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

    raw: Any = load_dataset(path=cfg.dataset_name, name=cfg.dataset_config)
    dataset: Dataset = raw["train"]

    sft_args: SFTConfig = SFTConfig(
        output_dir=str(cfg.checkpoint_directory),
        num_train_epochs=int(cfg.epochs),
        per_device_train_batch_size=int(cfg.batch_size_questions),
        gradient_accumulation_steps=2,
        learning_rate=float(cfg.lr),
        fp16=False,
        bf16=True,
        logging_steps=cfg.loogging_interval,
        save_steps=cfg.checkpoint_interval,
        save_total_limit=cfg.keep_last_checkpoints,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
        max_length=int(cfg.max_seq_len),
        packing=False,
    )

    formatting_func: Any = partial(
        formatting_prompts_func,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_args,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(output_dir=str(cfg.checkpoint_directory))
    tokenizer.save_pretrained(save_directory=str(cfg.checkpoint_directory))


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    train()
