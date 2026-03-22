# tool_use/langchain/train_tool_sft.py

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from utils.paths import check_cwd

from tool_use.langchain.config import MAX_THINK_CALLS, REPO_DIR
from tool_use.langchain.config import SFT_CONFIG as CONFIG
from tool_use.langchain.tool_handler import insert_tool_desciptions_in_system_propt
from tool_use.langchain.tools import TOOL_DICT, calculator


# ---------------------------------------------------------------------------
# Gold transcript generation (no model needed)
# ---------------------------------------------------------------------------

_CALC_RE: re.Pattern[str] = re.compile(r"<<([^=]+)=([^>]+)>>")


def _render_tools_block(tool_id: str, expression: str, result: str) -> str:
    """Render a single-tool <tools> block matching tool_handler format."""
    escaped_expr: str = expression.replace("'", "\\'")
    escaped_result: str = result.replace("'", "\\'")
    return (
        "<tools>\n"
        "{\n"
        f"    {tool_id}: {{\n"
        "        tool: 'calculator',\n"
        "        successful_execution: True,\n"
        "        args: {\n"
        f"            'expression': '{escaped_expr}',\n"
        "        },\n"
        f"        output: '{escaped_result}'\n"
        "    },\n"
        "}\n"
        "</tools>"
    )


def _build_gold_transcript(
    answer_field: str,
    *,
    max_iterations: int = MAX_THINK_CALLS,
) -> str | None:
    """Build a gold tool-use transcript from a GSM8K answer field.

    Parses <<expr=result>> markers and converts each calculation into a
    proper tool-call cycle:
      <think>...reasoning... @calculator(expression="...")->ID.</think>
      <tools>...</tools>

    The final answer uses @ID to reference the last calculator result.

    Args:
        answer_field: Raw GSM8K answer string (with #### marker).
        max_iterations: Maximum think iterations allowed.

    Returns:
        Complete assistant transcript, or None if unparseable.
    """
    # Split answer from reasoning
    if "####" not in answer_field:
        return None
    reasoning_part: str = answer_field.split("####")[0].strip()
    final_answer: str = answer_field.split("####")[-1].strip()

    # Find all <<expr=result>> markers
    calcs: list[tuple[str, str, int, int]] = []
    for m in _CALC_RE.finditer(reasoning_part):
        expr: str = m.group(1).strip()
        result: str = m.group(2).strip()
        calcs.append((expr, result, m.start(), m.end()))

    if not calcs:
        return None

    # Build steps: split reasoning text around each <<>> marker
    # and pair each chunk of reasoning with its calculator call.
    segments: list[str] = []
    prev_end: int = 0
    tool_id: int = 1

    for expr, result, start, end in calcs:
        # The reasoning text before this calculation
        text_before: str = reasoning_part[prev_end:start].strip()
        # Clean up any leftover result text after >> (e.g. "24 clips in May.")
        prev_end = end

        # Evaluate the expression with our calculator to get the real result
        try:
            real_result: str = calculator(expression=expr)
        except Exception:
            real_result = result

        # Build think block with tool call
        reasoning_text: str = text_before if text_before else f"I compute {expr}."
        # Remove any <<>> markers that might be in the reasoning text
        reasoning_text = _CALC_RE.sub("", reasoning_text).strip()
        if not reasoning_text:
            reasoning_text = f"I compute {expr}."

        think_content: str = (
            f'{reasoning_text} @calculator(expression="{expr}")->{tool_id}.'
        )
        tools_block: str = _render_tools_block(
            str(tool_id), expr, real_result
        )
        segments.append(f"<think>{think_content}</think>\n{tools_block}")
        tool_id += 1

    # If we have more iterations than allowed, batch the last ones together
    if len(segments) > max_iterations - 1:
        # Keep first (max_iterations - 2) individual steps, batch the rest
        kept: list[str] = segments[: max_iterations - 2]
        # For the batched segment, just keep the last one
        kept.append(segments[-1])
        segments = kept

    # Final think + answer referencing the last tool ID
    last_id: int = tool_id - 1
    final_think: str = f"<think>The answer is @{last_id}.</think>"
    final_answer_block: str = f"<answer>@{last_id}</answer>"
    segments.append(f"{final_think}\n{final_answer_block}")

    return "\n".join(segments)


def _example_to_gold_sft_text(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    system_prompt: str,
) -> str | None:
    """Generate a gold SFT training string for one GSM8K example.

    Instead of running the model (which produces bad habits), this builds
    a perfect transcript synthetically from the GSM8K answer annotations.

    Args:
        example: Dataset row with "question" and "answer".
        tokenizer: HF tokenizer.
        system_prompt: Full system prompt with tool descriptions.

    Returns:
        Rendered chat-template training text, or None if unparseable.
    """
    question: str = str(example["question"])
    answer_field: str = str(example["answer"])
    wrapped_question: str = f"<question>{question}</question>"

    transcript: str | None = _build_gold_transcript(answer_field)
    if transcript is None:
        return None

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": wrapped_question},
        {"role": "assistant", "content": transcript},
    ]

    return str(
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    """Create a BitsAndBytes quantization config."""
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def train() -> None:
    """Run LoRA SFT on GSM8K using gold tool-call transcripts."""
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

    # Build gold transcripts (no GPU needed — pure string manipulation)
    texts: list[str] = []
    skipped: int = 0
    for ex in tqdm(dataset, total=len(dataset), desc="Building gold transcripts"):
        text: str | None = _example_to_gold_sft_text(
            ex,
            tokenizer=tokenizer,
            system_prompt=str(cfg.system_prompt),
        )
        if text is not None:
            texts.append(text)
        else:
            skipped += 1

    print(f"Gold transcripts: {len(texts)} built, {skipped} skipped.")
    dataset_with_text: Dataset = Dataset.from_dict({"text": texts})

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

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset_with_text,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_args,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(output_dir=str(cfg.checkpoint_directory))
    tokenizer.save_pretrained(save_directory=str(cfg.checkpoint_directory))


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    train()
