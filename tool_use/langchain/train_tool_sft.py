# tool_use/langchain/train_tool_sft.py

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from utils.paths import check_cwd

from tool_use.langchain.config import REPO_DIR
from tool_use.langchain.config import SFT_CONFIG as CONFIG
from tool_use.langchain.tool_handler import insert_tool_desciptions_in_system_propt
from tool_use.langchain.tool_inference import run_tool_use_inference
from tool_use.langchain.tools import TOOL_DICT, get_langchain_tools


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


def _example_to_sft_text(
    example: dict[str, Any],
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: Any,
    tools: list[Any],
) -> str:
    """Generate a single SFT training string for one GSM8K example.

    This runs the tool-use loop in-process (GPU-safe) and wraps the resulting
    transcript as the assistant turn.

    Args:
        example: Dataset row with "question".
        model: HF causal LM.
        tokenizer: HF tokenizer.
        cfg: Tool-use inference config.
        tools: LangChain tools.

    Returns:
        Rendered chat-template training text.
    """
    question: str = str(example["question"])
    wrapped_question: str = f"<question>{question}</question>"

    full_output: str
    _parsed_answer_only: str
    _step_contents: list[str]
    full_output, _parsed_answer_only, _step_contents = run_tool_use_inference(
        question=question,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        tools=tools,
        formatted_references=True,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(cfg.system_prompt)},
        {"role": "user", "content": wrapped_question},
        {"role": "assistant", "content": full_output},
    ]

    return str(
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )


def train() -> None:
    """Run LoRA SFT on GSM8K using tool-loop generated transcripts."""
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

    tools: list[Any] = get_langchain_tools()

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

    # Build texts sequentially to avoid CUDA-in-fork issues.
    texts: list[str] = []
    for ex in tqdm(
        dataset, total=len(dataset), desc="Generating tool-loop transcripts"
    ):
        text: str = _example_to_sft_text(
            ex,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            tools=tools,
        )
        texts.append(text)

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
