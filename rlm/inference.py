from __future__ import annotations

import re
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from utils import check_cwd

from rlm.config import INFERENCE_CONFIG as CONFIG
from rlm.config import REPO_DIR


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    """Build a BitsAndBytes quantization config matching the training setup.

    Args:
        use_4bit: Whether to enable 4-bit NF4 quantization.

    Returns:
        BitsAndBytesConfig if 4-bit is enabled, otherwise None.
    """
    if not use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _build_messages(*, question: str, system_prompt: str) -> list[dict[str, str]]:
    """Create system/user messages for the chat template.

    Args:
        question: User question to be solved.
        system_prompt: System prompt defining format and behavior.

    Returns:
        A list of chat messages for apply_chat_template().
    """
    wrapped_question: str = f"<question>{question}</question>"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": wrapped_question},
    ]


def _extract_think_answer_block(*, text: str) -> str:
    """Extract the strict <think>...</think><answer>...</answer> block.

    Args:
        text: Generated text to parse.

    Returns:
        Substring from the first '<think>' up to the first '</answer>' (inclusive).

    Raises:
        ValueError: If the required tag block is not found.
    """
    pattern: re.Pattern[str] = re.compile(r"(<think>.*?</answer>)", flags=re.DOTALL)
    match: re.Match[str] | None = pattern.search(text)
    if match is None:
        raise ValueError("No <think>... </answer> block found in generated text.")
    return match.group(1).strip()


def load_rlm_model(
    *,
    cfg: CONFIG | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the base model and attach the trained LoRA adapter.

    Args:
        cfg: Optional SFT configuration instance. If None, a default one is created.

    Returns:
        A tuple (model, tokenizer) ready for inference.
    """
    cfg = cfg or CONFIG()

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=str(cfg.model_name),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(cfg.model_name),
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    model.generation_config.eos_token_id = int(tokenizer.eos_token_id)

    model = PeftModel.from_pretrained(
        model=model,
        model_id=str(cfg.checkpoint_directory),
    )
    model.eval()

    return model, tokenizer


@torch.inference_mode()
def generate_reasoning(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    cfg: CONFIG | None = None,
) -> str:
    """Generate a sectioned response for a question using the trained model.

    Args:
        prompt: The math question to solve.
        model: The loaded (base + LoRA) causal LM.
        tokenizer: The corresponding tokenizer.
        cfg: Optional SFT configuration instance.

    Returns:
        The generated assistant text strictly from <think> to </answer>.
    """
    cfg = cfg or CONFIG()

    messages: list[dict[str, str]] = _build_messages(
        question=prompt,
        system_prompt=str(cfg.system_prompt),
    )

    input_ids: torch.Tensor = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": int(cfg.max_new_tokens),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "do_sample": bool(cfg.do_sample),
    }

    if bool(cfg.do_sample):
        if cfg.temperature is not None:
            generation_kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_p is not None:
            generation_kwargs["top_p"] = float(cfg.top_p)
        if getattr(cfg, "top_k", None) is not None:
            generation_kwargs["top_k"] = int(cfg.top_k)

    outputs: torch.Tensor = model.generate(**generation_kwargs)

    prompt_len: int = int(input_ids.shape[-1])
    new_tokens: torch.Tensor = outputs[0, prompt_len:]
    generated_text: str = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return _extract_think_answer_block(text=generated_text)


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    cfg_: CONFIG = CONFIG()
    model_, tokenizer_ = load_rlm_model(cfg=cfg_)

    test_prompt_: str = (
        "Si tengo 3 manzanas y me dan el doble de las que tengo menos una, "
        "¿cuántas tengo?"
    )

    print(generate_reasoning(test_prompt_, model_, tokenizer_, cfg=cfg_))
