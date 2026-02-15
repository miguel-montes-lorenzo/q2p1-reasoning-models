from __future__ import annotations

from typing import Any

import os
import torch

try:
    # When imported as `rlm.inference` (e.g. from the API), prefer package-relative import.
    from .config import SFT_CONFIG as CONFIG  # type: ignore
except ImportError:
    # Fallback for running `python inference.py` from inside the rlm directory.
    from config import SFT_CONFIG as CONFIG
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ruta a tu modelo final de fase 1
# MODEL_PATH: str = "./weights/final_rlm_lora"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH: str = os.path.join(BASE_DIR, "rlm/weights/sft_lora/checkpoint-800")
BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

USE_4BIT: bool = True
MAX_NEW_TOKENS: int = 1024


def load_rlm_model() -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load the base model and the LoRA adapter for reasoning generation.

    Returns:
        A tuple (model, tokenizer) ready for inference.
    """
    cfg: CONFIG = CONFIG()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model=base_model, model_id=MODEL_PATH)

    # Align special tokens/config so generation behaves consistently.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.do_sample = bool(cfg.do_sample)
    model.generation_config.temperature = cfg.temperature
    model.generation_config.top_p = cfg.top_p

    model.eval()
    return (model, tokenizer)


def generate_reasoning(
    *,
    prompt: str,
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str | None = None,
) -> str:
    """Generate a response including visible chain-of-thought tags.

    Args:
        prompt: User prompt/question.
        model: Loaded causal LM with adapter.
        tokenizer: Corresponding tokenizer.

    Returns:
        Decoded model response.
    """
    cfg: CONFIG = CONFIG()
    active_system_prompt: str = system_prompt if system_prompt is not None else cfg.system_prompt

    # Use the model's chat template + system prompt (same as check_answers.py)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": active_system_prompt},
        {"role": "user", "content": prompt},
    ]
    full_prompt: str = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    device: torch.device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device=device)
    attention_mask = enc["attention_mask"].to(device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(cfg.max_new_tokens) if hasattr(cfg, "max_new_tokens") else MAX_NEW_TOKENS,
            do_sample=bool(cfg.do_sample),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
        )

    # Return completion only (exclude the prompt tokens)
    prompt_len: int = int(input_ids.shape[1])
    gen_ids: torch.Tensor = outputs[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


if __name__ == "__main__":
    # Prueba local
    model, tokenizer = load_rlm_model()
    test_prompt: str = (
        "Si tengo 3 manzanas y me dan el doble de las que tengo menos una, "
        "¿cuántas tengo?"
    )
    print(generate_reasoning(prompt=test_prompt, model=model, tokenizer=tokenizer))
