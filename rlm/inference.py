from __future__ import annotations

from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ruta a tu modelo final de fase 1
MODEL_PATH: str = "./weights/final_rlm_lora"
BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

USE_4BIT: bool = True
MAX_NEW_TOKENS: int = 512


def load_rlm_model() -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load the base model and the LoRA adapter for reasoning generation.

    Returns:
        A tuple (model, tokenizer) ready for inference.
    """
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
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model=base_model, model_id=MODEL_PATH)
    model.eval()
    return (model, tokenizer)


def generate_reasoning(*, prompt: str, model: torch.nn.Module, tokenizer: Any) -> str:
    """Generate a response including visible chain-of-thought tags.

    Args:
        prompt: User prompt/question.
        model: Loaded causal LM with adapter.
        tokenizer: Corresponding tokenizer.

    Returns:
        Decoded model response.
    """
    full_prompt: str = f"User: {prompt}\nAssistant:"
    enc = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
    )
    device: torch.device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device=device)
    attention_mask = enc["attention_mask"].to(device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
        )

    response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Prueba local
    model, tokenizer = load_rlm_model()
    test_prompt: str = (
        "Si tengo 3 manzanas y me dan el doble de las que tengo menos una, "
        "¿cuántas tengo?"
    )
    print(generate_reasoning(prompt=test_prompt, model=model, tokenizer=tokenizer))
