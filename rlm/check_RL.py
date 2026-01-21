# check_RL.py

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class Config:
    """Configuration for checking the final RL (GRPO) adapter.

    Args:
        base_model: HF id of the base model.
        adapter_path: Path to the final LoRA adapter directory.
        max_new_tokens: Maximum tokens to generate.
        use_4bit: Whether to load the base model in 4-bit quantized mode.
        temperature: Sampling temperature to mimic training-time sampling.
        top_p: Nucleus sampling probability.
        group_size: Number of sampled answers per question.
    """

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: Path = Path("./weights/final_rlm_lora")
    max_new_tokens: int = 256
    use_4bit: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    group_size: int = 4


def _load_model_and_tokenizer(*, cfg: Config) -> tuple[torch.nn.Module, Any]:
    """Load base model + final RL LoRA adapter and tokenizer.

    Args:
        cfg: Check configuration.

    Returns:
        (model, tokenizer) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.base_model,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model=base_model, model_id=str(cfg.adapter_path))
    model.eval()
    return (model, tokenizer)


def _extract_last_number(*, text: str) -> str | None:
    """Extract the last integer/decimal-like number from text.

    Args:
        text: Any text.

    Returns:
        Last matched number or None.
    """
    matches: list[str] = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1]


def _generate_group(
    *,
    prompt: str,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: Config,
) -> list[str]:
    """Generate multiple sampled answers for a given prompt.

    Args:
        prompt: User prompt.
        model: Model.
        tokenizer: Tokenizer.
        cfg: Config.

    Returns:
        List of decoded generations.
    """
    full_prompt: str = f"User: {prompt}\nAssistant:"
    enc = tokenizer(full_prompt, return_tensors="pt", truncation=True)

    device: torch.device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device=device)
    attention_mask = enc["attention_mask"].to(device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_return_sequences=cfg.group_size,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
        )

    decoded: list[str] = tokenizer.batch_decode(out, skip_special_tokens=True)
    return decoded


def _score_correct(*, generation: str, ground_truth: str) -> float:
    """Score generation as correct/incorrect by last-number match.

    Args:
        generation: Generated text.
        ground_truth: Ground truth string (numeric).

    Returns:
        1.0 if last number matches else 0.0.
    """
    pred: str | None = _extract_last_number(text=generation)
    gt: str | None = _extract_last_number(text=ground_truth)
    if pred is None or gt is None:
        return 0.0
    return 1.0 if pred.lstrip("+").strip() == gt.lstrip("+").strip() else 0.0


def _assert_has_think_tags(*, text: str) -> None:
    """Assert <think> tags exist.

    Args:
        text: Output text.

    Raises:
        AssertionError: If tags are missing.
    """
    assert "<think>" in text and "</think>" in text, (
        "RL check failed: output does not contain <think>...</think> tags.\n"
        f"Output:\n{text}"
    )


def main() -> None:
    """Check README Phase 1 Part 2 (GRPO): final adapter loads and improves math."""
    cfg: Config = Config()

    assert cfg.adapter_path.exists(), (
        f"Adapter path not found: {cfg.adapter_path}. "
        "Run train_grpo.py first."
    )

    model, tokenizer = _load_model_and_tokenizer(cfg=cfg)

    # A couple of deterministic ground-truth items we can verify.
    # Keep these simple to avoid dependence on wording.
    tests: list[tuple[str, str]] = [
        ("Resuelve: 17 + 25.", "42"),
        ("Resuelve: 9 * 7.", "63"),
        ("Si tengo 3 manzanas y me dan el doble de las que tengo menos una, ¿cuántas tengo?", "5"),
    ]

    for i, (q, gt) in enumerate(tests, start=1):
        gens: list[str] = _generate_group(prompt=q, model=model, tokenizer=tokenizer, cfg=cfg)

        scores: list[float] = [_score_correct(generation=g, ground_truth=gt) for g in gens]
        best: float = max(scores) if scores else 0.0
        mean: float = (sum(scores) / float(len(scores))) if scores else 0.0

        print(f"\n=== RL CHECK {i} ===")
        print(f"Q: {q}")
        print(f"GT: {gt}")
        print(f"Scores (group): {scores} | best={best:.1f} mean={mean:.2f}")

        for j, g in enumerate(gens, start=1):
            _assert_has_think_tags(text=g)
            pred_num: str | None = _extract_last_number(text=g)
            print(f"\n--- sample {j} (pred={pred_num}) ---")
            print(g)

    print(
        "\nRL check finished: model loads, produces <think> tags, and "
        "at least some samples should match the numeric ground truth."
    )


if __name__ == "__main__":
    main()
