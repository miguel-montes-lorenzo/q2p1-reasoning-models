# check_sft.py

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import CONFIG


@dataclass(frozen=True)
class LoadedModels:
    """Container for base and (optionally) adapted models.

    Args:
        base_model: Base causal LM loaded for inference.
        adapted_model: Base model with a PEFT adapter loaded (if provided).
        tokenizer: Tokenizer used for both models.
    """

    base_model: torch.nn.Module
    adapted_model: torch.nn.Module | None
    tokenizer: Any


@dataclass(frozen=True)
class QAItem:
    """Single QA item loaded from the questions JSON.

    Args:
        question: User question string.
        answer: Ground-truth answer (raw; later reduced to its last alnum).
    """

    question: str
    answer: str


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
        # Keep consistent with BF16 training.
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _align_special_tokens(*, model: Any, tokenizer: Any, cfg: CONFIG) -> None:
    """Align model and generation configs with tokenizer special tokens.

    Args:
        model: HF model with config and generation_config.
        tokenizer: HF tokenizer with pad/eos token ids.
        cfg: Shared configuration for generation hyperparameters.
    """
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.do_sample = False
    model.generation_config.temperature = cfg.temperature
    model.generation_config.top_p = cfg.top_p
    model.generation_config.top_k = None


def _load_base_model_and_tokenizer(*, cfg: CONFIG) -> tuple[torch.nn.Module, Any]:
    """Load the base model and tokenizer.

    Args:
        cfg: Shared configuration.

    Returns:
        (base_model, tokenizer) ready for inference.
    """
    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    base_model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    _align_special_tokens(model=base_model, tokenizer=tokenizer, cfg=cfg)
    base_model.eval()

    return (base_model, tokenizer)


def _resolve_adapter_dir(*, path: Path | None) -> Path | None:
    """Resolve an adapter directory from a user-provided path.

    Accepts either:
      - a directory containing adapter_config.json, or
      - a file inside such a directory, or
      - a "best-checkpoint" style directory name.

    Args:
        path: Candidate path.

    Returns:
        A resolved adapter directory path, or None.

    Raises:
        FileNotFoundError: If a path is provided but does not exist.
        ValueError: If a path exists but no adapter_config.json can be found.
    """
    if path is None:
        return None

    p: Path = path.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Adapter path not found: {p}")

    adapter_cfg: str = "adapter_config.json"

    if p.is_file():
        p = p.parent

    if (p / adapter_cfg).exists():
        return p

    # Common typo: "best_checkpoint" vs "best-checkpoint"
    alt: Path = p.parent / p.name.replace("_", "-")
    if alt != p and alt.exists() and (alt / adapter_cfg).exists():
        return alt

    raise ValueError(f"Can't find '{adapter_cfg}' under: {p}")


def _maybe_load_adapter(
    *,
    base_model: torch.nn.Module,
    adapter_dir: Path | None,
) -> torch.nn.Module | None:
    """Optionally load a PEFT adapter on top of the given base model.

    Args:
        base_model: Base model to wrap with a PEFT adapter.
        adapter_dir: Directory containing adapter_config.json and adapter weights.

    Returns:
        A PeftModel if adapter_dir is provided, else None.
    """
    if adapter_dir is None:
        return None

    model: Any = PeftModel.from_pretrained(
        model=base_model,
        model_id=str(adapter_dir),
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    return model


def _render_chat_prompt(*, prompt: str, cfg: CONFIG, tokenizer: Any) -> str:
    """Render a chat prompt using the model's chat template and system prompt.

    Args:
        prompt: User prompt.
        cfg: Shared configuration.
        tokenizer: Tokenizer providing apply_chat_template.

    Returns:
        Rendered string prompt.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(
    *,
    prompt: str,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: CONFIG,
) -> str:
    """Generate a completion using the chat template + system prompt.

    This returns only the generated completion tokens (not the prompt).

    Args:
        prompt: The user prompt string.
        model: Loaded model.
        tokenizer: Tokenizer.
        cfg: Shared configuration (includes system prompt and generation limits).

    Returns:
        Decoded generation (completion only).
    """
    full_prompt: str = _render_chat_prompt(prompt=prompt, cfg=cfg, tokenizer=tokenizer)
    enc: Any = tokenizer(full_prompt, return_tensors="pt", truncation=True)

    device: torch.device = next(model.parameters()).device
    input_ids: torch.Tensor = enc["input_ids"].to(device=device)
    attention_mask: torch.Tensor = enc["attention_mask"].to(device=device)

    with torch.no_grad():
        out: torch.Tensor = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(cfg.max_new_tokens),
            do_sample=False,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
        )

    prompt_len: int = int(input_ids.shape[1])
    gen_ids: torch.Tensor = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _has_think_tags(*, text: str) -> bool:
    """Check whether the output contains <think>...</think> tags.

    Args:
        text: Model output string.

    Returns:
        True if tags are present, else False.
    """
    return ("<think>" in text) and ("</think>" in text)


def _extract_last_alnum(*, text: str) -> str:
    """Extract the last alphanumeric character from a string.

    This is used to report just the last number/letter produced by the model.

    Args:
        text: Model output string.

    Returns:
        The last [A-Za-z0-9] character, or "?" if none is found.
    """
    matches: list[str] = re.findall(pattern=r"[A-Za-z0-9]", string=text)
    if len(matches) == 0:
        return "?"
    return matches[-1]


def _compute_accuracy(*, correct: list[str], pred: list[str]) -> float:
    """Compute accuracy as a fraction in [0, 1].

    Args:
        correct: Ground-truth labels (same length as pred).
        pred: Predicted labels.

    Returns:
        Accuracy fraction (correct matches / total). Returns 0.0 if empty.
    """
    assert len(correct) == len(pred)
    if len(correct) == 0:
        return 0.0
    hits: int = sum((c == p) for c, p in zip(correct, pred, strict=True))
    return float(hits) / float(len(correct))


def _print_answer_table(
    *,
    rows: list[int],
    correct: list[str],
    base_pred: list[str],
    ckpt_pred: list[str],
    checkpoint_name: str,
) -> None:
    """Print a small ASCII table with correct + predicted answers.

    Args:
        rows: Row identifiers (prompt numbers starting at 1).
        correct: Per-row correct answers (already reduced to last number/letter).
        base_pred: Per-row base model predicted answers (last number/letter).
        ckpt_pred: Per-row checkpoint model predicted answers (last number/letter).
        checkpoint_name: Column header for the checkpoint model.
    """
    assert len(rows) == len(correct) == len(base_pred) == len(ckpt_pred)

    col0: str = "#"
    col1: str = "correct"
    col2: str = "base"
    col3: str = checkpoint_name

    w0: int = max(len(col0), max((len(str(r)) for r in rows), default=1))
    w1: int = max(len(col1), max((len(x) for x in correct), default=1))
    w2: int = max(len(col2), max((len(x) for x in base_pred), default=1))
    w3: int = max(len(col3), max((len(x) for x in ckpt_pred), default=1))

    header: str = f"{col0:<{w0}} | {col1:<{w1}} | {col2:<{w2}} | {col3:<{w3}}"
    sep: str = f"{'-' * w0}-+-{'-' * w1}-+-{'-' * w2}-+-{'-' * w3}"

    print("\n=== SUMMARY TABLE ===")
    print(header)
    print(sep)
    for r, c, b, k in zip(rows, correct, base_pred, ckpt_pred, strict=True):
        line: str = f"{str(r):<{w0}} | {c:<{w1}} | {b:<{w2}} | {k:<{w3}}"
        print(line)

    base_acc: float = _compute_accuracy(correct=correct, pred=base_pred)
    ckpt_acc: float = _compute_accuracy(correct=correct, pred=ckpt_pred)

    acc_label: str = "acc"
    w0_acc: int = w0
    w1_acc: int = w1
    w2_acc: int = w2
    w3_acc: int = w3

    print(sep)
    print(
        f"{acc_label:<{w0_acc}} | {'':<{w1_acc}} | "
        f"{(base_acc * 100):>6.2f}%{'':<{max(0, w2_acc - 7)}} | "
        f"{(ckpt_acc * 100):>6.2f}%{'':<{max(0, w3_acc - 7)}}"
    )


def _load_questions(*, questions_path: Path) -> list[QAItem]:
    """Load questions from a JSON file.

    Expected JSON format:
        [
          {"question": "...", "answer": "..."},
          ...
        ]

    Args:
        questions_path: Path to the JSON file.

    Returns:
        List of QAItem entries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is not a list or fields are missing/invalid.
    """
    p: Path = questions_path.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {p}")

    raw_text: str = p.read_text(encoding="utf-8")
    data: Any = json.loads(raw_text)

    if not isinstance(data, list):
        raise ValueError("Questions JSON must be a list of objects.")

    items: list[QAItem] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Questions[{idx}] must be an object/dict.")

        q_any: Any = item.get("question")
        if not isinstance(q_any, str) or not q_any.strip():
            raise ValueError(f"Questions[{idx}]['question'] must be a non-empty string.")
        question: str = q_any.strip()

        a_any: Any = item.get("answer")
        if a_any is None:
            raise ValueError(f"Questions[{idx}]['answer'] is missing.")
        answer: str = str(a_any).strip()
        if not answer:
            raise ValueError(f"Questions[{idx}]['answer'] must be non-empty.")

        items.append(QAItem(question=question, answer=answer))

    if len(items) == 0:
        raise ValueError("Questions JSON is empty (no questions found).")

    return items


def main(
    *,
    checkpoint_path: Path,
    questions_path: Path,
    silent: bool = True,
) -> None:
    """Compare base model vs checkpoint-adapted model outputs on a questions set.

    Args:
        checkpoint_path: Path to the adapter directory (checkpoint or final adapter).
        questions_path: Path to a JSON file containing a list of questions.
        silent: If True, do not print prompts/outputs; show only a progress bar.
            If False, print each prompt and both model outputs.
    """
    cfg: CONFIG = CONFIG()

    base_model, tokenizer = _load_base_model_and_tokenizer(cfg=cfg)

    adapter_dir: Path | None = _resolve_adapter_dir(path=checkpoint_path)
    adapted_model: torch.nn.Module | None = _maybe_load_adapter(
        base_model=base_model,
        adapter_dir=adapter_dir,
    )
    if adapted_model is None:
        raise ValueError(f"Could not load adapter from: {checkpoint_path}")

    checkpoint_name: str = adapter_dir.name if adapter_dir is not None else "checkpoint"
    items: list[QAItem] = _load_questions(questions_path=questions_path)

    rows: list[int] = []
    correct: list[str] = []
    base_pred: list[str] = []
    ckpt_pred: list[str] = []

    iterable: Any
    if silent:
        iterable = tqdm(
            enumerate(items, start=1),
            total=len(items),
            desc="Evaluating questions",
            dynamic_ncols=True,
        )
    else:
        iterable = enumerate(items, start=1)

    for i, item in iterable:
        rows.append(i)

        correct_ans: str = _extract_last_alnum(text=item.answer)
        correct.append(correct_ans)

        if not silent:
            print(f"\n=== PROMPT {i} ===")
            print(item.question)
            print(f"\n--- CORRECT (last char) ---\n{correct_ans}")

        base_out: str = _generate(
            prompt=item.question,
            model=base_model,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        base_last: str = _extract_last_alnum(text=base_out)
        base_pred.append(base_last)

        adapted_out: str = _generate(
            prompt=item.question,
            model=adapted_model,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        ckpt_last: str = _extract_last_alnum(text=adapted_out)
        ckpt_pred.append(ckpt_last)

        if not silent:
            print("\n--- BASE MODEL (completion) ---")
            print(base_out)
            print(f"\nBASE (last char): {base_last}")
            print("\n--- CHECKPOINT MODEL (completion) ---")
            print(adapted_out)
            print(f"\nCHECKPOINT (last char): {ckpt_last}")

    _print_answer_table(
        rows=rows,
        correct=correct,
        base_pred=base_pred,
        ckpt_pred=ckpt_pred,
        checkpoint_name=checkpoint_name,
    )
    print("\nDone: base vs checkpoint evaluation completed.")


def _parse_optional_path_from_argv(
    *, argv: list[str]
) -> tuple[Path | None, Path | None]:
    """Parse optional checkpoint/questions paths from argv.

    Usage:
        python check_sft.py [checkpoint_path] [questions_path]

    Args:
        argv: sys.argv list.

    Returns:
        (checkpoint_path, questions_path) where each can be None.

    Raises:
        FileNotFoundError: If a provided path does not exist.
    """
    if len(argv) <= 1:
        return (None, None)

    ckpt: Path = Path(argv[1]).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")

    if len(argv) <= 2:
        return (ckpt, None)

    questions: Path = Path(argv[2]).expanduser().resolve()
    if not questions.exists():
        raise FileNotFoundError(f"Questions path not found: {questions}")

    return (ckpt, questions)


if __name__ == "__main__":
    adapter, questions = _parse_optional_path_from_argv(argv=sys.argv)

    if adapter is None:
        adapter = Path("weights/sft_lora/best-checkpoint")
    if questions is None:
        questions = Path("questions.json")

    main(checkpoint_path=adapter, questions_path=questions, silent=True)
