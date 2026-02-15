# check_answers.py

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from config import SFT_CONFIG as CONFIG
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class QAItem:
    """Single QA item loaded from the questions JSON.

    Args:
        question: User question string.
        answer: Ground-truth answer (raw; later reduced to its last digit run).
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

    model.generation_config.do_sample = bool(cfg.do_sample)
    model.generation_config.temperature = cfg.temperature
    model.generation_config.top_p = cfg.top_p
    model.generation_config.top_k = None


def _load_tokenizer(*, cfg: CONFIG) -> Any:
    """Load the tokenizer once (shared by all models).

    Args:
        cfg: Shared configuration.

    Returns:
        HF tokenizer.
    """
    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(*, cfg: CONFIG, tokenizer: Any) -> torch.nn.Module:
    """Load a fresh base model for inference.

    Args:
        cfg: Shared configuration.
        tokenizer: Tokenizer whose special tokens must be aligned.

    Returns:
        Base model (no LoRA attached).
    """
    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    base_model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    _align_special_tokens(model=base_model, tokenizer=tokenizer, cfg=cfg)
    base_model.eval()
    return cast(torch.nn.Module, base_model)


def _resolve_adapter_dir(*, path: Path) -> Path:
    """Resolve an adapter directory from a user-provided path.

    Accepts either:
      - a directory containing adapter_config.json, or
      - a file inside such a directory.

    Args:
        path: Candidate adapter path.

    Returns:
        A resolved adapter directory path.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If a path exists but no adapter_config.json can be found.
    """
    p: Path = path.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Adapter path not found: {p}")

    adapter_cfg: str = "adapter_config.json"

    if p.is_file():
        p = p.parent

    if (p / adapter_cfg).exists():
        return p

    alt: Path = p.parent / p.name.replace("_", "-")
    if alt != p and alt.exists() and (alt / adapter_cfg).exists():
        return alt

    raise ValueError(f"Can't find '{adapter_cfg}' under: {p}")


def _load_adapted_model(
    *,
    cfg: CONFIG,
    tokenizer: Any,
    adapter_dir: Path,
) -> torch.nn.Module:
    """Load a fresh base model and attach a LoRA adapter on top of it.

    Args:
        cfg: Shared configuration.
        tokenizer: Tokenizer whose special tokens must be aligned.
        adapter_dir: Directory containing adapter_config.json and adapter weights.

    Returns:
        A PeftModel instance with the adapter loaded.
    """
    fresh_base: torch.nn.Module = _load_base_model(cfg=cfg, tokenizer=tokenizer)
    adapted: Any = PeftModel.from_pretrained(
        model=fresh_base,
        model_id=str(adapter_dir),
        is_trainable=False,
        local_files_only=True,
    )
    _align_special_tokens(model=adapted, tokenizer=tokenizer, cfg=cfg)
    adapted.eval()
    return cast(torch.nn.Module, adapted)


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

    Args:
        prompt: The user prompt string.
        model: Loaded model.
        tokenizer: Tokenizer.
        cfg: Shared configuration (includes generation limits).

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
            do_sample=bool(cfg.do_sample),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
        )

    prompt_len: int = int(input_ids.shape[1])
    gen_ids: torch.Tensor = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _extract_answer_section(*, text: str) -> str:
    """Extract the content inside the <answer>...</answer> section.

    Args:
        text: Model completion text.

    Returns:
        The extracted answer content if tags are found, otherwise the original text.
    """
    m: re.Match[str] | None = re.search(
        pattern=r"<answer>\s*(.*?)\s*</answer>",
        string=text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m is None:
        return text
    return m.group(1).strip()


def _extract_last_digit_run(*, text: str) -> str:
    """Extract the last consecutive run of digits from a string.

    Args:
        text: Model output (or ground-truth answer) as a string.

    Returns:
        The last substring matching r"\\d+", or "?" if none is found.
    """
    matches: list[str] = re.findall(pattern=r"\d+", string=text)
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


def _make_unique_names(*, base_names: list[str]) -> list[str]:
    """Make adapter names unique by appending ' (k)' to duplicates.

    Example:
        ["best-checkpoint", "best-checkpoint", "final"] ->
        ["best-checkpoint (1)", "best-checkpoint (2)", "final"]

    Args:
        base_names: Proposed names, possibly with duplicates.

    Returns:
        Unique names, stable-order.
    """
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in base_names:
        seen: int = counts.get(name, 0) + 1
        counts[name] = seen
        if base_names.count(name) == 1:
            out.append(name)
        else:
            out.append(f"{name}_{seen}")
    return out


def _print_answer_table(
    *,
    rows: list[int],
    correct: list[str],
    base_pred: list[str],
    adapter_preds: list[list[str]],
    adapter_names: list[str],
) -> None:
    """Print an ASCII table with correct + predicted answers for many adapters.

    Args:
        rows: Row identifiers (prompt numbers starting at 1).
        correct: Per-row correct answers (last digit run).
        base_pred: Per-row base model predicted answers (last digit run).
        adapter_preds: List of per-adapter predictions, each of length len(rows).
        adapter_names: Per-adapter column names (same length as adapter_preds).
    """
    assert len(rows) == len(correct) == len(base_pred)
    assert len(adapter_preds) == len(adapter_names)
    for preds in adapter_preds:
        assert len(preds) == len(rows)

    headers: list[str] = ["#", "correct", "base", *adapter_names]

    columns: list[list[str]] = []
    columns.append([str(r) for r in rows])
    columns.append(correct)
    columns.append(base_pred)
    for preds in adapter_preds:
        columns.append(preds)

    widths: list[int] = []
    for header, col in zip(headers, columns, strict=True):
        w: int = max(len(header), max((len(x) for x in col), default=1))
        widths.append(max(w, 3))

    def _fmt_row(*, items: list[str]) -> str:
        parts: list[str] = []
        for item, w in zip(items, widths, strict=True):
            parts.append(f"{item:<{w}}")
        return " | ".join(parts)

    sep: str = "-+-".join("-" * w for w in widths)

    print("\n=== SUMMARY TABLE ===")
    print(_fmt_row(items=headers))
    print(sep)

    for i in range(len(rows)):
        row_items: list[str] = [columns[j][i] for j in range(len(columns))]
        print(_fmt_row(items=row_items))

    base_acc: float = _compute_accuracy(correct=correct, pred=base_pred)
    adapter_accs: list[float] = [
        _compute_accuracy(correct=correct, pred=preds) for preds in adapter_preds
    ]

    acc_cells: list[str] = ["acc", "", f"{base_acc:>4.2f}"]
    acc_cells.extend(f"{a:>4.2f}" for a in adapter_accs)

    print(sep)
    print(_fmt_row(items=acc_cells))


def _load_questions(*, questions_path: Path) -> list[QAItem]:
    """Load questions from a JSON file.

    Args:
        questions_path: Path to the JSON file.

    Returns:
        List of QAItem entries.
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
            raise ValueError(
                f"Questions[{idx}]['question'] must be a non-empty string."
            )
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


def _write_answers_json(
    *,
    answers_path: Path,
    items: list[QAItem],
    base_out: list[str],
    adapter_outs: list[list[str]],
    adapter_names: list[str],
) -> None:
    """Write per-question outputs (base + adapters) to a JSON file.

    Args:
        answers_path: Destination path for the JSON file.
        items: Original QA items (questions).
        base_out: Base model raw outputs (completion only).
        adapter_outs: Per-adapter raw outputs; each list has length len(items).
        adapter_names: Unique names for adapter outputs (same length as adapter_outs).
    """
    assert len(items) == len(base_out)
    assert len(adapter_outs) == len(adapter_names)
    for outs in adapter_outs:
        assert len(outs) == len(items)

    out_data: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        row: dict[str, Any] = {
            "question": item.question,
            "base_output": base_out[i],
        }
        for name, outs in zip(adapter_names, adapter_outs, strict=True):
            row[f"{name}_output"] = outs[i]
        out_data.append(row)

    p: Path = answers_path.expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(
    *,
    adapter_paths: list[Path],
    questions_path: Path,
    answers_path: Path,
) -> None:
    """Compare base model vs multiple adapter models on a questions set.

    Args:
        adapter_paths: List of adapter directories (or paths within them).
        questions_path: Path to a JSON file containing a list of questions.
        answers_path: Path to the JSON file where outputs will be written.
    """
    cfg: CONFIG = CONFIG()

    tokenizer: Any = _load_tokenizer(cfg=cfg)
    base_model: torch.nn.Module = _load_base_model(cfg=cfg, tokenizer=tokenizer)

    resolved_dirs: list[Path] = [_resolve_adapter_dir(path=p) for p in adapter_paths]

    raw_names: list[str] = [d.name for d in resolved_dirs]
    adapter_names: list[str] = _make_unique_names(base_names=raw_names)

    adapted_models: list[torch.nn.Module] = [
        _load_adapted_model(cfg=cfg, tokenizer=tokenizer, adapter_dir=d)
        for d in resolved_dirs
    ]

    items: list[QAItem] = _load_questions(questions_path=questions_path)

    rows: list[int] = []
    correct: list[str] = []
    base_pred: list[str] = []
    adapter_preds: list[list[str]] = [[] for _ in adapted_models]

    base_outputs: list[str] = []
    adapter_outputs: list[list[str]] = [[] for _ in adapted_models]

    iterable: Any = tqdm(
        enumerate(items, start=1),
        total=len(items),
        desc="Evaluating questions",
        dynamic_ncols=True,
    )

    for i, item in iterable:
        rows.append(i)

        correct_ans: str = _extract_last_digit_run(text=item.answer)
        correct.append(correct_ans)

        base_out: str = _generate(
            prompt=item.question,
            model=base_model,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        base_outputs.append(base_out)
        base_ans: str = _extract_answer_section(text=base_out)
        base_pred.append(_extract_last_digit_run(text=base_ans))

        for j, model in enumerate(adapted_models):
            out: str = _generate(
                prompt=item.question,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
            )
            adapter_outputs[j].append(out)
            ans: str = _extract_answer_section(text=out)
            adapter_preds[j].append(_extract_last_digit_run(text=ans))

    _write_answers_json(
        answers_path=answers_path,
        items=items,
        base_out=base_outputs,
        adapter_outs=adapter_outputs,
        adapter_names=adapter_names,
    )

    _print_answer_table(
        rows=rows,
        correct=correct,
        base_pred=base_pred,
        adapter_preds=adapter_preds,
        adapter_names=adapter_names,
    )
    print(f"\nWrote paired answers to: {answers_path.expanduser().resolve()}")
    print("Done: base vs adapters evaluation completed.")


def _parse_optional_path_from_argv(
    *, argv: list[str]
) -> tuple[list[Path] | None, Path | None, Path | None]:
    """Parse optional adapters/questions/answers paths from argv.

    Usage:
        python check_sft.py [adapter_path_1 ... adapter_path_n] [--questions Q]
            [--answers A]

    Notes:
        All positional args are treated as adapter paths, except when the
        flags --questions/--answers are used.

    Args:
        argv: sys.argv list.

    Returns:
        (adapter_paths, questions_path, answers_path) where each can be None.

    Raises:
        FileNotFoundError: If a provided adapter/questions path does not exist.
        ValueError: If a flag is missing its value.
    """
    if len(argv) <= 1:
        return (None, None, None)

    adapter_paths: list[Path] = []
    questions: Path | None = None
    answers: Path | None = None

    i: int = 1
    while i < len(argv):
        arg: str = argv[i]

        if arg == "--questions":
            if i + 1 >= len(argv):
                raise ValueError("--questions requires a path value.")
            questions = Path(argv[i + 1]).expanduser().resolve()
            if not questions.exists():
                raise FileNotFoundError(f"Questions path not found: {questions}")
            i += 2
            continue

        if arg == "--answers":
            if i + 1 >= len(argv):
                raise ValueError("--answers requires a path value.")
            answers = Path(argv[i + 1]).expanduser().resolve()
            i += 2
            continue

        p: Path = Path(arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Adapter path not found: {p}")
        adapter_paths.append(p)
        i += 1

    if len(adapter_paths) == 0:
        return (None, questions, answers)

    return (adapter_paths, questions, answers)


if __name__ == "__main__":
    adapters, questions, answers = _parse_optional_path_from_argv(argv=sys.argv)

    if adapters is None:
        adapters = [
            # Path("weights/sft_lora/best-checkpoint"),
            Path("weights/sft_lora/checkpoint-800"),
            # Path("weights/final_rlm_lora/best-checkpoint-600"),
        ]
    if questions is None:
        questions = Path("./QA/questions.json")
    if answers is None:
        answers = Path("./QA/answers.json")

    main(adapter_paths=adapters, questions_path=questions, answers_path=answers)
