# tool_use/check_tool_answers.py

from __future__ import annotations

import gc
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tool_use.config import INFERENCE_CONFIG as CONFIG
from tool_use.config import REPO_DIR
from tool_use.tool_handler import insert_tool_desciptions_in_system_propt
from tool_use.tool_inference import run_tool_use_inference
from tool_use.tools import TOOL_DICT


@dataclass(frozen=True)
class QAItem:
    """Single QA item loaded from the questions JSON.

    Args:
        question: User question string.
        answer: Ground-truth answer (raw).
    """

    question: str
    answer: str


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    """Create a BitsAndBytes quantization config.

    Args:
        use_4bit: Whether to enable 4-bit NF4 quantization.

    Returns:
        BitsAndBytesConfig if enabled, else None.
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
    """Align model and generation configs with tokenizer special tokens."""
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.do_sample = bool(cfg.do_sample)
    if cfg.temperature is not None:
        model.generation_config.temperature = float(cfg.temperature)
    if cfg.top_p is not None:
        model.generation_config.top_p = float(cfg.top_p)
    model.generation_config.top_k = None


def _load_tokenizer(*, cfg: CONFIG) -> Any:
    """Load tokenizer."""
    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(*, cfg: CONFIG, tokenizer: Any) -> torch.nn.Module:
    """Load base model for inference."""
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
    """Resolve an adapter directory from a user-provided path."""
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
    *, cfg: CONFIG, tokenizer: Any, adapter_dir: Path
) -> torch.nn.Module:
    """Load base model and attach LoRA adapter."""
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


def _load_questions(*, questions_path: Path) -> list[QAItem]:
    """Load questions from a JSON file."""
    p: Path = questions_path.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {p}")

    raw_text: str = p.read_text(encoding="utf-8")
    data: Any = json.loads(s=raw_text)

    if not isinstance(data, list):
        raise ValueError("Questions JSON must be a list of objects.")

    items: list[QAItem] = []
    for idx, item in enumerate(iterable=data, start=1):
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
    base_full: list[str],
    base_parsed_answer: list[str],
    adapter_full: list[list[str]],
    adapter_parsed_answer: list[list[str]],
    adapter_names: list[str],
) -> None:
    """Write outputs to JSON with the required contract."""
    assert len(items) == len(base_full) == len(base_parsed_answer)
    assert len(adapter_full) == len(adapter_names) == len(adapter_parsed_answer)
    for outs, pars in zip(adapter_full, adapter_parsed_answer, strict=True):
        assert len(outs) == len(items)
        assert len(pars) == len(items)

    out_data: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        row: dict[str, Any] = {
            "question": item.question,
            "base_output": base_full[i],
            "parsed_base_output": base_parsed_answer[i],
        }
        for name, outs, pars in zip(
            adapter_names, adapter_full, adapter_parsed_answer, strict=True
        ):
            key_out: str = f"{name}_output"
            row[key_out] = outs[i]
            row[f"parsed_{key_out}"] = pars[i]
        out_data.append(row)

    p: Path = answers_path.expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_answer_inner_text(*, text: str) -> str | None:
    """Extract inner text inside <answer>...</answer>."""
    m: re.Match[str] | None = re.search(
        pattern=r"<answer>\s*(.*?)\s*</answer>",
        string=text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m is None:
        return None
    return m.group(1).strip()


def _clean_math_expression(*, text: str) -> str:
    """Keep only math characters and return an eval-ready expression."""
    allowed: set[str] = set("0123456789+-*/(). ")
    normalized: str = text.replace(",", ".")
    cleaned: str = "".join([c for c in normalized if c in allowed]).strip()
    cleaned = re.sub(pattern=r"\s+", repl=" ", string=cleaned)
    return cleaned


def _safe_eval_math(*, expr: str) -> float | None:
    """Evaluate a cleaned math expression with eval, treating warnings as failure."""
    expr_s: str = expr.strip()
    if not expr_s:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=SyntaxWarning)
            value_any: Any = eval(expr_s, globals={"__builtins__": {}}, locals={})
        value: float = float(value_any)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _parse_number_from_text(*, text: str) -> float | None:
    """Parse numeric value from raw text."""
    cleaned: str = _clean_math_expression(text=text)
    value: float | None = _safe_eval_math(expr=cleaned)
    if value is not None:
        return value

    m: re.Match[str] | None = re.search(
        pattern=r"[-+]?\d+(?:\.\d+)?",
        string=text.replace(",", "."),
    )
    if m is None:
        return None

    try:
        v: float = float(m.group(0))
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _extract_validated_numeric_answer(*, parsed_answer_only: str) -> float | None:
    """Extract numeric value from parsed <answer>...</answer> only."""
    inner: str | None = _extract_answer_inner_text(text=parsed_answer_only)
    if inner is None:
        return None
    return _parse_number_from_text(text=inner)


def _answers_close(*, pred: float | None, gold: float | None, eps: float) -> bool:
    """Check numeric closeness with epsilon."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= eps


def _make_unique_names(*, base_names: list[str]) -> list[str]:
    """Make adapter names unique by appending '_k' to duplicates."""
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
    gold_vals: list[float | None],
    base_vals: list[float | None],
    adapter_vals: list[list[float | None]],
    adapter_names: list[str],
    eps: float,
) -> None:
    """Print an ASCII table with gold + predicted numeric values."""
    assert len(rows) == len(gold_vals) == len(base_vals)
    assert len(adapter_vals) == len(adapter_names)
    for preds in adapter_vals:
        assert len(preds) == len(rows)

    def _fmt(v: float | None) -> str:
        if v is None:
            return "?"
        if float(v).is_integer():
            return str(int(v))
        return f"{v:.6g}"

    headers: list[str] = ["#", "correct", "base", *adapter_names]

    columns: list[list[str]] = []
    columns.append([str(r) for r in rows])
    columns.append([_fmt(v) for v in gold_vals])
    columns.append([_fmt(v) for v in base_vals])
    for preds in adapter_vals:
        columns.append([_fmt(v) for v in preds])

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

    base_hits: int = sum(
        _answers_close(pred=p, gold=g, eps=eps) for g, p in zip(gold_vals, base_vals)
    )
    base_acc: float = float(base_hits) / float(len(rows)) if rows else 0.0

    adapter_accs: list[float] = []
    for preds in adapter_vals:
        hits: int = sum(
            _answers_close(pred=p, gold=g, eps=eps) for g, p in zip(gold_vals, preds)
        )
        adapter_accs.append(float(hits) / float(len(rows)) if rows else 0.0)

    acc_cells: list[str] = ["acc", "", f"{base_acc:>4.2f}"]
    acc_cells.extend(f"{a:>4.2f}" for a in adapter_accs)

    print(sep)
    print(_fmt_row(items=acc_cells))


def _free_model(*, model: torch.nn.Module | None) -> None:
    """Best-effort GPU/CPU memory cleanup after finishing a model."""
    if model is not None:
        try:
            del model
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main(
    *,
    adapter_paths: list[Path],
    questions_path: Path,
    answers_path: Path,
    eps: float = 1e-3,
) -> None:
    """Compare base model vs adapter models on a tool-use question set.

    This version evaluates sequentially per model (all questions for base, then
    all questions for each adapter), and shows one progress bar per model.
    """
    cfg_base: CONFIG = CONFIG()

    descriptions: dict[str, str] = {
        tool_name: str(tool_meta["description"])
        for tool_name, tool_meta in TOOL_DICT.items()
    }
    tool_augmented_system_prompt: str = insert_tool_desciptions_in_system_propt(
        descriptions=descriptions
    )
    cfg: CONFIG = replace(cfg_base, system_prompt=tool_augmented_system_prompt)

    tokenizer: Any = _load_tokenizer(cfg=cfg)

    items: list[QAItem] = _load_questions(questions_path=questions_path)
    rows: list[int] = list(range(1, len(items) + 1))
    gold_vals: list[float | None] = [
        _parse_number_from_text(text=i.answer) for i in items
    ]

    # Resolve adapter dirs/names now (but load models one-by-one later).
    resolved_dirs: list[Path] = [_resolve_adapter_dir(path=p) for p in adapter_paths]
    raw_names: list[str] = [d.name for d in resolved_dirs]
    adapter_names: list[str] = _make_unique_names(base_names=raw_names)

    # Outputs storage
    base_full_outputs: list[str] = []
    base_parsed_answers: list[str] = []
    base_vals: list[float | None] = []

    adapter_full_outputs: list[list[str]] = [[] for _ in resolved_dirs]
    adapter_parsed_answers: list[list[str]] = [[] for _ in resolved_dirs]
    adapter_vals: list[list[float | None]] = [[] for _ in resolved_dirs]

    # ---- Evaluate BASE (one progress bar) ----
    base_model: torch.nn.Module | None = _load_base_model(cfg=cfg, tokenizer=tokenizer)
    try:
        pbar_base: Any = tqdm(
            enumerate(items, start=1),
            total=len(items),
            desc="Evaluating model: base",
            dynamic_ncols=True,
        )
        for _idx, item in pbar_base:
            full_out: str
            parsed_ans: str
            _steps: list[str]
            full_out, parsed_ans, _steps = run_tool_use_inference(
                question=item.question,
                model=cast(torch.nn.Module, base_model),
                tokenizer=tokenizer,
                cfg=cfg,
                tool_dict=TOOL_DICT,
            )
            base_full_outputs.append(full_out)
            base_parsed_answers.append(parsed_ans)
            base_vals.append(
                _extract_validated_numeric_answer(parsed_answer_only=parsed_ans)
            )
    finally:
        _free_model(model=base_model)
        base_model = None

    # ---- Evaluate ADAPTERS sequentially (one progress bar per adapter) ----
    for j, (adapter_dir, adapter_name) in enumerate(
        zip(resolved_dirs, adapter_names, strict=True)
    ):
        adapted_model: torch.nn.Module | None = _load_adapted_model(
            cfg=cfg, tokenizer=tokenizer, adapter_dir=adapter_dir
        )
        try:
            pbar_ad: Any = tqdm(
                enumerate(items, start=1),
                total=len(items),
                desc=f"Evaluating model: {adapter_name}",
                dynamic_ncols=True,
            )
            for _idx, item in pbar_ad:
                full_out_j: str
                parsed_ans_j: str
                _steps_j: list[str]
                full_out_j, parsed_ans_j, _steps_j = run_tool_use_inference(
                    question=item.question,
                    model=cast(torch.nn.Module, adapted_model),
                    tokenizer=tokenizer,
                    cfg=cfg,
                    tool_dict=TOOL_DICT,
                )
                adapter_full_outputs[j].append(full_out_j)
                adapter_parsed_answers[j].append(parsed_ans_j)
                adapter_vals[j].append(
                    _extract_validated_numeric_answer(parsed_answer_only=parsed_ans_j)
                )
        finally:
            _free_model(model=adapted_model)
            adapted_model = None

    _write_answers_json(
        answers_path=answers_path,
        items=items,
        base_full=base_full_outputs,
        base_parsed_answer=base_parsed_answers,
        adapter_full=adapter_full_outputs,
        adapter_parsed_answer=adapter_parsed_answers,
        adapter_names=adapter_names,
    )

    _print_answer_table(
        rows=rows,
        gold_vals=gold_vals,
        base_vals=base_vals,
        adapter_vals=adapter_vals,
        adapter_names=adapter_names,
        eps=eps,
    )
    print(f"\nWrote paired answers to: {answers_path.expanduser().resolve()}")
    print("Done: base vs adapters tool-use evaluation completed.")


def _parse_optional_path_from_argv(
    *, argv: list[str]
) -> tuple[list[Path] | None, Path | None, Path | None]:
    """Parse optional adapters/questions/answers paths from argv."""
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
    adapters: list[Path] | None
    questions: Path | None
    answers: Path | None
    adapters, questions, answers = _parse_optional_path_from_argv(argv=sys.argv)

    if adapters is None:
        adapters = [
            REPO_DIR / "weights/tool_use/sft_lora/best_checkpoint",
        ]
    if questions is None:
        questions = REPO_DIR.joinpath("QA/questions.json")
    if answers is None:
        answers = REPO_DIR.joinpath("QA/answers/tool_use.json")

    main(adapter_paths=adapters, questions_path=questions, answers_path=answers)
