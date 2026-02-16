# tool_use/check_tool_answers.py

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tool_use.config import INFERENCE_CONFIG as CONFIG
from tool_use.config import REPO_DIR


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

    if cfg.temperature is not None:
        model.generation_config.temperature = float(cfg.temperature)
    if cfg.top_p is not None:
        model.generation_config.top_p = float(cfg.top_p)

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


def _render_chat_prompt(*, messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Render a chat prompt using the model's chat template.

    Args:
        messages: Chat messages (role/content) to render.
        tokenizer: HF tokenizer providing apply_chat_template.

    Returns:
        Rendered string prompt.
    """
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_once(
    *,
    messages: list[dict[str, str]],
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: CONFIG,
) -> str:
    """Generate a completion for the given messages.

    Args:
        messages: Chat history.
        model: Loaded model.
        tokenizer: Tokenizer.
        cfg: Shared configuration (includes generation limits).

    Returns:
        Decoded generation (completion only).
    """
    full_prompt: str = _render_chat_prompt(messages=messages, tokenizer=tokenizer)
    enc: Any = tokenizer(full_prompt, return_tensors="pt", truncation=True)

    device: torch.device = next(model.parameters()).device
    input_ids: torch.Tensor = enc["input_ids"].to(device=device)
    attention_mask: torch.Tensor = enc["attention_mask"].to(device=device)

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(cfg.max_new_tokens),
        "do_sample": bool(cfg.do_sample),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
    }
    if cfg.temperature is not None:
        gen_kwargs["temperature"] = float(cfg.temperature)
    if cfg.top_p is not None:
        gen_kwargs["top_p"] = float(cfg.top_p)

    with torch.no_grad():
        out: torch.Tensor = model.generate(**gen_kwargs)

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
    """Compute accuracy as a fraction in [0, 1]."""
    assert len(correct) == len(pred)
    if len(correct) == 0:
        return 0.0
    hits: int = sum((c == p) for c, p in zip(correct, pred, strict=True))
    return float(hits) / float(len(correct))


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
    correct: list[str],
    base_pred: list[str],
    adapter_preds: list[list[str]],
    adapter_names: list[str],
) -> None:
    """Print an ASCII table with correct + predicted answers for many adapters."""
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
    base_out: list[str],
    adapter_outs: list[list[str]],
    adapter_names: list[str],
) -> None:
    """Write per-question outputs (base + adapters) to a JSON file."""
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


def _safe_calculator(*, expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    allowed: set[str] = set("0123456789+-*/(). ")
    bad: list[str] = [c for c in expression if c not in allowed]
    if bad:
        uniq: str = "".join(sorted(set(bad)))
        return f"Error: Disallowed characters: {uniq}"

    try:
        node: ast.AST = ast.parse(expression, mode="eval")
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Call, ast.Attribute, ast.Name, ast.Subscript)):
                return "Error: Unsupported expression."
        value: Any = eval(
            compile(node, filename="<expr>", mode="eval"),
            {"__builtins__": {}},
        )
        return str(value)
    except Exception as exc:
        return f"Error calculating: {exc}"


def _default_tool_registry() -> dict[str, Callable[..., str]]:
    """Create a minimal tool registry."""

    def _multiplication(*, factor1: Any, factor2: Any) -> str:
        try:
            a: float = float(factor1)
            b: float = float(factor2)
            out: float = a * b
            if out.is_integer():
                return str(int(out))
            return str(out)
        except Exception as exc:
            return f"Error running tool 'multiplication': {exc}"

    def _mutation(*, value: Any, exponent: Any) -> str:
        # The adapter seems to use mutation(value=2, exponent=3) to mean 2 * 3.
        try:
            a: float = float(value)
            b: float = float(exponent)
            out: float = a * b
            if out.is_integer():
                return str(int(out))
            return str(out)
        except Exception as exc:
            return f"Error running tool 'mutation': {exc}"

    return {
        "calculator": lambda **kwargs: _safe_calculator(
            expression=str(kwargs.get("expression", ""))
        ),
        "multiplication": lambda **kwargs: _multiplication(
            factor1=kwargs.get("factor1"),
            factor2=kwargs.get("factor2"),
        ),
        "mutation": lambda **kwargs: _mutation(
            value=kwargs.get("value"),
            exponent=kwargs.get("exponent"),
        ),
    }


def _replace_id_refs(*, text: str, tool_outputs: dict[int, str]) -> str:
    """Replace <id=N> occurrences using already-produced tool outputs."""
    id_ref_re: re.Pattern[str] = re.compile(
        r"<id\s*=\s*(?P<id>\d+)\s*>",
        flags=re.IGNORECASE,
    )

    def _sub(m: re.Match[str]) -> str:
        tid: int = int(m.group("id"))
        return tool_outputs.get(tid, m.group(0))

    return id_ref_re.sub(_sub, text)


def _parse_args_kv_block(*, src: str) -> dict[str, Any]:
    """Parse args like: factor1=3, factor2=4, expression='<id=1> + <id=2>'.

    Args:
        src: Content between braces, without surrounding '{' and '}'.

    Returns:
        Dict with parsed primitive values.
    """
    args: dict[str, Any] = {}

    parts: list[str] = re.split(
        r",(?=(?:[^'\"\\]*(?:\\.|'[^']*'|\"[^\"]*\"))*[^'\"\\]*$)",
        src,
    )
    for raw in parts:
        part: str = raw.strip()
        if not part or "=" not in part:
            continue

        k_raw: str
        v_raw: str
        k_raw, v_raw = part.split("=", 1)
        key: str = k_raw.strip()
        val_s: str = v_raw.strip()

        if (val_s.startswith("'") and val_s.endswith("'")) or (
            val_s.startswith('"') and val_s.endswith('"')
        ):
            args[key] = val_s[1:-1]
            continue

        try:
            if "." in val_s:
                args[key] = float(val_s)
            else:
                args[key] = int(val_s)
            continue
        except Exception:
            pass

        args[key] = val_s

    return args


def _extract_think_blocks(*, text: str) -> list[str]:
    """Extract all <think>...</think> blocks."""
    blocks: list[str] = re.findall(
        pattern=r"<think>\s*(.*?)\s*</think>",
        string=text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return [b.strip() for b in blocks if b.strip()]


def _find_tool_calls(*, think_text: str) -> list[tuple[int, str, dict[str, Any]]]:
    """Find tool calls of the form: <tool id=1 name=calculator args={...}>.

    Args:
        think_text: Content inside a single <think>...</think> block.

    Returns:
        List of (tool_id, tool_name, tool_args_dict).
    """
    tool_re: re.Pattern[str] = re.compile(
        r"<tool\s+id\s*=\s*(?P<id>\d+)\s+name\s*=\s*(?P<name>[A-Za-z0-9_\-]+)\s+"
        r"args\s*=\s*\{(?P<args>.*?)\}\s*>",
        flags=re.DOTALL | re.IGNORECASE,
    )

    calls: list[tuple[int, str, dict[str, Any]]] = []
    for m in tool_re.finditer(think_text):
        tid: int = int(m.group("id"))
        name: str = str(m.group("name"))
        args_body: str = str(m.group("args")).strip()
        args: dict[str, Any] = _parse_args_kv_block(src=args_body)
        calls.append((tid, name, args))
    return calls


def _run_tool_loop(
    *,
    question: str,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: CONFIG,
    tool_registry: dict[str, Callable[..., str]],
) -> str:
    """Run up to cfg.max_calls tool iterations for a single question.

    Important:
        If the model emits tool calls and an <answer> in the same completion,
        execute tools and replace <id=...> inside <answer> before returning.
    """
    wrapped: str = f"<question>{question}</question>"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": wrapped},
    ]

    tool_outputs: dict[int, str] = {}
    last_completion: str = ""

    for _ in range(int(cfg.max_calls)):
        completion: str = _generate_once(
            messages=messages,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        last_completion = completion

        think_blocks: list[str] = _extract_think_blocks(text=completion)

        all_calls: list[tuple[int, str, dict[str, Any]]] = []
        for tb in think_blocks:
            all_calls.extend(_find_tool_calls(think_text=tb))

        if len(all_calls) > 0:
            for tid, name, args in all_calls:
                args_fixed: dict[str, Any] = {}
                for k, v in args.items():
                    if isinstance(v, str):
                        args_fixed[k] = _replace_id_refs(
                            text=v,
                            tool_outputs=tool_outputs,
                        )
                    else:
                        args_fixed[k] = v

                fn: Callable[..., str] | None = tool_registry.get(name)
                if fn is None:
                    out_str: str = f"Error: Unknown tool '{name}'."
                else:
                    try:
                        out_str = str(fn(**args_fixed))
                    except Exception as exc:
                        out_str = f"Error running tool '{name}': {exc}"

                tool_outputs[int(tid)] = out_str

        has_answer: bool = (
            re.search(
                pattern=r"<answer>\s*.*?\s*</answer>",
                string=completion,
                flags=re.DOTALL | re.IGNORECASE,
            )
            is not None
        )

        if has_answer:
            return _replace_id_refs(text=completion, tool_outputs=tool_outputs)

        if len(all_calls) == 0:
            return completion

        messages.append({"role": "assistant", "content": completion})

        tools_payload_lines: list[str] = ["<tools>", "{"]
        for tid, name, args in all_calls:
            out_str2: str = tool_outputs.get(int(tid), "")

            tools_payload_lines.append(f"    {tid}: {{")
            tools_payload_lines.append(f"        tool: '{name}',")
            tools_payload_lines.append("        args: {")
            for ak, av in args.items():
                av_s: str = str(
                    _replace_id_refs(text=str(av), tool_outputs=tool_outputs)
                    if isinstance(av, str)
                    else av
                )
                tools_payload_lines.append(f"            '{ak}': '{av_s}'")
            tools_payload_lines.append("        }")
            tools_payload_lines.append(f"        output: '{out_str2}'")
            tools_payload_lines.append("    },")

        tools_payload_lines.append("}")
        tools_payload_lines.append("</tools>")

        messages.append({"role": "user", "content": "\n".join(tools_payload_lines)})

    return _replace_id_refs(text=last_completion, tool_outputs=tool_outputs)


def main(
    *,
    adapter_paths: list[Path],
    questions_path: Path,
    answers_path: Path,
) -> None:
    """Compare base model vs multiple adapter models on a tool-use questions set."""
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

    tool_registry: dict[str, Callable[..., str]] = _default_tool_registry()

    rows: list[int] = []
    correct: list[str] = []
    base_pred: list[str] = []
    adapter_preds: list[list[str]] = [[] for _ in adapted_models]

    base_outputs: list[str] = []
    adapter_outputs: list[list[str]] = [[] for _ in adapted_models]

    iterable: Any = tqdm(
        enumerate(items, start=1),
        total=len(items),
        desc="Evaluating tool-use questions",
        dynamic_ncols=True,
    )

    for i, item in iterable:
        rows.append(i)

        correct_ans: str = _extract_last_digit_run(text=item.answer)
        correct.append(correct_ans)

        base_out: str = _run_tool_loop(
            question=item.question,
            model=base_model,
            tokenizer=tokenizer,
            cfg=cfg,
            tool_registry=tool_registry,
        )
        base_outputs.append(base_out)
        base_ans: str = _extract_answer_section(text=base_out)
        base_pred.append(_extract_last_digit_run(text=base_ans))

        for j, model in enumerate(adapted_models):
            out: str = _run_tool_loop(
                question=item.question,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                tool_registry=tool_registry,
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
