# tool_use/tool_inference.py

from __future__ import annotations

import re
from typing import Any

import torch

from tool_use.tool_handler import (
    ensure_response_contains_answer,
    parse_and_execute_tool_call,
)


def _render_chat_prompt(*, messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Render a chat prompt with the model chat template.

    Args:
        messages: List of chat messages with {role, content}.
        tokenizer: HF tokenizer.

    Returns:
        Rendered prompt string.
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
    cfg: Any,
) -> str:
    """Generate one assistant completion for the given messages.

    Args:
        messages: Chat messages.
        model: Loaded model.
        tokenizer: Tokenizer.
        cfg: Inference config (must expose generation fields).

    Returns:
        Decoded assistant text (no special tokens).
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
    if getattr(cfg, "temperature", None) is not None:
        gen_kwargs["temperature"] = float(cfg.temperature)
    if getattr(cfg, "top_p", None) is not None:
        gen_kwargs["top_p"] = float(cfg.top_p)

    with torch.no_grad():
        out: torch.Tensor = model.generate(**gen_kwargs)

    prompt_len: int = int(input_ids.shape[1])
    gen_ids: torch.Tensor = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _split_into_think_steps(*, text: str) -> list[str]:
    """Split transcript into steps, each from <think> to next <think> or end.

    Args:
        text: Full transcript string.

    Returns:
        List of step contents. If no <think> is present, returns [text].
    """
    starts: list[int] = [
        m.start() for m in re.finditer(r"<think>", text, re.IGNORECASE)
    ]
    if not starts:
        t: str = text.strip()
        return [t] if t else []

    chunks: list[str] = []
    for idx, s in enumerate(starts):
        e: int = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        chunk: str = text[s:e].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def run_tool_use_inference(
    *,
    question: str,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: Any,
    tool_dict: dict[str, Any],
) -> tuple[str, str, list[str]]:
    """Run tool-loop inference for a single question.

    Performs up to cfg.max_calls iterations. On errors, continues (tool_handler
    returns <error> blocks) until a valid answer is produced or the iteration
    budget is exhausted.

    Args:
        question: Raw user question string.
        model: Loaded model.
        tokenizer: Tokenizer.
        cfg: Tool inference config (must include system_prompt, max_calls, etc).
        tool_dict: Tool registry.

    Returns:
        Tuple:
          - full_output_transcript: Concatenated raw outputs for all iterations.
          - parsed_answer_only_final: Final validated <answer>...</answer>.
          - step_contents: Split contents from each <think> to next <think> or end.
    """
    wrapped_question: str = f"<question>{question}</question>"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(cfg.system_prompt)},
        {"role": "user", "content": wrapped_question},
    ]

    used_tool_ids: set[int] = set()
    global_outputs: dict[int, str] = {}

    transcript_raw: list[str] = []
    last_parsed_answer_only: str = "<answer>null</answer>"

    for _ in range(int(cfg.max_calls)):
        assistant_out: str = _generate_once(
            messages=messages,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
        )

        (
            should_continue,
            prompt_appendix,
            raw_full,
            _formatted_full,
            parsed_answer_only,
        ) = parse_and_execute_tool_call(
            model_output=assistant_out,
            tool_dict=tool_dict,
            max_calls=int(cfg.max_calls),
            used_tool_ids=used_tool_ids,
            global_outputs=global_outputs,
        )

        transcript_raw.append(raw_full)
        last_parsed_answer_only = parsed_answer_only

        if not should_continue:
            full_raw: str = "\n".join(transcript_raw)
            full_raw = ensure_response_contains_answer(full_prompt=full_raw)
            last_parsed_answer_only = ensure_response_contains_answer(
                full_prompt=last_parsed_answer_only
            )
            step_contents: list[str] = _split_into_think_steps(text=full_raw)
            return full_raw, last_parsed_answer_only, step_contents

        messages.append({"role": "assistant", "content": assistant_out})
        messages.append({"role": "user", "content": prompt_appendix})

    full_raw2: str = "\n".join(transcript_raw)
    full_raw2 = ensure_response_contains_answer(full_prompt=full_raw2)
    last_parsed_answer_only = ensure_response_contains_answer(
        full_prompt=last_parsed_answer_only
    )
    step_contents2: list[str] = _split_into_think_steps(text=full_raw2)
    return full_raw2, last_parsed_answer_only, step_contents2
