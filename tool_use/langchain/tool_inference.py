# tool_use/langchain/tool_inference.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from tool_use.langchain.tool_handler import (
    ensure_response_contains_answer,
    parse_and_execute_tool_call,
)


@dataclass(frozen=True)
class HFGenerationConfig:
    """Minimal generation config for HuggingFace model generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate per iteration.
        do_sample: Whether to sample.
        temperature: Optional sampling temperature.
        top_p: Optional nucleus sampling probability.
    """

    max_new_tokens: int
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None


def _get_gen_cfg(*, cfg: Any) -> HFGenerationConfig:
    """Derive a HFGenerationConfig from an inference config object.

    Supports both:
      - cfg.gen_cfg (already a HFGenerationConfig-like object)
      - "flat" cfg fields: max_new_tokens, do_sample, temperature, top_p

    Args:
        cfg: Inference config object.

    Returns:
        HFGenerationConfig built from cfg.
    """
    gen_cfg_any: Any = getattr(cfg, "gen_cfg", None)
    if gen_cfg_any is not None:
        return HFGenerationConfig(
            max_new_tokens=int(gen_cfg_any.max_new_tokens),
            do_sample=bool(gen_cfg_any.do_sample),
            temperature=(
                float(gen_cfg_any.temperature)
                if getattr(gen_cfg_any, "temperature", None) is not None
                else None
            ),
            top_p=(
                float(gen_cfg_any.top_p)
                if getattr(gen_cfg_any, "top_p", None) is not None
                else None
            ),
        )

    return HFGenerationConfig(
        max_new_tokens=int(cfg.max_new_tokens),
        do_sample=bool(cfg.do_sample),
        temperature=(
            float(cfg.temperature)
            if getattr(cfg, "temperature", None) is not None
            else None
        ),
        top_p=(float(cfg.top_p) if getattr(cfg, "top_p", None) is not None else None),
    )


class HFChatModel:
    """Tiny LangChain-like chat wrapper for HF chat models.

    This wrapper keeps your exact tokenizer.apply_chat_template workflow, but
    exposes an `invoke(messages)` method compatible with a LangChain-style
    message list.

    Args:
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer with apply_chat_template.
        gen_cfg: Generation configuration.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        gen_cfg: HFGenerationConfig,
    ) -> None:
        self._model: torch.nn.Module = model
        self._tokenizer: Any = tokenizer
        self._gen_cfg: HFGenerationConfig = gen_cfg

    def _render_chat_prompt(self, *, messages: list[BaseMessage]) -> str:
        """Render a chat prompt using the tokenizer chat template.

        Args:
            messages: LangChain-style messages.

        Returns:
            Rendered prompt string for the model.
        """
        convo: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role: str = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            convo.append({"role": role, "content": str(m.content)})

        return self._tokenizer.apply_chat_template(
            conversation=convo,
            tokenize=False,
            add_generation_prompt=True,
        )

    def invoke(self, *, messages: list[BaseMessage]) -> str:
        """Generate one assistant completion.

        Args:
            messages: Conversation messages.

        Returns:
            Decoded assistant text (no special tokens).
        """
        full_prompt: str = self._render_chat_prompt(messages=messages)
        enc: Any = self._tokenizer(full_prompt, return_tensors="pt", truncation=True)

        device: torch.device = next(self._model.parameters()).device
        input_ids: torch.Tensor = enc["input_ids"].to(device=device)
        attention_mask: torch.Tensor = enc["attention_mask"].to(device=device)

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(self._gen_cfg.max_new_tokens),
            "do_sample": bool(self._gen_cfg.do_sample),
            "pad_token_id": int(self._tokenizer.pad_token_id),
            "eos_token_id": int(self._tokenizer.eos_token_id),
        }
        if self._gen_cfg.temperature is not None:
            gen_kwargs["temperature"] = float(self._gen_cfg.temperature)
        if self._gen_cfg.top_p is not None:
            gen_kwargs["top_p"] = float(self._gen_cfg.top_p)

        with torch.no_grad():
            out: torch.Tensor = self._model.generate(**gen_kwargs)

        prompt_len: int = int(input_ids.shape[1])
        gen_ids: torch.Tensor = out[0, prompt_len:]
        return str(self._tokenizer.decode(gen_ids, skip_special_tokens=True))


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
    tools: list[BaseTool],
) -> tuple[str, str, list[str]]:
    """Run tool-loop inference for a single question using LangChain tools.

    This mimics your workflow:
      - User message wraps <question>...</question>
      - Assistant produces <think>...</think> with @tool(...)->id calls
      - We execute tools via LangChain tool objects, produce <tools> block
      - Loop continues until a valid <answer> is produced or max_calls is hit

    Args:
        question: Raw user question string.
        model: Loaded HF model.
        tokenizer: HF tokenizer.
        cfg: Tool-use inference config (must include system_prompt, max_calls,
            and generation fields like max_new_tokens/do_sample[/temperature/top_p]).
        tools: LangChain tool list.

    Returns:
        Tuple:
          - full_output_transcript: Concatenated raw outputs for all iterations.
          - parsed_answer_only_final: Final validated <answer>...</answer>.
          - step_contents: Split contents from each <think> to next <think> or end.
    """
    gen_cfg: HFGenerationConfig = _get_gen_cfg(cfg=cfg)
    chat: HFChatModel = HFChatModel(model=model, tokenizer=tokenizer, gen_cfg=gen_cfg)

    wrapped_question: str = f"<question>{question}</question>"
    messages: list[BaseMessage] = [
        SystemMessage(content=str(cfg.system_prompt)),
        HumanMessage(content=wrapped_question),
    ]

    used_tool_ids: set[str] = set()
    global_outputs: dict[str, str] = {}

    transcript_raw: list[str] = []
    last_parsed_answer_only: str = "<answer>null</answer>"

    max_calls: int = int(cfg.max_calls)
    for it in range(max_calls):
        assistant_out: str = chat.invoke(messages=messages)

        (
            should_continue,
            prompt_appendix,
            raw_full,
            _formatted_full,
            parsed_answer_only,
        ) = parse_and_execute_tool_call(
            model_output=assistant_out,
            tools=tools,
            max_calls=max_calls,
            used_tool_ids=used_tool_ids,
            global_outputs=global_outputs,
            is_last_iteration=(it == (max_calls - 1)),
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

        messages.append(AIMessage(content=assistant_out))
        messages.append(HumanMessage(content=prompt_appendix))

    full_raw2: str = "\n".join(transcript_raw)
    full_raw2 = ensure_response_contains_answer(full_prompt=full_raw2)
    last_parsed_answer_only = ensure_response_contains_answer(
        full_prompt=last_parsed_answer_only
    )
    step_contents2: list[str] = _split_into_think_steps(text=full_raw2)
    return full_raw2, last_parsed_answer_only, step_contents2
