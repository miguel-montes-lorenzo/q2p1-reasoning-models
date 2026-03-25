# tool_use/langchain/tool_inference.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from tool_use.langchain.config import (
    REACT_FORCE_ANSWER_PROMPT,
    REACT_NUDGE_PROMPT,
    REACT_STALE_PROMPT,
)
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
    formatted_references: bool = False,
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
        formatted_references: If True, the returned full transcript contains
            formatted <answer> blocks with @ID references replaced by tool outputs.
            If False, returns the raw transcript (may include unresolved @ID).

    Returns:
        Tuple:
          - full_output_transcript: Concatenated raw/pretty outputs for all iterations.
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

    transcript: list[str] = []
    last_parsed_answer_only: str = "<answer>null</answer>"

    max_calls: int = int(cfg.max_calls)
    for it in range(max_calls):
        assistant_out: str = chat.invoke(messages=messages)

        (
            should_continue,
            prompt_appendix,
            raw_full,
            formatted_full,
            parsed_answer_only,
        ) = parse_and_execute_tool_call(
            model_output=assistant_out,
            tools=tools,
            max_calls=max_calls,
            used_tool_ids=used_tool_ids,
            global_outputs=global_outputs,
            is_last_iteration=(it == (max_calls - 1)),
        )

        transcript.append(formatted_full if formatted_references else raw_full)
        last_parsed_answer_only = parsed_answer_only

        if not should_continue:
            full_out: str = "\n".join(transcript)
            full_out = ensure_response_contains_answer(full_prompt=full_out)
            last_parsed_answer_only = ensure_response_contains_answer(
                full_prompt=last_parsed_answer_only
            )
            step_contents: list[str] = _split_into_think_steps(text=full_out)
            return full_out, last_parsed_answer_only, step_contents

        messages.append(AIMessage(content=assistant_out))
        messages.append(HumanMessage(content=prompt_appendix))

    full_out2: str = "\n".join(transcript)
    full_out2 = ensure_response_contains_answer(full_prompt=full_out2)
    last_parsed_answer_only = ensure_response_contains_answer(
        full_prompt=last_parsed_answer_only
    )
    step_contents2: list[str] = _split_into_think_steps(text=full_out2)
    return full_out2, last_parsed_answer_only, step_contents2


# ---------------------------------------------------------------------------
# ReAct-specific inference loop (Phase 4)
# ---------------------------------------------------------------------------

_COMPRESS_MSG_THRESHOLD: int = 12


def _maybe_compress_messages(
    *,
    messages: list[BaseMessage],
    global_outputs: dict[str, str],
) -> list[BaseMessage]:
    """Compress older tool-result messages when the conversation is too long.

    Keeps the system message (index 0), the user question (index 1), and the
    last 4 conversation messages intact.  Everything in between is replaced by
    a single summary HumanMessage that lists tool IDs and their outputs.

    Args:
        messages: Current conversation messages.
        global_outputs: All tool outputs collected so far.

    Returns:
        Possibly shortened message list.
    """
    if len(messages) <= _COMPRESS_MSG_THRESHOLD:
        return messages

    # Keep: system(0) + question(1) + ... + last 4
    keep_head: int = 2
    keep_tail: int = 4
    middle: list[BaseMessage] = messages[keep_head:-keep_tail]
    if not middle:
        return messages

    summary_lines: list[str] = [
        "[SYSTEM] Conversation compressed. Tool outputs collected so far:"
    ]
    for tid, output in global_outputs.items():
        truncated: str = output[:300] + "..." if len(output) > 300 else output
        summary_lines.append(f"  @{tid} = {truncated}")

    summary_msg: HumanMessage = HumanMessage(content="\n".join(summary_lines))
    return [*messages[:keep_head], summary_msg, *messages[-keep_tail:]]


_ERROR_HINT_EXAMPLE: str = (
    "Reminder — correct tool-call syntax examples:\n"
    '  @calculator(expression="2+2")->1\n'
    '  @food_data_central_search(query="banana raw", page_size="1")->2\n'
    '  @knowledge_base_search(query="muffin recipe")->3\n'
    "Each call needs: @TOOL_NAME(arg=\"value\")->UNIQUE_ID"
)


def run_react_inference(
    *,
    question: str,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: Any,
    tools: list[BaseTool],
) -> list[dict[str, Any]]:
    """Run the ReAct tool-loop with stale detection, forced finalization, and compression.

    Unlike ``run_tool_use_inference`` (Phase 2), this function is designed for
    multi-step queries that may require many iterations (recipe + nutrition +
    calculator chains).

    Key features:
      - **Stale detection**: after 3 consecutive errors or 3 consecutive
        iterations with no new tool call, injects ``REACT_STALE_PROMPT``.
      - **Forced finalization**: at ``max_calls - 2`` injects a nudge; at
        ``max_calls - 1`` injects a force-answer prompt.
      - **Message compression**: when the message list exceeds a threshold,
        older tool-result messages are collapsed into a summary.
      - **Error escalation**: after 2+ consecutive errors, appends concrete
        tool-call format examples.
      - **Fallback answer synthesis**: if the model never produces a valid
        ``<answer>``, one is synthesized from ``global_outputs``.

    Args:
        question: Raw user question string.
        model: Loaded HF model.
        tokenizer: HF tokenizer.
        cfg: REACT_INFERENCE_CONFIG (system_prompt, max_calls, generation params).
        tools: LangChain tool list.

    Returns:
        Structured trace — ``list[dict]`` where each dict has keys
        ``{"step", "type", "content"}``.  ``type`` is one of
        ``"model_output"``, ``"observation"``, or ``"final_answer"``.
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

    trace: list[dict[str, Any]] = []
    last_parsed_answer_only: str = "<answer>null</answer>"

    max_calls: int = int(cfg.max_calls)
    consecutive_errors: int = 0
    consecutive_no_new_tool: int = 0
    prev_tool_count: int = 0
    step_idx: int = 0

    for it in range(max_calls):
        # --- Forced finalization prompts ---
        if it == max_calls - 2:
            messages.append(HumanMessage(content=REACT_NUDGE_PROMPT))
        elif it == max_calls - 1:
            messages.append(HumanMessage(content=REACT_FORCE_ANSWER_PROMPT))

        # --- Stale detection injection ---
        stale: bool = consecutive_errors >= 3 or consecutive_no_new_tool >= 3
        if stale and it < max_calls - 1:
            messages.append(HumanMessage(content=REACT_STALE_PROMPT))

        # --- Message compression ---
        messages = _maybe_compress_messages(
            messages=messages,
            global_outputs=global_outputs,
        )

        # --- Generate ---
        assistant_out: str = chat.invoke(messages=messages)

        trace.append({
            "step": step_idx,
            "type": "model_output",
            "content": assistant_out,
        })
        step_idx += 1

        # --- Parse & execute ---
        (
            should_continue,
            prompt_appendix,
            raw_full,
            formatted_full,
            parsed_answer_only,
        ) = parse_and_execute_tool_call(
            model_output=assistant_out,
            tools=tools,
            max_calls=max_calls,
            used_tool_ids=used_tool_ids,
            global_outputs=global_outputs,
            is_last_iteration=(it == max_calls - 1),
        )

        last_parsed_answer_only = parsed_answer_only

        # Track observation
        trace.append({
            "step": step_idx,
            "type": "observation",
            "content": formatted_full,
        })
        step_idx += 1

        # --- Track staleness ---
        is_error: bool = "<error>" in raw_full.lower()
        new_tool_count: int = len(used_tool_ids)
        made_new_tool: bool = new_tool_count > prev_tool_count
        prev_tool_count = new_tool_count

        if is_error:
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        if not made_new_tool and not is_error:
            consecutive_no_new_tool += 1
        else:
            consecutive_no_new_tool = 0

        # --- Done? ---
        if not should_continue:
            final_answer: str = ensure_response_contains_answer(
                full_prompt=last_parsed_answer_only
            )
            trace.append({
                "step": step_idx,
                "type": "final_answer",
                "content": final_answer,
            })
            return trace

        # --- Build next prompt ---
        messages.append(AIMessage(content=assistant_out))

        # Error escalation: after 2+ consecutive errors, add format hint
        if consecutive_errors >= 2:
            prompt_appendix = f"{prompt_appendix}\n{_ERROR_HINT_EXAMPLE}"

        messages.append(HumanMessage(content=prompt_appendix))

    # --- Exhausted max_calls: fallback answer synthesis ---
    final_answer_str: str = _synthesize_fallback_answer(
        last_parsed=last_parsed_answer_only,
        global_outputs=global_outputs,
        question=question,
    )
    trace.append({
        "step": step_idx,
        "type": "final_answer",
        "content": final_answer_str,
    })
    return trace


def _synthesize_fallback_answer(
    *,
    last_parsed: str,
    global_outputs: dict[str, str],
    question: str,
) -> str:
    """Synthesize an answer from collected tool outputs when the model failed to produce one.

    If ``last_parsed`` already contains a non-null answer, return it as-is.
    Otherwise build a best-effort answer from ``global_outputs``.

    Args:
        last_parsed: Last parsed answer string (may be ``<answer>null</answer>``).
        global_outputs: All tool outputs collected during the run.
        question: Original user question (for context in the synthesized answer).

    Returns:
        An ``<answer>...</answer>`` string.
    """
    # Check if we already have a real answer
    m: re.Match[str] | None = re.search(
        r"<answer>(.*?)</answer>", last_parsed, flags=re.DOTALL | re.IGNORECASE
    )
    if m and m.group(1).strip().lower() not in ("null", ""):
        return last_parsed

    if not global_outputs:
        return "<answer>null</answer>"

    # Build a summary from all tool outputs
    parts: list[str] = []
    for tid, output in global_outputs.items():
        truncated: str = output[:500] if len(output) > 500 else output
        parts.append(f"[Tool @{tid}]: {truncated}")

    synthesized: str = (
        f"Based on the information gathered:\n" + "\n".join(parts)
    )
    return f"<answer>{synthesized}</answer>"
