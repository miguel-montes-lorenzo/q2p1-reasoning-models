import re
from collections.abc import Callable
from typing import Any

from rlm.config import SYSTEM_PROMPT, SYSTEM_PROMPT_END

from tool_use.config import TOOL_SYSTEM_PROMPT


def insert_tool_desciptions_in_system_propt(descriptions: dict[str, str]) -> str:
    tool_descriptions: str = "TOOL DESCRIPTIONS:"
    for tool_name, tool_description in descriptions.items():
        tool_descriptions = (
            f"{tool_descriptions}\n{tool_name}:\n{tool_description}\n---"
        )
    tool_descriptions = f"{tool_descriptions}\n\n"
    return f"{SYSTEM_PROMPT}{TOOL_SYSTEM_PROMPT}{tool_descriptions}{SYSTEM_PROMPT_END}"


def parse_and_execute_tool_call(
    model_output: str,
    tool_dict: dict[str, Any],
    max_calls: int = 3,
) -> tuple[bool, str]:
    """Parse and execute <tool ...> calls inside a <think>...</think> block.

    This implements the tool and <id=...> evaluation semantics described in the
    tool system prompt and controls the iteration loop.

    The returned boolean indicates whether the caller should invoke the model
    again with the newly constructed prompt segment.

    Args:
        model_output: Raw model output containing <think>...</think> and possibly
            embedded <tool ...> tags.
        tool_dict: Mapping of tool names to dicts containing a callable under
            key "function".
        max_calls: Maximum number of allowed tool-iteration calls.

    Returns:
        A tuple (should_continue, prompt_appendix).
        - should_continue is True if the caller must invoke the model again.
        - prompt_appendix is the exact text to append to the prompt (includes
          <think>...</think> and <tools>...</tools>) if tools were executed,
          otherwise an empty string.
    """

    def _extract_first_tag_block(text: str, tag: str) -> str | None:
        pattern: str = rf"<{tag}>(.*?)</{tag}>"
        match: re.Match[str] | None = re.search(
            pattern=pattern, string=text, flags=re.DOTALL
        )
        if match is None:
            return None
        return match.group(0)

    def _parse_tool_attributes(tag_text: str) -> tuple[int, str, str]:
        id_match: re.Match[str] | None = re.search(
            pattern=r"\bid\s*=\s*(\d+)\b", string=tag_text
        )
        name_match: re.Match[str] | None = re.search(
            pattern=r"\bname\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\b", string=tag_text
        )
        args_match: re.Match[str] | None = re.search(
            pattern=r"\bargs\s*=\s*\{(.*)\}\s*>$", string=tag_text, flags=re.DOTALL
        )

        if id_match is None or name_match is None or args_match is None:
            raise ValueError(f"Invalid <tool ...> tag attributes: {tag_text}")

        tool_id: int = int(id_match.group(1))
        tool_name: str = name_match.group(1)
        args_body: str = args_match.group(1)
        return tool_id, tool_name, args_body

    def _split_args_items(args_body: str) -> list[str]:
        items: list[str] = []
        current: list[str] = []
        in_quotes: bool = False
        escape: bool = False

        for ch in args_body:
            if escape:
                current.append(ch)
                escape = False
                continue

            if ch == "\\":
                current.append(ch)
                escape = True
                continue

            if ch == '"':
                current.append(ch)
                in_quotes = not in_quotes
                continue

            if ch == "," and not in_quotes:
                item: str = "".join(current).strip()
                if item:
                    items.append(item)
                current = []
                continue

            current.append(ch)

        tail: str = "".join(current).strip()
        if tail:
            items.append(tail)

        return items

    def _parse_args_dict(args_body: str) -> dict[str, str]:
        args: dict[str, str] = {}
        for item in _split_args_items(args_body=args_body):
            m: re.Match[str] | None = re.match(
                pattern=r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(.*)"\s*$',
                string=item,
                flags=re.DOTALL,
            )
            if m is None:
                raise ValueError(f"Invalid args item: {item}")
            key: str = m.group(1)
            value: str = m.group(2)
            args[key] = value
        return args

    def _resolve_id_refs(value: str, outputs: dict[int, str]) -> str:
        def repl(match: re.Match[str]) -> str:
            ref_id: int = int(match.group(1))
            if ref_id not in outputs:
                raise ValueError(
                    f"Invalid <id={ref_id}> reference: tool output not available."
                )
            return outputs[ref_id]

        return re.sub(pattern=r"<id=(\d+)>", repl=repl, string=value)

    def _get_tool_fn(name: str) -> Callable[..., str]:
        if name not in tool_dict:
            raise ValueError(f"Unknown tool name: {name}")
        fn: Any = tool_dict[name].get("function")
        if not callable(fn):
            raise ValueError(f"Tool '{name}' does not provide a callable 'function'.")
        return fn

    think_block: str | None = _extract_first_tag_block(text=model_output, tag="think")
    answer_block: str | None = _extract_first_tag_block(text=model_output, tag="answer")

    # If an answer already exists, stop iteration
    if answer_block is not None:
        return False, ""

    if think_block is None:
        return False, ""

    tool_tag_pattern: re.Pattern[str] = re.compile(pattern=r"<tool\b[^>]*>")
    tool_tags: list[str] = tool_tag_pattern.findall(string=think_block)

    # If no tools are present, stop iteration
    if not tool_tags:
        return False, ""

    # Enforce max_calls by counting previous <tools> blocks
    previous_calls: int = len(re.findall(pattern=r"<tools>", string=model_output))
    if previous_calls >= max_calls:
        return False, ""

    executed_ids: set[int] = set()
    outputs: dict[int, str] = {}
    tool_records: list[dict[str, Any]] = []

    for tool_tag in tool_tags:
        tool_id: int
        tool_name: str
        args_body: str
        tool_id, tool_name, args_body = _parse_tool_attributes(tag_text=tool_tag)

        if tool_id in executed_ids:
            raise ValueError(f"Repeated tool id inside the same <think>: {tool_id}")
        executed_ids.add(tool_id)

        raw_args: dict[str, str] = _parse_args_dict(args_body=args_body)
        resolved_args: dict[str, str] = {
            k: _resolve_id_refs(value=v, outputs=outputs) for k, v in raw_args.items()
        }

        tool_fn: Callable[..., str] = _get_tool_fn(name=tool_name)
        result: str = tool_fn(**resolved_args)

        outputs[tool_id] = result
        tool_records.append({
            "id": tool_id,
            "tool": tool_name,
            "args": raw_args,
            "output": result,
        })

    def _render_tools_block(records: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        lines.append("<tools>")
        lines.append("{")
        for rec in records:
            tool_id_int: int = int(rec["id"])
            tool_name_str: str = str(rec["tool"])
            args_dict_str: dict[str, str] = dict(rec["args"])
            output_str: str = str(rec["output"])

            lines.append(f"    {tool_id_int}: {{")
            lines.append(f'        tool: "{tool_name_str}",')
            lines.append("        args: {")
            for k, v in args_dict_str.items():
                escaped_v: str = v.replace('"', r"\"")
                lines.append(f'            "{k}": "{escaped_v}"')
            lines.append("        }")
            escaped_out: str = output_str.replace('"', r"\"")
            lines.append(f'        output: "{escaped_out}"')
            lines.append("    }")
        lines.append("}")
        lines.append("</tools>")
        return "\n".join(lines)

    tools_block: str = _render_tools_block(records=tool_records)
    prompt_appendix: str = f"{think_block}\n{tools_block}\n"

    return True, prompt_appendix


def ensure_response_contains_answer(full_prompt: str) -> str:
    """Ensure the final response contains an <answer>...</answer> tag.

    This function must be called after the last invocation of
    `parse_and_execute_tool_call`, i.e., when no further tool iterations
    should occur. If the response already contains an <answer> block,
    the prompt is returned unchanged. Otherwise, a blank <answer></answer>
    tag is appended to the end of the prompt.

    Args:
        full_prompt: Complete prompt after the final tool-processing step.

    Returns:
        The prompt guaranteed to contain an <answer>...</answer> section.
    """

    answer_pattern: re.Pattern[str] = re.compile(
        pattern=r"<answer>.*?</answer>", flags=re.DOTALL
    )

    if answer_pattern.search(string=full_prompt) is not None:
        return full_prompt

    # Append a minimal blank answer as required by the tool system rules
    if full_prompt.endswith("\n"):
        return f"{full_prompt}<answer></answer>"
    return f"{full_prompt}\n<answer></answer>"
