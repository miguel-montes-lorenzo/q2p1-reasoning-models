# tool_use/tool_handler.py

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlm.config import SYSTEM_PROMPT, SYSTEM_PROMPT_END

from tool_use.config import TOOL_SYSTEM_PROMPT


def insert_tool_desciptions_in_system_propt(descriptions: dict[str, str]) -> str:
    """Build a system prompt with tool spec and per-tool descriptions.

    Args:
        descriptions: Mapping from tool name to a human-readable description.

    Returns:
        Full system prompt including tool rules and tool descriptions.
    """
    tool_descriptions: str = "TOOL DESCRIPTIONS:"
    for tool_name, tool_description in descriptions.items():
        tool_descriptions = (
            f"{tool_descriptions}\n{tool_name}:\n{tool_description}\n---"
        )
    tool_descriptions = f"{tool_descriptions}\n\n"
    return f"{SYSTEM_PROMPT}{TOOL_SYSTEM_PROMPT}{tool_descriptions}{SYSTEM_PROMPT_END}"


_ALLOWED_PARENT_TAGS: tuple[str, ...] = ("question", "think", "tools", "answer")

_ERR_BAD_TOOL_FORMAT: str = "Error: bad format for <tool ...> tag"
_ERR_BAD_ID_FORMAT: str = "Error: bad format for <id ...> tag"
_ERR_ID_UNIDENTIFIED: str = "Error: <id ...> tag references unidentified result"
_ERR_CONTENT_OUTSIDE: str = "Error: content outside of any tag section"
_ERR_MISSING_ANSWER: str = "Error: missing <answer> section"


@dataclass(frozen=True)
class ToolCall:
    """Parsed tool call.

    Attributes:
        tool_id: Integer id for the tool call.
        name: Tool name.
        raw_args: Args dict with raw string values (may contain <id=...>).
        deps: Referenced tool ids inside args values.
    """

    tool_id: int
    name: str
    raw_args: dict[str, str]
    deps: set[int]


@dataclass(frozen=True)
class ParsedIteration:
    """Parsed representation of a model output iteration.

    Attributes:
        think: Raw think inner text (no tags), or None if missing.
        answer: Raw answer inner text (no tags), or None if missing.
        tool_calls: Parsed tool calls found inside think.
        error: Error message if parsing must stop, else None.
    """

    think: str | None
    answer: str | None
    tool_calls: list[ToolCall]
    error: str | None


def _find_valid_parent_blocks(*, text: str) -> list[tuple[str, int, int, str]]:
    """Find valid top-level parent blocks among allowed tags.

    A parent block is:
      - properly opened and closed (<tag> ... </tag>)
      - not nested inside another valid parent block

    Invalid/incomplete tags are ignored and do not cause errors.

    Args:
        text: Full model output.

    Returns:
        List of (tag, start, end, inner_text) in document order.
    """
    open_pat: re.Pattern[str] = re.compile(
        pattern=r"<(question|think|tools|answer)>", flags=re.IGNORECASE
    )
    close_pat: re.Pattern[str] = re.compile(
        pattern=r"</(question|think|tools|answer)>", flags=re.IGNORECASE
    )

    tokens: list[tuple[str, str, int, int]] = []
    for m in open_pat.finditer(text):
        tokens.append(("open", m.group(1).lower(), int(m.start()), int(m.end())))
    for m in close_pat.finditer(text):
        tokens.append(("close", m.group(1).lower(), int(m.start()), int(m.end())))

    tokens.sort(key=lambda x: (x[2], 0 if x[0] == "open" else 1))

    stack: list[tuple[str, int, int]] = []
    blocks: list[tuple[str, int, int, str]] = []

    for kind, tag, s, e in tokens:
        if kind == "open":
            stack.append((tag, s, e))
            continue

        if not stack:
            continue
        top_tag, top_s, top_e = stack[-1]
        if top_tag != tag:
            continue

        stack.pop()
        inner: str = text[top_e:s]
        if stack:
            continue  # nested -> not a parent block
        blocks.append((tag, top_s, e, inner))

    blocks.sort(key=lambda x: x[1])
    return blocks


def _extract_single_parent(
    *, blocks: list[tuple[str, int, int, str]], tag: str
) -> str | None:
    """Return inner text of the first parent block for tag, else None."""
    for t, _, __, inner in blocks:
        if t == tag:
            return inner
    return None


def _iter_id_tags(*, text: str) -> list[str]:
    """Return all raw '<id...>' tags found in text."""
    return [
        m.group(0) for m in re.finditer(r"<\s*id\b[^>]*>", text, flags=re.IGNORECASE)
    ]


def _validate_id_tags(*, text: str) -> bool:
    """Validate that all <id ...> are exactly '<id=INTEGER>'."""
    for raw in _iter_id_tags(text=text):
        if re.fullmatch(r"<id=\d+>", raw) is None:
            return False
    return True


def _validate_no_tool_end_tags(*, text: str) -> bool:
    """Disallow '</tool>' and self-closing '<tool .../>'."""
    if re.search(r"</\s*tool\s*>", text, flags=re.IGNORECASE) is not None:
        return False
    if re.search(r"<tool\b[^>]*/\s*>", text, flags=re.IGNORECASE) is not None:
        return False
    return True


def _iter_tool_start_tags(*, text: str) -> list[str]:
    """Extract <tool ...> start tags while honoring quotes inside attributes.

    This prevents truncating tags when '>' appears inside quoted values, e.g.
    args={expression='<id=1> / <id=2>'}.

    Args:
        text: A think block's inner text.

    Returns:
        List of full '<tool ...>' start tag strings.
    """
    tags: list[str] = []
    i: int = 0
    n: int = len(text)

    while True:
        start: int = text.find("<tool", i)
        if start == -1:
            break

        j: int = start
        quote: str | None = None
        escape: bool = False

        while j < n:
            ch: str = text[j]

            if escape:
                escape = False
                j += 1
                continue

            if ch == "\\":
                escape = True
                j += 1
                continue

            if quote is None and ch in ("'", '"'):
                quote = ch
                j += 1
                continue

            if quote is not None and ch == quote:
                quote = None
                j += 1
                continue

            if ch == ">" and quote is None:
                tags.append(text[start : j + 1])
                i = j + 1
                break

            j += 1
        else:
            break

    return tags


def _split_args_items(*, args_body: str) -> list[str]:
    """Split args dict entries by commas, respecting quotes and escapes."""
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
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

        if quote is None and ch in ("'", '"'):
            current.append(ch)
            quote = ch
            continue

        if quote is not None and ch == quote:
            current.append(ch)
            quote = None
            continue

        if ch == "," and quote is None:
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


def _unescape_value(*, value: str) -> str:
    """Unescape simple backslash-escaped quotes."""
    return value.replace("\\'", "'").replace('\\"', '"')


def _parse_args_dict(*, args_body: str) -> dict[str, str]:
    """Parse args={...} content into dict[str, str].

    Accepted formats:
      - key="value" / key='value'
      - "key": "value" / 'key': 'value'

    Args:
        args_body: Inside args={...} without braces.

    Returns:
        Parsed args mapping.

    Raises:
        ValueError: If any item cannot be parsed.
    """
    args: dict[str, str] = {}

    patterns: list[re.Pattern[str]] = [
        re.compile(
            pattern=(
                r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
                r"(?:'((?:\\.|[^'])*)'|\"((?:\\.|[^\"])*)\")\s*$"
            ),
            flags=re.DOTALL,
        ),
        re.compile(
            pattern=(
                r"\s*(?:'([^']*)'|\"([^\"]*)\")\s*:\s*"
                r"(?:'((?:\\.|[^'])*)'|\"((?:\\.|[^\"])*)\")\s*$"
            ),
            flags=re.DOTALL,
        ),
    ]

    for item in _split_args_items(args_body=args_body):
        m0: re.Match[str] | None = patterns[0].match(string=item)
        if m0 is not None:
            key0: str = m0.group(1)
            val0: str = m0.group(2) if m0.group(2) is not None else m0.group(3) or ""
            args[key0] = _unescape_value(value=val0)
            continue

        m1: re.Match[str] | None = patterns[1].match(string=item)
        if m1 is not None:
            key1: str = m1.group(1) if m1.group(1) is not None else m1.group(2) or ""
            val1: str = m1.group(3) if m1.group(3) is not None else m1.group(4) or ""
            args[str(key1)] = _unescape_value(value=val1)
            continue

        raise ValueError("bad_args_item")

    return args


def _parse_tool_start_tag(*, tag_text: str) -> tuple[int, str, str]:
    """Parse a strict <tool ...> start tag.

    Required format:
      <tool id=INTEGER name=IDENT args={...}>

    Args:
        tag_text: Raw start tag text.

    Returns:
        (tool_id, tool_name, args_body_inside_braces)

    Raises:
        ValueError: If parsing fails.
    """
    m: re.Match[str] | None = re.fullmatch(
        pattern=(
            r"<tool\s+id\s*=\s*(\d+)\s+"
            r"name\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s+"
            r"args\s*=\s*\{(.*)\}\s*>"
        ),
        string=tag_text,
        flags=re.DOTALL,
    )
    if m is None:
        raise ValueError("bad_tool_tag")
    return int(m.group(1)), m.group(2), m.group(3)


def _extract_id_deps(*, value: str) -> set[int]:
    """Extract referenced ids from a string value."""
    deps: set[int] = set()
    for m in re.finditer(r"<id=(\d+)>", value):
        deps.add(int(m.group(1)))
    return deps


def _format_ids_strict(*, text: str, outputs: dict[int, str]) -> str:
    """Replace <id=N> with outputs[N]. Missing ids are left intact."""

    def repl(match: re.Match[str]) -> str:
        ref_id: int = int(match.group(1))
        if ref_id in outputs:
            return outputs[ref_id]
        return match.group(0)

    return re.sub(pattern=r"<id=(\d+)>", repl=repl, string=text)


def _resolve_ids_or_raise(*, value: str, outputs: dict[int, str]) -> str:
    """Replace all <id=N> with outputs[N]. Missing ids raise ValueError."""

    def repl(match: re.Match[str]) -> str:
        ref_id: int = int(match.group(1))
        if ref_id not in outputs:
            raise ValueError("id_unidentified")
        return outputs[ref_id]

    return re.sub(pattern=r"<id=(\d+)>", repl=repl, string=value)


def _get_tool_fn(*, tool_dict: dict[str, Any], name: str) -> Callable[..., str]:
    """Return tool function by name."""
    if name not in tool_dict:
        raise ValueError("unknown_tool")
    fn_any: Any = tool_dict[name].get("function")
    if not callable(fn_any):
        raise ValueError("tool_not_callable")
    return fn_any


def _toposort_tools(*, calls: list[ToolCall]) -> list[ToolCall] | None:
    """Topologically sort tool calls by id-dependencies.

    Args:
        calls: Tool calls list.

    Returns:
        Ordered list if acyclic, else None.
    """
    by_id: dict[int, ToolCall] = {c.tool_id: c for c in calls}
    indeg: dict[int, int] = {c.tool_id: 0 for c in calls}
    children: dict[int, set[int]] = defaultdict(set)

    for c in calls:
        for dep in c.deps:
            if dep not in by_id:
                continue
            children[dep].add(c.tool_id)
            indeg[c.tool_id] += 1

    q: deque[int] = deque([tid for tid, d in indeg.items() if d == 0])
    out_ids: list[int] = []

    while q:
        tid: int = q.popleft()
        out_ids.append(tid)
        for ch in children.get(tid, set()):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                q.append(ch)

    if len(out_ids) != len(calls):
        return None
    return [by_id[tid] for tid in out_ids]


def _render_tools_block(*, records: list[dict[str, Any]]) -> str:
    """Render a tools execution trace in the requested <tools> block format."""
    lines: list[str] = ["<tools>", "{"]

    for rec in records:
        tool_id: int = int(rec["id"])
        tool_name: str = str(rec["tool"])
        args: dict[str, str] = dict(rec["args"])
        output: str = str(rec["output"])

        lines.append(f"    {tool_id}: {{")
        lines.append(f'        tool: "{tool_name}",')
        lines.append("        args: {")
        for k, v in args.items():
            escaped_v: str = v.replace('"', r"\"")
            lines.append(f'            "{k}": "{escaped_v}"')
        lines.append("        }")
        escaped_out: str = output.replace('"', r"\"")
        lines.append(f'        output: "{escaped_out}"')
        lines.append("    }")

    lines.append("}")
    lines.append("</tools>")
    return "\n".join(lines)


def _parse_iteration(*, model_output: str) -> ParsedIteration:
    """Parse a model output into think/answer/tool calls or an error.

    Enforced rules:
      - Parent blocks are valid only if top-level and properly closed.
      - Content outside parent blocks is ignored (never an error).
      - <id=...> tags are validated only inside think+answer.
      - <tool ...> calls are parsed only inside think.
      - <tool ...> tags are forbidden inside answer.

    Args:
        model_output: Raw model output text.

    Returns:
        ParsedIteration containing sections and parsed tool calls, or an error.
    """
    blocks: list[tuple[str, int, int, str]] = _find_valid_parent_blocks(
        text=model_output
    )

    think: str | None = _extract_single_parent(blocks=blocks, tag="think")
    answer: str | None = _extract_single_parent(blocks=blocks, tag="answer")

    # Forbid <tool ...> inside answer.
    if answer is not None:
        if (
            re.search(pattern=r"<tool\b", string=answer, flags=re.IGNORECASE)
            is not None
        ):
            return ParsedIteration(
                think=think, answer=answer, tool_calls=[], error=_ERR_BAD_TOOL_FORMAT
            )

    # Validate <id ...> format inside think+answer only.
    inner_concat: str = ""
    if think is not None:
        inner_concat += think
    if answer is not None:
        inner_concat += answer
    if not _validate_id_tags(text=inner_concat):
        return ParsedIteration(
            think=think, answer=answer, tool_calls=[], error=_ERR_BAD_ID_FORMAT
        )

    if think is None:
        return ParsedIteration(think=None, answer=answer, tool_calls=[], error=None)

    if not _validate_no_tool_end_tags(text=think):
        return ParsedIteration(
            think=think, answer=answer, tool_calls=[], error=_ERR_BAD_TOOL_FORMAT
        )

    tool_tags: list[str] = _iter_tool_start_tags(text=think)
    tool_calls: list[ToolCall] = []

    for raw_tool_tag in tool_tags:
        try:
            tool_id, tool_name, args_body = _parse_tool_start_tag(tag_text=raw_tool_tag)
            raw_args: dict[str, str] = _parse_args_dict(args_body=args_body)
        except Exception:
            return ParsedIteration(
                think=think, answer=answer, tool_calls=[], error=_ERR_BAD_TOOL_FORMAT
            )

        deps: set[int] = set()
        for v in raw_args.values():
            deps |= _extract_id_deps(value=v)

        tool_calls.append(
            ToolCall(tool_id=tool_id, name=tool_name, raw_args=raw_args, deps=deps)
        )

    return ParsedIteration(
        think=think, answer=answer, tool_calls=tool_calls, error=None
    )


def parse_and_execute_tool_call(
    model_output: str,
    tool_dict: dict[str, Any],
    max_calls: int = 3,
    *,
    used_tool_ids: set[int] | None = None,
    global_outputs: dict[int, str] | None = None,
) -> tuple[bool, str, str, str, str]:
    """Parse one iteration, optionally execute tools, and decide whether to continue.

    Returns:
        (should_continue, prompt_appendix, raw_full_output, formatted_full_output,
         parsed_answer_only)
    """
    _ = max_calls

    used_ids: set[int] = used_tool_ids if used_tool_ids is not None else set()
    outputs_all: dict[int, str] = global_outputs if global_outputs is not None else {}

    parsed: ParsedIteration = _parse_iteration(model_output=model_output)
    if parsed.error is not None:
        think_block: str = (
            f"<think>{parsed.think}</think>"
            if parsed.think is not None
            else "<think></think>"
        )
        forced_answer: str = f"<answer>{parsed.error}</answer>"
        raw_full: str = f"{think_block}\n{forced_answer}"
        return False, "", raw_full, raw_full, forced_answer

    think_block: str = (
        f"<think>{parsed.think}</think>"
        if parsed.think is not None
        else "<think></think>"
    )
    answer_block_raw: str | None = None
    if parsed.answer is not None:
        answer_block_raw = f"<answer>{parsed.answer}</answer>"

    # Global uniqueness of tool ids.
    for c in parsed.tool_calls:
        if c.tool_id in used_ids:
            forced_answer = f"<answer>{_ERR_BAD_TOOL_FORMAT}</answer>"
            raw_full = f"{think_block}\n{forced_answer}"
            return False, "", raw_full, raw_full, forced_answer

    topo: list[ToolCall] | None = _toposort_tools(calls=parsed.tool_calls)
    if topo is None:
        forced_answer = f"<answer>{_ERR_BAD_TOOL_FORMAT}</answer>"
        raw_full = f"{think_block}\n{forced_answer}"
        return False, "", raw_full, raw_full, forced_answer

    tool_records: list[dict[str, Any]] = []

    for c in topo:
        used_ids.add(c.tool_id)

        try:
            resolved_args: dict[str, str] = {
                k: _resolve_ids_or_raise(value=v, outputs=outputs_all)
                for k, v in c.raw_args.items()
            }
        except Exception:
            forced_answer = f"<answer>{_ERR_ID_UNIDENTIFIED}</answer>"
            raw_full = f"{think_block}\n{forced_answer}"
            return False, "", raw_full, raw_full, forced_answer

        try:
            fn: Callable[..., str] = _get_tool_fn(tool_dict=tool_dict, name=c.name)
        except Exception:
            forced_answer = f"<answer>{_ERR_BAD_TOOL_FORMAT}</answer>"
            raw_full = f"{think_block}\n{forced_answer}"
            return False, "", raw_full, raw_full, forced_answer

        try:
            result: str = fn(**resolved_args)
        except Exception:
            forced_answer = f"<answer>{_ERR_BAD_TOOL_FORMAT}</answer>"
            raw_full = f"{think_block}\n{forced_answer}"
            return False, "", raw_full, raw_full, forced_answer

        outputs_all[c.tool_id] = result
        tool_records.append({
            "id": c.tool_id,
            "tool": c.name,
            "args": dict(c.raw_args),
            "output": result,
        })

    tools_block: str = (
        _render_tools_block(records=tool_records)
        if tool_records
        else "<tools>\n{\n}\n</tools>"
    )

    # If answer exists: validate <id=...> refs and return final outputs.
    if parsed.answer is not None:
        for m in re.finditer(r"<id=(\d+)>", parsed.answer):
            ref_id: int = int(m.group(1))
            if ref_id not in outputs_all:
                forced_answer = f"<answer>{_ERR_ID_UNIDENTIFIED}</answer>"
                raw_full = f"{think_block}\n{forced_answer}"
                return False, "", raw_full, raw_full, forced_answer

        answer_formatted_inner: str = _format_ids_strict(
            text=parsed.answer,
            outputs=outputs_all,
        )
        parsed_answer_only: str = f"<answer>{answer_formatted_inner}</answer>"

        raw_full_output: str = f"{think_block}\n{tools_block}\n{answer_block_raw}"
        formatted_full_output: str = (
            f"{think_block}\n{tools_block}\n{parsed_answer_only}"
        )
        return False, "", raw_full_output, formatted_full_output, parsed_answer_only

    # No answer => continue, provide appendix with resolved ids in args.
    resolved_records_for_prompt: list[dict[str, Any]] = []
    for rec in tool_records:
        raw_args: dict[str, str] = dict(rec["args"])
        resolved_args: dict[str, str] = {}
        for k, v in raw_args.items():
            resolved_args[k] = _format_ids_strict(text=v, outputs=outputs_all)
        resolved_records_for_prompt.append({
            "id": rec["id"],
            "tool": rec["tool"],
            "args": resolved_args,
            "output": rec["output"],
        })

    tools_block_for_prompt: str = (
        _render_tools_block(records=resolved_records_for_prompt)
        if resolved_records_for_prompt
        else tools_block
    )

    prompt_appendix: str = f"{think_block}\n{tools_block_for_prompt}\n"
    raw_full_output = f"{think_block}\n{tools_block}"
    formatted_full_output = raw_full_output
    parsed_answer_only = f"<answer>{_ERR_MISSING_ANSWER}</answer>"

    return (
        True,
        prompt_appendix,
        raw_full_output,
        formatted_full_output,
        parsed_answer_only,
    )


def ensure_response_contains_answer(full_prompt: str) -> str:
    """Ensure final output includes an <answer> section.

    Policy:
      - If model produced <answer>...</answer> (even empty), keep it.
      - If missing, force a non-empty error answer.

    Args:
        full_prompt: Final output candidate.

    Returns:
        Output guaranteed to contain an <answer>...</answer> section.
    """
    has_answer: bool = (
        re.search(r"<answer>.*?</answer>", full_prompt, flags=re.DOTALL | re.IGNORECASE)
        is not None
    )
    if has_answer:
        return full_prompt

    think_match: re.Match[str] | None = re.search(
        pattern=r"<think>.*?</think>",
        string=full_prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    think_block: str = (
        think_match.group(0) if think_match is not None else "<think></think>"
    )
    forced_answer: str = f"<answer>{_ERR_MISSING_ANSWER}</answer>"

    if full_prompt.strip() == think_block.strip():
        return f"{think_block}\n{forced_answer}"
    return f"{full_prompt}\n{forced_answer}"
