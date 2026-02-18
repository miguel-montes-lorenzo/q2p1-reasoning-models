# tool_use/tool_handler.py

from __future__ import annotations

import inspect
import re
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlm.config import SYSTEM_PROMPT, SYSTEM_PROMPT_END
from tool_use.custom.config import TOOL_SYSTEM_PROMPT

_ALLOWED_PARENT_TAGS: tuple[str, ...] = ("question", "think", "tools", "answer")

# Centralized pattern for id references: only "@<id>" where <id> is alnum/_ and
# it is NOT a tool call (i.e., not followed by "(" possibly with spaces).
_ID_REF_RE: re.Pattern[str] = re.compile(pattern=r"@([A-Za-z0-9_]+)\b(?!\s*\()")

# These strings must match the "ALLOWED ERROR MESSAGES" contract.
_ERR_INCORRECT_AT_USE: str = (
    "Incorrect use of @, allowed formats are: "
    "{@available_funcion_name(arg1=..., arg2=..., ...)->int_id, @defined_int_id}"
)
_ERR_CALL_UNEXISTENT_FUNCTION_PREFIX: str = "Call to unexistent function: "
_ERR_REF_UNDEFINED_ID_PREFIX: str = "Reference to undefined id: "
_ERR_UNEXISTENT_ARGUMENTS_PREFIX: str = "Use of unexistent arguments for function "


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


@dataclass(frozen=True)
class ToolCall:
    """Parsed tool call.

    Attributes:
        tool_id: Identifier id for the tool call.
        name: Tool name.
        raw_args: Args dict with raw string values (may contain @ID references).
        deps: Referenced tool ids inside args values.
    """

    tool_id: str
    name: str
    raw_args: dict[str, str]
    deps: set[str]


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
        top_tag, _, top_e = stack[-1]
        if top_tag != tag:
            continue

        stack.pop()
        inner: str = text[top_e:s]
        if stack:
            continue
        blocks.append((tag, s, e, inner))

    blocks.sort(key=lambda x: x[1])
    return blocks


def _extract_last_parent(
    *, blocks: list[tuple[str, int, int, str]], tag: str
) -> str | None:
    """Return inner text of the last parent block for tag, else None."""
    last: str | None = None
    for t, _, __, inner in blocks:
        if t == tag:
            last = inner
    return last


def _split_args_items(*, args_body: str) -> list[str]:
    """Split call arguments by commas, respecting quotes and escapes."""
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


def _parse_call_args(*, args_body: str) -> dict[str, str]:
    """Parse @tool_name(...) argument content into dict[str, str].

    Accepted formats:
      - key="value" / key='value'
      - "key": "value" / 'key': 'value'

    Note:
      - Treat \" as " and \' as ' for parsing purposes.

    Args:
        args_body: Inside (...) without parentheses.

    Returns:
        Parsed args mapping.

    Raises:
        ValueError: If any item cannot be parsed.
    """
    args_body = args_body.replace('\\"', '"').replace("\\'", "'")

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


def _iter_tool_calls(*, text: str) -> list[tuple[str, str, str]]:
    """Extract @tool_name(...)->ID calls from text.

    This scanner respects quotes and backslash escapes inside (...).

    Args:
        text: A think block's inner text.

    Returns:
        List of (tool_name, args_body, id_str) in document order.
    """
    calls: list[tuple[str, str, str]] = []
    i: int = 0
    n: int = len(text)

    while i < n:
        start: int = text.find("@", i)
        if start == -1:
            break

        if start + 1 >= n or re.match(r"[A-Za-z_]", text[start + 1]) is None:
            i = start + 1
            continue

        m_name: re.Match[str] | None = re.match(
            pattern=r"@([A-Za-z_][A-Za-z0-9_]*)\(",
            string=text[start:],
            flags=re.DOTALL,
        )
        if m_name is None:
            i = start + 1
            continue

        tool_name: str = m_name.group(1)
        j: int = start + len(m_name.group(0))

        quote: str | None = None
        escape: bool = False

        while j < n:
            ch: str = text[j]

            if ch == "\\" and (j + 1) < n and text[j + 1] in ("'", '"'):
                j += 1
                continue

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

            if ch == ")" and quote is None:
                break

            j += 1

        if j >= n or text[j] != ")":
            i = start + 1
            continue

        args_body: str = text[start + len(f"@{tool_name}(") : j]

        k: int = j + 1
        while k < n and text[k].isspace():
            k += 1
        if k + 1 >= n or text[k : k + 2] != "->":
            i = start + 1
            continue

        k += 2
        while k < n and text[k].isspace():
            k += 1

        m_id: re.Match[str] | None = re.match(
            pattern=r"([A-Za-z0-9_]+)", string=text[k:]
        )
        if m_id is None:
            i = start + 1
            continue

        id_str: str = m_id.group(1)
        calls.append((tool_name, args_body, id_str))

        i = k + len(id_str)

    return calls


def _extract_id_deps(*, value: str) -> set[str]:
    """Extract referenced ids (@ID) from a string value."""
    deps: set[str] = set()
    for m in _ID_REF_RE.finditer(value):
        deps.add(str(m.group(1)))
    return deps


def _format_ids_strict(*, text: str, outputs: dict[str, str]) -> str:
    """Replace @ID with outputs[ID]. Missing ids are left intact."""

    def repl(match: re.Match[str]) -> str:
        ref_id: str = str(match.group(1))
        if ref_id in outputs:
            return outputs[ref_id]
        return match.group(0)

    return _ID_REF_RE.sub(repl=repl, string=text)


def _resolve_ids_or_raise(*, value: str, outputs: dict[str, str]) -> str:
    """Replace all @ID with outputs[ID]. Missing ids raise ValueError."""

    def repl(match: re.Match[str]) -> str:
        ref_id: str = str(match.group(1))
        if ref_id not in outputs:
            raise ValueError(f"undefined_id:{ref_id}")
        return outputs[ref_id]

    return _ID_REF_RE.sub(repl=repl, string=value)


def _get_tool_fn(*, tool_dict: dict[str, Any], name: str) -> Callable[..., str]:
    """Return tool function by name."""
    fn_any: Any = tool_dict.get(name, {}).get("function")
    if not callable(fn_any):
        raise ValueError("unknown_tool")
    return fn_any


def _validate_kwargs_or_raise(
    *, fn: Callable[..., Any], tool_name: str, kwargs: dict[str, str]
) -> None:
    """Validate that kwargs exist in the function signature.

    Args:
        fn: Callable tool function.
        tool_name: Tool name (for error formatting).
        kwargs: Parsed keyword arguments.

    Raises:
        ValueError: If any argument name is not accepted by the function.
    """
    sig: inspect.Signature = inspect.signature(fn)
    params: dict[str, inspect.Parameter] = dict(sig.parameters)

    has_var_kw: bool = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_var_kw:
        return

    allowed: set[str] = set()
    for name, p in params.items():
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            allowed.add(name)

    for k in kwargs:
        if k not in allowed:
            raise ValueError(f"bad_arg:{tool_name}:{k}")


def _toposort_tools(*, calls: list[ToolCall]) -> list[ToolCall] | None:
    """Topologically sort tool calls by id-dependencies.

    Args:
        calls: Tool calls list.

    Returns:
        Ordered list if acyclic, else None.
    """
    by_id: dict[str, ToolCall] = {c.tool_id: c for c in calls}
    indeg: dict[str, int] = {c.tool_id: 0 for c in calls}
    children: dict[str, set[str]] = defaultdict(set)

    for c in calls:
        for dep in c.deps:
            if dep not in by_id:
                continue
            children[dep].add(c.tool_id)
            indeg[c.tool_id] += 1

    q: deque[str] = deque([tid for tid, d in indeg.items() if d == 0])
    out_ids: list[str] = []

    while q:
        tid: str = q.popleft()
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
        tool_id: str = str(rec["id"])
        tool_name: str = str(rec["tool"])
        args: dict[str, str] = dict(rec["args"])
        output: str = str(rec["output"])
        successful_execution: bool = bool(rec["successful_execution"])

        lines.append(f"    {tool_id}: {{")
        lines.append(f"        tool: '{tool_name}',")
        lines.append(f"        successful_execution: {successful_execution},")
        lines.append("        args: {")
        for k, v in args.items():
            escaped_v: str = v.replace("'", "\\'")
            lines.append(f"            '{k}': '{escaped_v}',")
        lines.append("        },")
        escaped_out: str = output.replace("'", "\\'")
        lines.append(f"        output: '{escaped_out}'")
        lines.append("    },")

    lines.append("}")
    lines.append("</tools>")
    return "\n".join(lines)


def _render_error_block(*, error_message: str) -> str:
    """Render an <error> block."""
    return f"<error>{error_message}</error>"


def _find_first_invalid_at_use(
    *, text: str, tool_names: set[str]
) -> tuple[bool, str | None]:
    """Validate that every '@' use matches allowed formats.

    Allowed formats:
      - @TOOL_NAME(...)->ID
      - @ID
      - Any other '@' not followed by digit/letter/'_' is ignored.

    Args:
        text: Think block inner text.
        tool_names: Set of available tool names.

    Returns:
        (ok, error_message_if_any)
    """
    n: int = len(text)
    i: int = 0

    while i < n:
        at_pos: int = text.find("@", i)
        if at_pos == -1:
            return True, None

        if at_pos + 1 >= n:
            i = at_pos + 1
            continue

        nxt: str = text[at_pos + 1]

        # Ignore stray '@' like "@ ", "@.", "@," (not a tool/ref attempt).
        if not (nxt.isalnum() or nxt == "_"):
            i = at_pos + 1
            continue

        # Accept numeric-leading id references like @1, @12, @2_foo.
        if nxt.isdigit():
            j_num: int = at_pos + 1
            while j_num < n and (text[j_num].isalnum() or text[j_num] == "_"):
                j_num += 1
            i = j_num
            continue

        # If it looks like a tool call, validate the whole call.
        m_name: re.Match[str] | None = re.match(
            pattern=r"@([A-Za-z_][A-Za-z0-9_]*)",
            string=text[at_pos:],
            flags=re.DOTALL,
        )
        if m_name is None:
            return False, _ERR_INCORRECT_AT_USE

        token: str = m_name.group(1)
        j: int = at_pos + 1 + len(token)

        # Ignore the literal placeholders that appear in our own error contract,
        # in case the model quotes them inside <think>/<answer>.
        if token in {"available_funcion_name", "defined_int_id"}:
            i = j
            continue

        # Tool call iff next non-space is '('.
        k0: int = j
        while k0 < n and text[k0].isspace():
            k0 += 1

        if k0 < n and text[k0] == "(":
            tool_name: str = token
            if tool_name not in tool_names:
                return False, f"{_ERR_CALL_UNEXISTENT_FUNCTION_PREFIX}@{tool_name}"

            j2: int = k0 + 1
            quote: str | None = None
            escape: bool = False

            while j2 < n:
                ch: str = text[j2]

                if ch == "\\" and (j2 + 1) < n and text[j2 + 1] in ("'", '"'):
                    j2 += 1
                    continue

                if escape:
                    escape = False
                    j2 += 1
                    continue

                if ch == "\\":
                    escape = True
                    j2 += 1
                    continue

                if quote is None and ch in ("'", '"'):
                    quote = ch
                    j2 += 1
                    continue

                if quote is not None and ch == quote:
                    quote = None
                    j2 += 1
                    continue

                if ch == ")" and quote is None:
                    break

                j2 += 1

            if j2 >= n or text[j2] != ")":
                return False, _ERR_INCORRECT_AT_USE

            k: int = j2 + 1
            while k < n and text[k].isspace():
                k += 1
            if k + 1 >= n or text[k : k + 2] != "->":
                return False, _ERR_INCORRECT_AT_USE

            k += 2
            while k < n and text[k].isspace():
                k += 1
            if k >= n or not (text[k].isalnum() or text[k] == "_"):
                return False, _ERR_INCORRECT_AT_USE

            while k < n and (text[k].isalnum() or text[k] == "_"):
                k += 1

            i = k
            continue

        # Otherwise it's a reference @ID (consume token).
        i = j

    return True, None


def _parse_iteration(
    *, model_output: str, tool_dict: dict[str, Any]
) -> ParsedIteration:
    """Parse a model output into think/answer/tool calls or an error.

    Enforced rules:
      - Parent blocks are valid only if top-level and properly closed.
      - Content outside parent blocks is ignored.
      - Tool calls are parsed only inside <think>.
      - Tool calls are forbidden inside <answer> (only @ID refs are allowed).

    Args:
        model_output: Raw model output text.
        tool_dict: Tool registry (used for tool-name validation).

    Returns:
        ParsedIteration containing sections and parsed tool calls, or an error.
    """
    blocks: list[tuple[str, int, int, str]] = _find_valid_parent_blocks(
        text=model_output
    )

    # Use LAST blocks to handle model outputs that include multiple think/error chunks.
    think: str | None = _extract_last_parent(blocks=blocks, tag="think")
    answer: str | None = _extract_last_parent(blocks=blocks, tag="answer")

    if answer is not None:
        if (
            re.search(pattern=r"@[A-Za-z_][A-Za-z0-9_]*\s*\(", string=answer)
            is not None
        ):
            return ParsedIteration(
                think=think,
                answer=answer,
                tool_calls=[],
                error=(
                    "Incorrect use of @ in ANSWER, allowed formats are: "
                    "{@available_funcion_name(arg1=..., arg2=..., ...)->int_id, "
                    "@defined_int_id} "
                    f"[{answer}]"
                ),
            )
        if "@" in answer and _ID_REF_RE.search(answer) is None:
            return ParsedIteration(
                think=think,
                answer=answer,
                tool_calls=[],
                error=(
                    "Incorrect use of @ in ANSWER, allowed formats are: "
                    "{@available_funcion_name(arg1=..., arg2=..., ...)->int_id, "
                    "@defined_int_id} "
                    f"[{answer}]"
                ),
            )

    if think is None:
        return ParsedIteration(think=None, answer=answer, tool_calls=[], error=None)

    tool_names: set[str] = set(tool_dict.keys())
    ok_at, at_err = _find_first_invalid_at_use(text=think, tool_names=tool_names)
    if not ok_at:
        return ParsedIteration(think=think, answer=answer, tool_calls=[], error=at_err)

    tool_calls_raw: list[tuple[str, str, str]] = _iter_tool_calls(text=think)
    tool_calls: list[ToolCall] = []

    for tool_name, args_body, id_str in tool_calls_raw:
        if tool_name not in tool_dict:
            return ParsedIteration(
                think=think,
                answer=answer,
                tool_calls=[],
                error=f"{_ERR_CALL_UNEXISTENT_FUNCTION_PREFIX}@{tool_name}",
            )

        try:
            tool_id: str = str(id_str)
            raw_args: dict[str, str] = _parse_call_args(args_body=args_body)
        except Exception:
            return ParsedIteration(
                think=think,
                answer=answer,
                tool_calls=[],
                error=_ERR_INCORRECT_AT_USE,
            )

        deps: set[str] = set()
        for v in raw_args.values():
            deps |= _extract_id_deps(value=v)

        tool_calls.append(
            ToolCall(tool_id=tool_id, name=tool_name, raw_args=raw_args, deps=deps)
        )

    return ParsedIteration(
        think=think, answer=answer, tool_calls=tool_calls, error=None
    )


def _return_error_continue(
    *,
    think_block: str,
    tools_block: str | None,
    error_message: str,
) -> tuple[bool, str, str, str, str]:
    """Return an <error> block and force continuing the loop.

    Any <answer> produced in the same iteration must be ignored.

    Args:
        think_block: Rendered <think>...</think>.
        tools_block: Optional rendered <tools>...</tools>.
        error_message: Error message to put inside <error>.

    Returns:
        (should_continue, prompt_appendix, raw_full, formatted_full, parsed_answer_only)
    """
    err_block: str = _render_error_block(error_message=error_message)
    pieces: list[str] = [think_block]
    if tools_block is not None:
        pieces.append(tools_block)
    pieces.append(err_block)

    raw_full: str = "\n".join(pieces)
    prompt_appendix: str = f"{raw_full}\n"
    parsed_answer_only: str = "<answer>null</answer>"
    return True, prompt_appendix, raw_full, raw_full, parsed_answer_only


def parse_and_execute_tool_call(
    model_output: str,
    tool_dict: dict[str, Any],
    max_calls: int = 3,
    *,
    used_tool_ids: set[str] | None = None,
    global_outputs: dict[str, str] | None = None,
    is_last_iteration: bool = False,
) -> tuple[bool, str, str, str, str]:
    """Parse one iteration, execute tools, and decide whether to continue.

    Rules implemented:
      - On any tool format/semantic error: return <error>...</error> and CONTINUE.
        Any <answer> in the same iteration is ignored.
      - Tool ids are globally unique across the entire loop.
      - If a valid non-null <answer> exists, format @ID references and STOP.
      - If <answer>null</answer> is produced by the model, do NOT stop early.
      - If any tool execution fails (runtime error), continue and do NOT add <answer>
        in this iteration.

    Args:
        model_output: Raw model output from the LLM.
        tool_dict: Tool registry.
        max_calls: Maximum iterations (not enforced here, only in the caller).
        used_tool_ids: Global used ids across the loop.
        global_outputs: Global tool outputs across the loop.
        is_last_iteration: Whether this is the final iteration allowed by the caller.

    Returns:
        (should_continue, prompt_appendix, raw_full_output, formatted_full_output,
         parsed_answer_only)
    """
    _ = max_calls

    used_ids: set[str] = used_tool_ids if used_tool_ids is not None else set()
    outputs_all: dict[str, str] = global_outputs if global_outputs is not None else {}

    parsed: ParsedIteration = _parse_iteration(
        model_output=model_output,
        tool_dict=tool_dict,
    )

    think_block: str = (
        f"<think>{parsed.think}</think>"
        if parsed.think is not None
        else "<think></think>"
    )

    if parsed.error is not None:
        return _return_error_continue(
            think_block=think_block,
            tools_block=None,
            error_message=str(parsed.error),
        )

    for c in parsed.tool_calls:
        if c.tool_id in used_ids:
            return _return_error_continue(
                think_block=think_block,
                tools_block=None,
                error_message=_ERR_INCORRECT_AT_USE,
            )

    topo: list[ToolCall] | None = _toposort_tools(calls=parsed.tool_calls)
    if topo is None:
        return _return_error_continue(
            think_block=think_block,
            tools_block=None,
            error_message=_ERR_INCORRECT_AT_USE,
        )

    tool_records: list[dict[str, Any]] = []
    any_tool_failed: bool = False

    for c in topo:
        used_ids.add(c.tool_id)

        try:
            resolved_args: dict[str, str] = {
                k: _resolve_ids_or_raise(value=v, outputs=outputs_all)
                for k, v in c.raw_args.items()
            }
        except ValueError as e:
            msg: str = str(e)
            if msg.startswith("undefined_id:"):
                ref_id: str = msg.split(":", 1)[1]
                return _return_error_continue(
                    think_block=think_block,
                    tools_block=None,
                    error_message=f"{_ERR_REF_UNDEFINED_ID_PREFIX}@{ref_id}",
                )
            return _return_error_continue(
                think_block=think_block,
                tools_block=None,
                error_message=_ERR_INCORRECT_AT_USE,
            )

        try:
            fn: Callable[..., str] = _get_tool_fn(tool_dict=tool_dict, name=c.name)
        except Exception:
            return _return_error_continue(
                think_block=think_block,
                tools_block=None,
                error_message=f"{_ERR_CALL_UNEXISTENT_FUNCTION_PREFIX}@{c.name}",
            )

        try:
            _validate_kwargs_or_raise(fn=fn, tool_name=c.name, kwargs=resolved_args)
        except ValueError as e:
            msg2: str = str(e)
            if msg2.startswith("bad_arg:"):
                _, tool_name, bad_arg = msg2.split(":", 2)
                return _return_error_continue(
                    think_block=think_block,
                    tools_block=None,
                    error_message=f"{_ERR_UNEXISTENT_ARGUMENTS_PREFIX}@{tool_name}: "
                    f"{bad_arg}",
                )
            return _return_error_continue(
                think_block=think_block,
                tools_block=None,
                error_message=_ERR_INCORRECT_AT_USE,
            )

        successful_execution: bool = True
        try:
            result: str = str(fn(**resolved_args))
        except Exception as e:
            successful_execution = False
            any_tool_failed = True
            result = str(e)

        outputs_all[c.tool_id] = result
        tool_records.append({
            "id": c.tool_id,
            "tool": c.name,
            "args": dict(c.raw_args),
            "output": result,
            "successful_execution": successful_execution,
        })

        if any_tool_failed:
            break

    tools_block: str = (
        _render_tools_block(records=tool_records)
        if tool_records
        else "<tools>\n{\n}\n</tools>"
    )

    # If any tool failed, do not accept an <answer> in this iteration.
    if any_tool_failed:
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
                "successful_execution": rec["successful_execution"],
            })

        tools_block_for_prompt: str = _render_tools_block(
            records=resolved_records_for_prompt
        )
        prompt_appendix: str = f"{think_block}\n{tools_block_for_prompt}\n"
        raw_full_output: str = f"{think_block}\n{tools_block}"
        formatted_full_output: str = raw_full_output
        parsed_answer_only: str = "<answer>null</answer>"
        _ = is_last_iteration
        return (
            True,
            prompt_appendix,
            raw_full_output,
            formatted_full_output,
            parsed_answer_only,
        )

    # Do not stop early if the model outputs <answer>null</answer>.
    if parsed.answer is not None and parsed.answer.strip().lower() == "null":
        resolved_records_for_prompt2: list[dict[str, Any]] = []
        for rec in tool_records:
            raw_args2: dict[str, str] = dict(rec["args"])
            resolved_args2: dict[str, str] = {}
            for k, v in raw_args2.items():
                resolved_args2[k] = _format_ids_strict(text=v, outputs=outputs_all)
            resolved_records_for_prompt2.append({
                "id": rec["id"],
                "tool": rec["tool"],
                "args": resolved_args2,
                "output": rec["output"],
                "successful_execution": rec["successful_execution"],
            })

        tools_block_for_prompt2: str = (
            _render_tools_block(records=resolved_records_for_prompt2)
            if resolved_records_for_prompt2
            else tools_block
        )
        prompt_appendix2: str = f"{think_block}\n{tools_block_for_prompt2}\n"
        raw_full_output2: str = f"{think_block}\n{tools_block}"
        formatted_full_output2: str = raw_full_output2
        parsed_answer_only2: str = "<answer>null</answer>"
        return (
            True,
            prompt_appendix2,
            raw_full_output2,
            formatted_full_output2,
            parsed_answer_only2,
        )

    if parsed.answer is not None:
        for m in _ID_REF_RE.finditer(parsed.answer):
            ref_id: str = str(m.group(1))
            if ref_id not in outputs_all:
                return _return_error_continue(
                    think_block=think_block,
                    tools_block=tools_block,
                    error_message=(
                        f"Reference to undefined id in ANSWER: @{ref_id} "
                        f"[{parsed.answer}]"
                    ),
                )

        answer_formatted_inner: str = _format_ids_strict(
            text=parsed.answer,
            outputs=outputs_all,
        )
        parsed_answer_only3: str = f"<answer>{answer_formatted_inner}</answer>"

        raw_full_output3: str = (
            f"{think_block}\n{tools_block}\n<answer>{parsed.answer}</answer>"
        )
        formatted_full_output3: str = (
            f"{think_block}\n{tools_block}\n{parsed_answer_only3}"
        )
        return False, "", raw_full_output3, formatted_full_output3, parsed_answer_only3

    resolved_records_for_prompt3: list[dict[str, Any]] = []
    for rec in tool_records:
        raw_args3: dict[str, str] = dict(rec["args"])
        resolved_args3: dict[str, str] = {}
        for k, v in raw_args3.items():
            resolved_args3[k] = _format_ids_strict(text=v, outputs=outputs_all)
        resolved_records_for_prompt3.append({
            "id": rec["id"],
            "tool": rec["tool"],
            "args": resolved_args3,
            "output": rec["output"],
            "successful_execution": rec["successful_execution"],
        })

    tools_block_for_prompt3: str = _render_tools_block(
        records=resolved_records_for_prompt3
    )
    prompt_appendix3: str = f"{think_block}\n{tools_block_for_prompt3}\n"
    raw_full_output4: str = f"{think_block}\n{tools_block}"
    formatted_full_output4: str = raw_full_output4
    parsed_answer_only4: str = "<answer>null</answer>"

    return (
        True,
        prompt_appendix3,
        raw_full_output4,
        formatted_full_output4,
        parsed_answer_only4,
    )


def ensure_response_contains_answer(full_prompt: str) -> str:
    """Ensure final output includes an <answer> section.

    Policy:
      - If model produced <answer>...</answer> (even empty), keep it.
      - If missing, force <answer>null</answer>.

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
    return f"{full_prompt}\n<answer>null</answer>"
