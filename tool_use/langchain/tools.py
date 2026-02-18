# tool_use/langchain/tools.py

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

load_dotenv()  # Load from current working directory if present


def calculator(*, expression: str) -> str:
    """Deterministically evaluate a safe arithmetic expression and return its result.

    This tool performs exact numerical computation over a restricted subset of
    Python arithmetic syntax. It should be used whenever a reasoning step requires
    a precise numeric value instead of mental math or approximation.

    The evaluation is sandboxed by strict character whitelisting before calling
    `eval`. Only characters in the set `0123456789+-*/(). ` are accepted.

    Supported syntax:
        - Integers and decimal numbers (e.g., `2`, `3.5`)
        - Parentheses for grouping (e.g., `(2 + 3) * 4`)
        - Chained arithmetic operations
        - Exponentiation using Python syntax `**` (e.g., `45234**(1/2)`)

    WHEN TO USE (**VERY IMPORTANT!!!**):
        - Assume that you **cannot do any trivial math operation without this tool**.
          Not even 1+1 or 1*1, and of course nothing harder either.
        - Use this tool for **all scalar mathematical operations**.

    Args:
        expression: Arithmetic expression string composed only of allowed
            numeric characters, operators, parentheses, decimal points,
            and whitespace.

    Returns:
        The evaluated numeric result as a string.

    Raises:
        ValueError: If the expression contains disallowed characters.
        RuntimeError: If evaluation fails for any reason.
    """
    allowed: set[str] = set("0123456789+-*/(). ")
    disallowed: set[str] = set(expression) - allowed
    if disallowed:
        invalid_char: str = sorted(disallowed)[0]
        raise ValueError(f"Disallowed character '{invalid_char}' in expression.")

    try:
        result: object = eval(expression)
        return str(result)
    except Exception as exc:
        raise RuntimeError(f"Error calculating expression: {exc}") from exc


@dataclass(frozen=True)
class ToolSpec:
    """Small tool registry entry for interop with legacy code.

    Attributes:
        function: Callable that implements the tool.
        description: Human-readable tool description.
    """

    function: Callable[..., str]
    description: str


def _docstring_description(*, fn: Callable[..., Any]) -> str:
    """Extract a clean tool description from a function docstring.

    This uses `inspect.getdoc`, which dedents and strips the docstring. If the
    docstring is missing/empty, it returns a short fallback string.

    Args:
        fn: Tool function.

    Returns:
        Cleaned docstring text suitable for tool descriptions.
    """
    doc: str | None = inspect.getdoc(fn)
    if doc is None or not doc.strip():
        return f"{fn.__name__} tool (no docstring provided)."
    return doc.strip()


TOOL_DICT: dict[str, dict[str, Any]] = {
    "calculator": {
        "function": calculator,
        "description": _docstring_description(fn=calculator),
    }
}


def get_langchain_tools() -> list[StructuredTool]:
    """Build LangChain StructuredTool objects from TOOL_DICT.

    Returns:
        A list of StructuredTool instances that mirror TOOL_DICT.
    """
    tools: list[StructuredTool] = []
    for name, payload in TOOL_DICT.items():
        fn: Callable[..., str] = payload["function"]
        description: str = str(payload["description"])
        tools.append(
            StructuredTool.from_function(
                func=fn,
                name=name,
                description=description,
            )
        )
    return tools
