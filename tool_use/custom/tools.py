# tool_use/tools.py

from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Intenta cargar desde el directorio de trabajo actual


def calculator(expression: str) -> str:
    """Deterministically evaluate a safe arithmetic expression and return its result.

    This tool performs exact numerical computation over a restricted subset of
    Python arithmetic syntax. It should be used whenever a reasoning step requires
    a precise numeric value instead of mental math or approximation.

    The evaluation is sandboxed by strict character whitelisting before calling
    ``eval``. Only characters in the set ``0123456789+-*/(). `` are accepted.
    This means:
        - Digits ``0–9`` for integer or decimal numbers
        - Arithmetic operators ``+``, ``-``, ``*``, ``/``
        - Parentheses ``(``, ``)`` for grouping operations
        - The period ``.`` for decimal notation
        - Whitespace for readability

    Supported syntax:
        - Integers and decimal numbers (e.g., ``2``, ``3.5``)
        - Parentheses for grouping (e.g., ``(2 + 3) * 4``)
        - Chained arithmetic operations
        - Exponentiation using Python syntax ``**`` (e.g., ``45234**(1/2)``)

    Tool-usage guidance:
        - Use this tool whenever an exact arithmetic result is required.
        - Prefer it for multi-step math, chained operations, or intermediate
          values that will be referenced later via tool IDs.
        - Avoid mental approximations when precision matters.

    Args:
        expression: Arithmetic expression string composed only of allowed
            numeric characters, operators, parentheses, decimal points,
            and whitespace.

    Returns:
        String containing either the exact evaluated numeric result.
    """
    # Restrict allowed characters for safety
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


# CALCULATOR_DESCRIPTION: str = (
#     "The calculator tool deterministically evaluates simple arithmetic expressions and "
#     "returns the numeric result as a string. It accepts a single argument, `expression`, "
#     "containing digits, whitespace, parentheses, and operators `+`, `-`, `*`, `/`, and "
#     "exponentiation using Python syntax (e.g., `x**(1/2)`). Invalid characters are "
#     "rejected to prevent unsafe execution. If valid, the expression is evaluated and the "
#     "result returned as a string; otherwise, a clear error message is produced. This tool "
#     "must be used whenever an exact numerical computation is required, especially for "
#     "non-trivial arithmetic, chained operations, or intermediate results referenced "
#     "during reasoning.\n"
# )


CALCULATOR_DESCRIPTION: str = """
Deterministically evaluate a safe arithmetic expression and return its result.

This tool performs exact numerical computation over a restricted subset of
Python arithmetic syntax. It should be used whenever a reasoning step requires
a precise numeric value instead of mental math or approximation.

The evaluation is sandboxed by strict character whitelisting before calling
``eval``. Only characters in the set ``0123456789+-*/(). `` are accepted.
This means:
    - Digits ``0–9`` for integer or decimal numbers
    - Arithmetic operators ``+``, ``-``, ``*``, ``/``
    - Parentheses ``(``, ``)`` for grouping operations
    - The period ``.`` for decimal notation
    - Whitespace for readability

Supported syntax:
    - Integers and decimal numbers (e.g., ``2``, ``3.5``)
    - Parentheses for grouping (e.g., ``(2 + 3) * 4``)
    - Chained arithmetic operations
    - Exponentiation using Python syntax ``**`` (e.g., ``45234**(1/2)``)

Tool-usage guidance:
    - Use this tool whenever an exact arithmetic result is required.
    - Prefer it for multi-step math, chained operations, or intermediate
        values that will be referenced later via tool IDs.
    - Avoid mental approximations when precision matters.

Args:
    expression: Arithmetic expression string composed only of allowed
        numeric characters, operators, parentheses, decimal points,
        and whitespace.

Returns:
    String containing either the exact evaluated numeric result or a
    deterministic error message if validation or evaluation fails.
"""


TOOL_DICT: dict[str, Any] = {
    "calculator": {"function": calculator, "description": CALCULATOR_DESCRIPTION}
}
