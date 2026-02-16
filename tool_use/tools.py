# --- Implementación de las herramientas ---

from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Intenta cargar desde el directorio de trabajo actual


def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression.

    Args:
        expression: Arithmetic expression containing digits and basic operators.

    Returns:
        Result of the evaluated expression as a string, or an error message.
    """
    try:
        # Restrict allowed characters for safety
        allowed: set[str] = set("0123456789+-*/(). ")
        disallowed: set[str] = set(expression) - allowed
        if disallowed:
            invalid_char: str = sorted(disallowed)[0]
            return f"Error: Disallowed character '{invalid_char}'."

        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"


CALCULATOR_DESCRIPTION: str = (
    "The calculator tool deterministically evaluates simple arithmetic expressions and "
    "returns the numeric result as a string. It accepts a single argument, `expression`, "
    "containing digits, whitespace, parentheses, and operators `+`, `-`, `*`, `/`, and "
    "exponentiation using Python syntax (e.g., `x**(1/2)`). Invalid characters are "
    "rejected to prevent unsafe execution. If valid, the expression is evaluated and the "
    "result returned as a string; otherwise, a clear error message is produced. This tool "
    "must be used whenever an exact numerical computation is required, especially for "
    "non-trivial arithmetic, chained operations, or intermediate results referenced "
    "during reasoning.\n"
)


TOOL_DICT: dict[str, Any] = {
    "calculator": {"function": calculator, "description": CALCULATOR_DESCRIPTION}
}
