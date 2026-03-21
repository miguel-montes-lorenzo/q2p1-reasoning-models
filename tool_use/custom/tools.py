# tool_use/tools.py

import inspect
import json
import os
from typing import Any
from urllib import error, parse, request

from dotenv import load_dotenv

load_dotenv()  # Intenta cargar desde el directorio de trabajo actual

_FDC_BASE_URL: str = "https://api.nal.usda.gov/fdc/v1"


def food_data_central_search(query: str, page_size: str = "5") -> str:
    """Search USDA FoodData Central and return a compact list of matching foods.

    This tool calls the FoodData Central `/foods/search` endpoint to find foods
    matching a free-text query. It is useful for nutrition-related questions,
    ingredient lookup, or identifying exact branded/foundation food entries.

    Authentication:
        - Requires environment variable `FDC_API_KEY`.
        - If not set, the tool automatically falls back to `DEMO_KEY`.

    Args:
        query: Search text such as "cheddar cheese" or "banana".
        page_size: Number of results to return, as string. Valid range is 1-25.

    Returns:
        JSON string with:
            - query
            - total_hits
            - foods: list of compact food records with `fdcId`, `description`,
              `dataType`, and `brandName` (when available).
    """
    api_key: str = os.getenv("FDC_API_KEY", "DEMO_KEY").strip() or "DEMO_KEY"

    if not query.strip():
        raise ValueError("`query` cannot be empty.")

    try:
        parsed_page_size: int = int(page_size)
    except ValueError as exc:
        raise ValueError("`page_size` must be an integer between 1 and 25.") from exc

    parsed_page_size = max(1, min(25, parsed_page_size))

    url: str = f"{_FDC_BASE_URL}/foods/search?api_key={parse.quote(api_key)}"
    payload: dict[str, Any] = {"query": query, "pageSize": parsed_page_size}
    body: bytes = json.dumps(payload).encode("utf-8")

    req: request.Request = request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=20) as resp:
            raw: bytes = resp.read()
    except error.HTTPError as exc:
        details: str = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FoodData Central HTTP {exc.code}: {details[:300]}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"FoodData Central connection error: {exc.reason}") from exc

    data: dict[str, Any] = json.loads(raw.decode("utf-8"))
    foods: list[dict[str, Any]] = []
    for food in data.get("foods", []):
        foods.append(
            {
                "fdcId": food.get("fdcId"),
                "description": food.get("description"),
                "dataType": food.get("dataType"),
                "brandName": food.get("brandName"),
            }
        )

    result: dict[str, Any] = {
        "query": query,
        "total_hits": data.get("totalHits", 0),
        "foods": foods,
    }
    return json.dumps(result, ensure_ascii=True)


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
    "calculator": {"function": calculator, "description": CALCULATOR_DESCRIPTION},
    "food_data_central_search": {
        "function": food_data_central_search,
        "description": inspect.getdoc(food_data_central_search) or "",
    },
}
