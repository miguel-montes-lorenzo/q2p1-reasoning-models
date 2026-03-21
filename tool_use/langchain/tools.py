# tool_use/langchain/tools.py

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from rag.rag_engine import retrieve_context

load_dotenv()  # Load from current working directory if present

_FDC_BASE_URL: str = "https://api.nal.usda.gov/fdc/v1"


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


def knowledge_base_search(*, query: str) -> str:
    """Search the knowledge base for recipes, cooking techniques, and food information.

    This tool searches a vector database of cookbooks and food documents. It returns
    the most relevant text passages. Use this tool when the question is about recipes,
    cooking methods, ingredients, or food preparation.

    WHEN TO USE:
        - The user asks for a recipe (e.g. "banana bread recipe", "how to make soup").
        - The user asks about cooking techniques or food preparation.
        - The user asks about ingredient lists or cooking instructions.

    WHEN NOT TO USE:
        - The user asks for specific nutrient values (use food_data_central_search).
        - The question is purely mathematical (use calculator).

    Args:
        query: Natural language search query describing the information needed.

    Returns:
        Retrieved text passages from the knowledge base, separated by dividers.
    """
    results: list[str] = retrieve_context(query, k=3)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(results)


_KEY_NUTRIENT_IDS: dict[int, str] = {
    1008: "energy_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    2000: "sugar_g",
    1079: "fiber_g",
    1253: "cholesterol_mg",
    1093: "sodium_mg",
}


def food_data_central_search(*, query: str, page_size: str = "3") -> str:
    """Search USDA FoodData Central for foods and their nutrition data.

    This tool searches the USDA food database and returns matching foods with
    their key nutrients (calories, protein, fat, carbs, sugar, fiber per 100g).

    WHEN TO USE:
        - The user asks about calories, nutrients, or nutritional info of a food.
        - You need to compare nutritional values of different foods.
        - You need a numeric nutrient value to do a calculation.

    WHEN NOT TO USE:
        - The question is about recipes or cooking techniques (use knowledge_base_search).
        - The question is purely mathematical (use calculator).

    Args:
        query: Food search text, e.g. "banana raw", "cheddar cheese", "chicken breast".
        page_size: Number of results (as string). Default "3", max "5".

    Returns:
        JSON with matching foods and their nutrients per 100g.
    """
    api_key: str = os.getenv("FDC_API_KEY", "DEMO_KEY").strip() or "DEMO_KEY"

    if not query.strip():
        raise ValueError("`query` cannot be empty.")

    try:
        parsed_page_size: int = int(page_size)
    except ValueError as exc:
        raise ValueError("`page_size` must be an integer between 1 and 5.") from exc

    parsed_page_size = max(1, min(5, parsed_page_size))

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
        nutrients: dict[str, float] = {}
        for n in food.get("foodNutrients", []):
            nid: int = n.get("nutrientId", 0)
            if nid in _KEY_NUTRIENT_IDS:
                nutrients[_KEY_NUTRIENT_IDS[nid]] = n.get("value", 0)

        foods.append(
            {
                "description": food.get("description"),
                "dataType": food.get("dataType"),
                "nutrients_per_100g": nutrients,
            }
        )

    result: dict[str, Any] = {
        "query": query,
        "total_hits": data.get("totalHits", 0),
        "foods": foods,
    }
    return json.dumps(result, ensure_ascii=True)


TOOL_DICT: dict[str, dict[str, Any]] = {
    "calculator": {
        "function": calculator,
        "description": _docstring_description(fn=calculator),
    },
    "knowledge_base_search": {
        "function": knowledge_base_search,
        "description": _docstring_description(fn=knowledge_base_search),
    },
    "food_data_central_search": {
        "function": food_data_central_search,
        "description": _docstring_description(fn=food_data_central_search),
    },
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
