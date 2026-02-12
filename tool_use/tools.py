from collections.abc import Callable

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

"""
otras opciones para agentes:
- from llama_index
    from llama_index.llms import Ollama
    from llama_index.agent import ReActAgent
    from llama_index.tools import FunctionTool
- https://www.tavily.com/
- https://www.langchain.com/langgraph

"""


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión matemática simple."""
    try:
        # Por seguridad, limitamos caracteres
        allowed: set[str] = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed):
            return "Error: Caracteres no permitidos."
        return str(eval(expression))
    except Exception as e:
        return f"Error calculando: {e}"


@tool
def simulated_search(query: str) -> str:
    """
    Busca información en una base de datos.
    SIEMPRE usa esta herramienta para buscar información sobre personas, lugares,
    tecnología o cualquier dato factual. Input: la consulta de búsqueda.
    """
    query_lower: str = query.lower()
    if "hermano" in query_lower and "miguel" in query_lower:
        return "Miguel tiene un hermano llamado Juan."
    elif "capital" in query_lower and "francia" in query_lower:
        return "La capital de Francia es París."
    elif "python" in query_lower:
        return "Python es un lenguaje de programación de alto nivel."
    else:
        return "No se encontraron resultados relevantes en el buscador simulado."


@tool
def get_book_info(title: str) -> str:
    """
    API PÚBLICA (Temática Equipo: Libros).
    Usa Open Library para buscar información.
    """
    try:
        url: str = f"https://openlibrary.org/search.json?q={title.replace(' ', '+')}"
        resp: requests.Response = requests.get(url=url)
        data = resp.json()
        if data.get("numFound", 0) > 0:
            book = data["docs"][0]
            return (
                f"Título: {book.get('title')}, "
                f"Autor: {book.get('author_name', ['?'])[0]}, "
                f"Año: {book.get('first_publish_year')}"
            )
        return "Libro no encontrado."
    except Exception as e:
        return f"Error API: {e}"


tools_map: dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "simulated_search": simulated_search,
    "get_book_info": get_book_info,
}

# 2. JSON Schema definition to instruct the LLM on available tools
AVAILABLE_TOOLS_SCHEMA: list[
    dict[str, str | dict[str, str | dict[str, dict[str, str]] | list[str]]]
] = [
    {
        "name": "calculator",
        "description": (
            "Performs mathematical calculations."
            "Use this for expressions like '2 + 2' or '25 * 4'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "simulated_search",
        "description": (
            "Searches for factual information about "
            "people, places, or tech in a local database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_book_info",
        "description": (
            "Searches for real book information (author, year) "
            "using the Open Library API."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the book."}
            },
            "required": ["title"],
        },
    },
]
