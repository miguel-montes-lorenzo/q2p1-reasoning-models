# --- Implementación de las herramientas ---

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

load_dotenv()  # Intenta cargar desde el directorio de trabajo actual

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
    """Evalúa una expresión matemática simple. Útil para realizar cálculos aritméticos."""
    try:
        # Idealmente, usar una librería segura como numexpr o asteval.
        return str(eval(expression))
    except Exception as e:
        return f"Error calculando: {e}"


@tool
def simulated_search(query: str) -> str:
    """Busca información en una base de datos. SIEMPRE usa esta herramienta para buscar información sobre personas, lugares, tecnología o cualquier dato factual. Input: la consulta de búsqueda."""
    query_lower = query.lower()
    if "hermano" in query_lower and "miguel" in query_lower:
        return "Miguel tiene un hermano llamado Juan."
    elif "capital" in query_lower and "francia" in query_lower:
        return "La capital de Francia es París."
    elif "python" in query_lower:
        return "Python es un lenguaje de programación de alto nivel."
    else:
        return "No se encontraron resultados relevantes en el buscador simulado."


# Lista de herramientas disponibles
tools = [calculator, simulated_search]

def get_azure_model():
    """Crea y retorna el modelo de Azure OpenAI con validación de credenciales."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        api_key=api_key,
    )


SYSTEM_PROMPT = """Eres un asistente que SIEMPRE usa las herramientas disponibles para responder preguntas.

REGLAS IMPORTANTES:
1. Para CUALQUIER pregunta sobre personas, lugares, datos o hechos, USA la herramienta simulated_search PRIMERO.
2. Para cálculos matemáticos, USA la herramienta calculator.
3. NUNCA respondas basándote en tu conocimiento propio sin antes consultar las herramientas.
4. Si una herramienta no devuelve resultados, entonces puedes indicar que no encontraste la información."""


def main():
    """Ejemplo de uso de un agente con herramientas usando LangChain y Azure OpenAI."""
    
    # Crear el modelo de Azure OpenAI
    azure_model = get_azure_model()
    
    # Crear el agente con el modelo de Azure y system prompt
    agent = create_agent(
        model=azure_model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )
    
    # # Ejemplo 1: Pregunta que requiere cálculo
    print("=" * 60)
    print("Ejemplo 1: Cálculo matemático")
    print("=" * 60)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "¿Cuánto es 25 * 4 + 100?"}]
    })
    print(f"Respuesta: {result['messages'][-1].content}\n")
    
    # Ejemplo 2: Pregunta que requiere búsqueda
    print("=" * 60)
    print("Ejemplo 2: Búsqueda de información")
    print("=" * 60)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "¿Quién es el hermano de Miguel?"}]
    })
    print(f"Respuesta: {result['messages'][-1].content}\n")
    


if __name__ == "__main__":
    main()
