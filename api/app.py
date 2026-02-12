import os
import sys

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Añadir el directorio raíz al path para poder importar los módulos de las fases
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = 8182

# --- IMPORTACIONES DE LOS MÓDULOS DE LOS ALUMNOS ---
# TODO: Descomentar a medida que se implementen las fases
from rlm.inference import generate_reasoning, load_rlm_model
from tool_use.tool_handler import parse_and_execute_tool_call

# from rag.rag_engine import retrieve_context, format_rag_prompt
# from react.agent import ReActAgent

app = FastAPI(
    title="Práctica Master: Modelos Generativos Profundos",
    description="API para evaluar las 4 fases de la práctica.",
)

# --- Variables Globales (Modelos) ---
# Se cargan al inicio para no recargarlos en cada petición
MODEL = None
TOKENIZER = None
AGENT = None


@app.on_event("startup")
async def startup_event():
    global MODEL, TOKENIZER, AGENT
    print("Inicializando API...")
    # Cargar el modelo de la Fase 1
    MODEL, TOKENIZER = load_rlm_model()
    # if MODEL:
    #      AGENT = ReActAgent(MODEL, TOKENIZER)
    print("Modelos cargados (RLM).")


# --- Modelos de Pydantic para Request/Response ---
class QueryRequest(BaseModel):
    prompt: str


class GenericResponse(BaseModel):
    response: str
    trace: list[dict] = []
    details: dict = {}


# ================= ENDPOINTS DE EVALUACIÓN =================


# --- FASE 1: Razonamiento (RLM) ---
@app.post(path="/phase1/reasoning", response_model=GenericResponse, tags=["Fase 1"])
async def phase1_endpoint(request: QueryRequest):
    """
    Evalúa el modelo RLM. Debe devolver la respuesta con el razonamiento (CoT) visible.
    """
    if not MODEL or not TOKENIZER:
        return {
            "response": "ERROR: Modelo de Fase 1 no cargado.",
            "details": {"status": "todo"},
        }

    # Usar la función de inferencia de Fase 1
    response_text: str = generate_reasoning(
        prompt=request.prompt, model=MODEL, tokenizer=TOKENIZER
    )
    return {
        "response": response_text,
        "trace": [{"step": 0, "content": response_text}],
        "details": {"stage": "sft_grpo"},
    }


# --- FASE 2: Tool Use ---
@app.post(path="/phase2/tools", response_model=GenericResponse, tags=["Fase 2"])
async def phase2_endpoint(request: QueryRequest):
    """
    Evalúa la capacidad de llamar herramientas.
    Si el prompt requiere una herramienta, debe devolver la ejecución simulada.
    """
    # 1. Simular generación del modelo (o usar el real si ya sabe usar tools)
    # model_output_simulated = '''... Thought: Necesito la calculadora. Action: '''

    # 2. Usar el handler de Fase 2
    # TODO: Descomentar
    # tool_result = parse_and_execute_tool_call(model_output_simulated)

    # tool_result = "Placeholder: Resultado de herramienta (Fase 2) no implementado."

    # if tool_result:
    #     return {"response": f"Tool execution result: {tool_result}", "details": {"tool_called": True}}
    # else:
    #     return {"response": "No tool call detected or needed.", "details": {"tool_called": False}}

    # 1. Verificar que tu modelo local está cargado (reusamos el de la Fase 1)
    if not MODEL or not TOKENIZER:
        return {
            "response": "ERROR: El modelo propio (MODEL/TOKENIZER) no está cargado.",
            "details": {"status": "error_model_not_loaded"},
        }

    try:
        # 2. Construir el Prompt Final
        # Concatenamos las instrucciones de herramientas con la pregunta del usuario.
        # ADAPTALO: Si tu modelo usa un formato especial (ej: <|system|>, [INST], etc.), añádelo aquí.

        final_prompt = None  # Poner el promt de Miguel

        # 3. Generar respuesta con tu modelo local
        # Usamos la misma función de inferencia que en la Fase 1
        # Asegúrate de que generate_reasoning acepte el string completo
        raw_content = generate_reasoning(
            prompt=final_prompt, model=MODEL, tokenizer=TOKENIZER
        )

        # 4. Lógica de Herramientas (Parsing y Ejecución)
        # Tu handler busca el bloque JSON en 'raw_content'
        execution_data = parse_and_execute_tool_call(raw_content)

        # 5. Construir respuesta (Trace y Details)
        trace_data = [
            {
                "step": 0,
                "content": raw_content,
            }  # Lo que generó tu modelo (debería ser el JSON)
        ]

        final_response = execution_data["result"]
        was_tool_used = execution_data["executed"]

        if was_tool_used:
            trace_data.append({
                "step": 1,
                "content": f"Resultado Herramienta: {final_response}",
            })
            # Opcional: Podrías volver a invocar al modelo aquí pasándole el resultado para que redacte una frase final.

        return {
            "response": final_response,  # Devuelve el resultado de la herramienta o el texto generado
            "trace": trace_data,
            "details": {
                "tool_called": was_tool_used,
                "stage": "tool_use_local_model",
                "model_used": "Custom Local Model",
            },
        }

    except Exception as e:
        return {
            "response": f"Error en Fase 2 (Modelo Propio): {e!s}",
            "trace": [],
            "details": {"error": str(e)},
        }


# --- FASE 3: RAG ---
@app.post(path="/phase3/rag", response_model=GenericResponse, tags=["Fase 3"])
async def phase3_endpoint(request: QueryRequest):
    """
    Evalúa el RAG. Debe recuperar contexto de los documentos y responder.
    """
    # TODO: Implementar lógica RAG
    # 1. Recuperar contexto
    # context_list = retrieve_context(request.prompt)
    # 2. Formatear prompt
    # rag_prompt = format_rag_prompt(request.prompt, context_list)
    # 3. Generar con el modelo (opcional, o devolver solo el contexto recuperado para evaluar)

    return {
        "response": "Placeholder Fase 3 (RAG)",
        "details": {"retrieved_docs": ["doc1_placeholder", "doc2_placeholder"]},
    }


# --- FASE 4: Agente ReAct ---
@app.post("/phase4/agent", tags=["Fase 4"])
async def phase4_endpoint(request: QueryRequest):
    """
    Evalúa el agente completo. Devuelve la respuesta final y la traza de ejecución.
    """
    if not AGENT:
        return {"final_answer": "ERROR: Agente no inicializado.", "trace": []}

    # TODO: Ejecutar agente
    # result = AGENT.run(request.prompt)
    result = {
        "final_answer": "Placeholder Fase 4 Agent",
        "trace": [{"step": 0, "content": "..."}],
    }  # TODO remove

    return result


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=PORT)
