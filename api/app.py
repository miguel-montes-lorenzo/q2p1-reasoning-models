from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os

# Añadir el directorio raíz al path para poder importar los módulos de las fases
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- IMPORTACIONES DE LOS MÓDULOS DE LOS ALUMNOS ---
# TODO: Descomentar a medida que se implementen las fases
# from rlm.inference import load_rlm_model, generate_reasoning
# from tool_use.tool_handler import parse_and_execute_tool_call
# from rag.rag_engine import retrieve_context, format_rag_prompt
# from react.agent import ReActAgent

app = FastAPI(
    title="Práctica Master: Modelos Generativos Profundos",
    description="API para evaluar las 4 fases de la práctica."
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
    # TODO: Cargar el modelo de la Fase 1 aquí
    # MODEL, TOKENIZER = load_rlm_model()
    # if MODEL:
    #      AGENT = ReActAgent(MODEL, TOKENIZER)
    print("Modelos cargados (PLACEHOLDER).")


# --- Modelos de Pydantic para Request/Response ---
class QueryRequest(BaseModel):
    prompt: str

class GenericResponse(BaseModel):
    response: str
    trace: list[dict] = []
    details: dict = {}


# ================= ENDPOINTS DE EVALUACIÓN =================

# --- FASE 1: Razonamiento (RLM) ---
@app.post("/phase1/reasoning", response_model=GenericResponse, tags=["Fase 1"])
async def phase1_endpoint(request: QueryRequest):
    """
    Evalúa el modelo RLM. Debe devolver la respuesta con el razonamiento (CoT) visible.
    """
    if not MODEL or not TOKENIZER:
        return {"response": "ERROR: Modelo de Fase 1 no cargado.", "details": {"status": "todo"}}
    
    # TODO: Usar la función de inferencia de Fase 1
    # response_text = generate_reasoning(request.prompt, MODEL, TOKENIZER)
    response_text = f"Placeholder Fase 1 para: {request.prompt}" # TODO Remove
    return {
        "response": response_text, "trace": [{"step": 0, "content": response_text}], "details": {"stage": "sft_grpo"}
    }


# --- FASE 2: Tool Use ---
@app.post("/phase2/tools", response_model=GenericResponse, tags=["Fase 2"])
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

    tool_result = "Placeholder: Resultado de herramienta (Fase 2) no implementado."
    
    if tool_result:
        return {"response": f"Tool execution result: {tool_result}", "details": {"tool_called": True}}
    else:
        return {"response": "No tool call detected or needed.", "details": {"tool_called": False}}


# --- FASE 3: RAG ---
@app.post("/phase3/rag", response_model=GenericResponse, tags=["Fase 3"])
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
    
    return {"response": "Placeholder Fase 3 (RAG)", "details": {"retrieved_docs": ["doc1_placeholder", "doc2_placeholder"]}}


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
    result = {"final_answer": "Placeholder Fase 4 Agent", "trace": [{"step": 0, "content": "..."}]} # TODO remove

    return result

if __name__ == "__main__":
    # Para correr localmente: python api/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
