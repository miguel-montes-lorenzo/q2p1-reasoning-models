# api/app.py

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rlm.config import INFERENCE_CONFIG as NORMAL_INFERENCE_CONFIG
from tool_use.config import INFERENCE_CONFIG as TOOL_INFERENCE_CONFIG
from tool_use.tool_handler import (
    insert_tool_desciptions_in_system_propt,
)
from tool_use.tool_inference import run_tool_use_inference
from tool_use.tools import TOOL_DICT

# Add repo root to path so we can import phase modules
sys.path.append(os.path.dirname(os.path.dirname(p=os.path.abspath(path=__file__))))


# --- IMPORTACIONES DE LOS MÓDULOS DE LOS ALUMNOS ---
# TODO: Descomentar a medida que se implementen las fases
from rlm.inference import generate_reasoning, load_rlm_model

# from rag.rag_engine import retrieve_context, format_rag_prompt
# from react.agent import ReActAgent

# --- STUDENT PHASE MODULE IMPORTS ---


app: FastAPI = FastAPI(
    title="Práctica Master: Modelos Generativos Profundos",
    description="API para evaluar las 4 fases de la práctica.",
)

PORT: int = 8182
MODEL: Any | None = None
TOKENIZER: Any | None = None
TOOL_CFG: Any | None = None
AGENT = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load models once at startup to avoid reloading per request."""
    global MODEL, TOKENIZER, TOOL_CFG

    print("Inicializando API...")
    MODEL, TOKENIZER = load_rlm_model()

    descriptions: dict[str, str] = {
        tool_name: str(tool_meta["description"])
        for tool_name, tool_meta in TOOL_DICT.items()
    }
    tool_augmented_system_prompt: str = insert_tool_desciptions_in_system_propt(
        descriptions=descriptions
    )
    TOOL_CFG = replace(
        TOOL_INFERENCE_CONFIG(), system_prompt=tool_augmented_system_prompt
    )

    print("Modelos cargados (RLM).")
    print("Tool-use config listo (prompt con descripciones).")


class QueryRequest(BaseModel):
    prompt: str


class GenericResponse(BaseModel):
    response: str
    trace: list[dict] = []
    details: dict = {}


# --- FASE 1: Tool Use ---
@app.post(path="/phase1/reasoning", response_model=GenericResponse, tags=["Fase 1"])
async def phase1_endpoint(request: QueryRequest) -> dict[str, Any]:
    """Evaluate phase-1 RLM. Returns response with visible CoT."""
    if MODEL is None or TOKENIZER is None:
        return {
            "response": "ERROR: Modelo de Fase 1 no cargado.",
            "details": {"status": "todo"},
        }

    response_text: str = generate_reasoning(
        prompt=request.prompt,
        model=MODEL,
        tokenizer=TOKENIZER,
        cfg=NORMAL_INFERENCE_CONFIG,
    )
    return {
        "response": response_text,
        "trace": [{"step": 0, "content": response_text}],
        "details": {"stage": "sft_grpo"},
    }


# --- FASE 2: Tool Use ---
@app.post(path="/phase2/tools", response_model=GenericResponse, tags=["Fase 2"])
async def phase2_endpoint(request: QueryRequest) -> dict[str, Any]:
    """Evalúa tool-use usando el tool loop + tool_handler."""
    if MODEL is None or TOKENIZER is None or TOOL_CFG is None:
        return {
            "response": "ERROR: Modelo/Tokenizer/Tool cfg de Fase 2 no cargado.",
            "details": {"status": "todo"},
        }

    full_output: str
    parsed_answer_only: str
    step_contents: list[str]
    full_output, parsed_answer_only, step_contents = run_tool_use_inference(
        question=request.prompt,
        model=MODEL,
        tokenizer=TOKENIZER,
        cfg=TOOL_CFG,
        tool_dict=TOOL_DICT,
    )

    trace: list[dict[str, Any]] = [
        {"step": i, "content": content}
        for i, content in enumerate(iterable=step_contents)
    ]

    response_text: str = parsed_answer_only
    return {
        "response": response_text,
        "trace": trace,
        "details": {"stage": "tool use"},
    }


# --- FASE 3: RAG ---
@app.post(path="/phase3/rag", response_model=GenericResponse, tags=["Fase 3"])
async def phase3_endpoint(request: QueryRequest) -> dict[str, Any]:
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
async def phase4_endpoint(request: QueryRequest) -> dict[str, Any]:
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


# --- Web UI served by FastAPI ---

WEB_PUBLIC_DIR: Path = Path(os.environ.get("WEB_PUBLIC_DIR", default="/home/root/web"))
INDEX_HTML: Path = WEB_PUBLIC_DIR / "index.html"

app.mount(
    path="/static",
    app=StaticFiles(directory=str(WEB_PUBLIC_DIR), html=False),
    name="static",
)


@app.get("/")
async def web_index() -> FileResponse:
    return FileResponse(path=str(INDEX_HTML))


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=PORT)
