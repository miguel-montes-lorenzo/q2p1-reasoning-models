# api/app.py

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add repo root to path so local packages (rlm, tool_use, rag, react) are importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rlm.config import INFERENCE_CONFIG as NORMAL_INFERENCE_CONFIG
from tool_use.langchain.config import INFERENCE_CONFIG as TOOL_INFERENCE_CONFIG
from tool_use.langchain.config import REACT_INFERENCE_CONFIG
from tool_use.langchain.tool_handler import (
    insert_tool_desciptions_in_system_propt,
)
from tool_use.langchain.tool_inference import run_tool_use_inference
from tool_use.langchain.tools import TOOL_DICT, get_langchain_tools
from rag.rag_engine import retrieve_context, format_rag_prompt
from react.agent import ReActAgent

# --- IMPORTACIONES DE LOS MÓDULOS DE LOS ALUMNOS ---
from rlm.inference import generate_reasoning, load_rlm_model

app: FastAPI = FastAPI(
    title="Práctica Master: Modelos Generativos Profundos",
    description="API para evaluar las 4 fases de la práctica.",
)

PORT: int = 8182
MODEL: Any | None = None
TOKENIZER: Any | None = None
TOOL_CFG: Any | None = None
TOOLS: list[Any] | None = None
AGENT = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON instead of plain 'Internal Server Error' for unhandled exceptions."""
    tb: str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    # Keep it simple + debuggable for your internal setup.
    return JSONResponse(
        status_code=500,
        content={
            "response": "ERROR: Unhandled exception in backend.",
            "trace": [],
            "details": {
                "path": str(request.url.path),
                "error": str(exc),
                "traceback": tb,
            },
        },
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Load models once at startup to avoid reloading per request."""
    global MODEL, TOKENIZER, TOOL_CFG, TOOLS, AGENT

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

    TOOLS = get_langchain_tools()

    # ReAct agent gets its own config with higher iteration limit and multi-step prompt
    from tool_use.langchain.config import REACT_SYSTEM_PROMPT
    react_augmented_system_prompt: str = insert_tool_desciptions_in_system_propt(
        descriptions=descriptions, extra_prompt=REACT_SYSTEM_PROMPT
    )
    react_cfg = replace(
        REACT_INFERENCE_CONFIG(), system_prompt=react_augmented_system_prompt
    )

    AGENT = ReActAgent(
        model=MODEL, tokenizer=TOKENIZER, cfg=react_cfg, tools=TOOLS
    )

    print("Modelos cargados (RLM).")
    print("Tool-use config listo (prompt con descripciones).")
    print("ReAct agent inicializado.")


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
    if MODEL is None or TOKENIZER is None or TOOL_CFG is None or TOOLS is None:
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
        tools=TOOLS,
        formatted_references=True,
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
    """Evaluate phase-3 RAG. Retrieves context, augments prompt, generates with RLM."""
    if MODEL is None or TOKENIZER is None:
        return {
            "response": "ERROR: Modelo de Fase 1 no cargado.",
            "details": {"status": "todo"},
        }

    # Step 1: Retrieve relevant documents
    context_chunks: list[str] = retrieve_context(request.prompt, k=3)

    # Step 2: Build augmented prompt
    augmented_prompt: str = format_rag_prompt(request.prompt, context_chunks)

    # Step 3: Generate reasoning with the RLM model
    response_text: str = generate_reasoning(
        prompt=augmented_prompt,
        model=MODEL,
        tokenizer=TOKENIZER,
        cfg=NORMAL_INFERENCE_CONFIG,
    )

    # Step 4: Build trace for the UI
    context_display: str = (
        "\n\n---\n\n".join(context_chunks) if context_chunks else "(no documents retrieved)"
    )
    trace: list[dict[str, Any]] = [
        {"step": 0, "content": f"<context>{context_display}</context>"},
        {"step": 1, "content": response_text},
    ]

    return {
        "response": response_text,
        "trace": trace,
        "details": {
            "stage": "rag",
            "num_chunks_retrieved": len(context_chunks),
        },
    }


# --- FASE 4: Agente ReAct ---
@app.post(path="/phase4/agent", response_model=GenericResponse, tags=["Fase 4"])
async def phase4_endpoint(request: QueryRequest) -> dict[str, Any]:
    """Evaluate phase-4 ReAct agent. Combines reasoning, tools, and RAG."""
    if AGENT is None:
        return {
            "response": "ERROR: Agente no inicializado.",
            "details": {"status": "todo"},
        }

    result: dict[str, Any] = AGENT.run(request.prompt)
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
