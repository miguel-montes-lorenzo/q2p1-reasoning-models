import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rlm.config import INFERENCE_CONFIG as NORMAL_INFERENCE_CONFIG
from tool_use.config import INFERENCE_CONFIG as TOOL_INFERENCE_CONFIG
from tool_use.tool_handler import (
    ensure_response_contains_answer,
    insert_tool_desciptions_in_system_propt,
)
from tool_use.tools import TOOL_DICT

# Añadir el directorio raíz al path para poder importar los módulos de las fases
sys.path.append(os.path.dirname(os.path.dirname(p=os.path.abspath(path=__file__))))

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
async def startup_event() -> None:
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
async def phase1_endpoint(request: QueryRequest) -> dict[str, Any]:
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
    """Evaluate tool-use ability with an iterative tool loop.

    This endpoint:
      1) Builds a tool-augmented system prompt (calculator description injected).
      2) Generates an assistant response.
      3) If the assistant emits <tool ...> tags (and no <answer> yet), executes tools
         and appends the required "<think>...</think>\n<tools>...</tools>" block as
         a new user message, then generates again.
      4) Stops when an <answer> is produced or when max_calls is reached.
      5) Ensures the final assistant message contains <answer>...</answer>.

    Returns:
        GenericResponse-compatible dict with response, trace, and details.
    """
    if not MODEL or not TOKENIZER:
        return {
            "response": "ERROR: Modelo de Fase 2 no cargado.",
            "details": {"status": "todo"},
        }

    cfg: Any = TOOL_INFERENCE_CONFIG()

    descriptions: dict[str, str] = {
        tool_name: str(tool_meta["description"])
        for tool_name, tool_meta in TOOL_DICT.items()
    }
    system_prompt_with_tools: str = insert_tool_desciptions_in_system_propt(
        descriptions=descriptions
    )

    prompt: str = str(request.prompt)
    has_question_tags: bool = (
        re.search(pattern=r"<question>.*?</question>", string=prompt, flags=re.DOTALL)
        is not None
    )
    wrapped_question: str = (
        prompt if has_question_tags else f"<question>{prompt}</question>"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt_with_tools},
        {"role": "user", "content": wrapped_question},
    ]

    trace: list[dict[str, Any]] = []
    tool_calls_used: int = 0

    def generate_from_messages(*, msgs: list[dict[str, str]]) -> str:
        """Generate a single assistant completion from the current chat state."""
        text: str = TOKENIZER.apply_chat_template(
            conversation=msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs: Any = TOKENIZER(text, return_tensors="pt")
        device: torch.device = next(MODEL.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(cfg.max_new_tokens),
            "do_sample": bool(cfg.do_sample),
            "eos_token_id": TOKENIZER.eos_token_id,
            "pad_token_id": TOKENIZER.pad_token_id,
        }
        if cfg.temperature is not None:
            gen_kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_p is not None:
            gen_kwargs["top_p"] = float(cfg.top_p)

        with torch.no_grad():
            out: Any = MODEL.generate(**inputs, **gen_kwargs)

        prompt_len: int = int(inputs["input_ids"].shape[-1])
        generated_ids: Any = out[0][prompt_len:]
        return TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()

    while True:
        assistant_text: str = generate_from_messages(msgs=messages)
        trace.append({
            "step": len(trace),
            "role": "assistant",
            "content": assistant_text,
        })
        messages.append({"role": "assistant", "content": assistant_text})

        has_answer: bool = (
            re.search(
                pattern=r"<answer>.*?</answer>",
                string=assistant_text,
                flags=re.DOTALL,
            )
            is not None
        )
        if has_answer:
            break

        if tool_calls_used >= int(cfg.max_calls):
            break

        should_continue: bool
        prompt_appendix: str
        should_continue, prompt_appendix = parse_and_execute_tool_call(
            model_output=assistant_text,
            tool_dict=TOOL_DICT,
            max_calls=int(cfg.max_calls),
        )

        if not should_continue:
            break

        tool_calls_used += 1
        trace.append({
            "step": len(trace),
            "role": "user",
            "content": prompt_appendix,
            "type": "tools_evaluation",
        })
        messages.append({"role": "user", "content": prompt_appendix})

    # Enforce <answer> presence in the final assistant message.
    last_assistant: str = str(messages[-1]["content"])
    fixed_last_assistant: str = ensure_response_contains_answer(
        full_prompt=last_assistant
    )
    if fixed_last_assistant != last_assistant:
        messages[-1]["content"] = fixed_last_assistant
        trace.append({
            "step": len(trace),
            "role": "assistant",
            "content": fixed_last_assistant,
            "type": "answer_enforced",
        })

    return {
        "response": str(messages[-1]["content"]),
        "trace": trace,
        "details": {"stage": "tool_use", "tool_calls_used": tool_calls_used},
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

# WEB_PUBLIC_DIR = Path(os.environ.get("WEB_PUBLIC_DIR", default="/home/root/web/public"))
# INDEX_HTML: Path = WEB_PUBLIC_DIR / "index.html"


# # Sirve estáticos en /static (NO en /)
# app.mount(
#     path="/static",
#     app=StaticFiles(directory=str(WEB_PUBLIC_DIR), html=False),
#     name="static",
# )


# @app.get("/")
# async def web_index() -> FileResponse:
#     return FileResponse(path=str(INDEX_HTML))


# # IMPORTANTE: montar estáticos al final para no “pisar” /phase* ni /api/*
# app.mount(
#     path="/",
#     app=StaticFiles(directory=str(WEB_PUBLIC_DIR), html=True),
#     name="web",
# )


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
