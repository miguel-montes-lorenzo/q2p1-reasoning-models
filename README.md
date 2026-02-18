# Prácticas MIA 2o Cuatrimestre

Este repositorio contiene el esqueleto para la práctica final del máster. El objetivo es construir, paso a paso, un agente de IA autónomo capaz de razonar, usar herramientas y consultar documentación externa, partiendo de un modelo de lenguaje base.

## Estructura de la Práctica

La práctica se divide en 4 fases acumulativas. Cada fase tiene su propio directorio con instrucciones específicas (README.md) y código base.

* **FASE 1 (`rlm`): De LM a RLM (Reasoning Language Model).**
  * Entrenamiento supervisado (SFT) para seguir instrucciones y formato de pensamiento.
  * Alineación con Aprendizaje por Refuerzo usando GRPO (Group Relative Policy Optimization).
* **FASE 2 (`tool_use`): Uso de Herramientas.**
  * Dotar al modelo de la capacidad de invocar funciones externas (calculadora, búsqueda).
* **FASE 3 (`rag`): RAG (Retrieval Augmented Generation).**
  * Conectar el modelo a una base de conocimiento documental privada.
* **FASE 4 (`react`): Agente ReAct.**
  * Integrar todo en un bucle autónomo de Razonamiento-Acción-Observación.

## Evaluación

La evaluación se realizará exponiendo la funcionalidad de cada fase a través de una API REST.

1. Debes completar el código en cada carpeta de fase.
2. Debes conectar tus implementaciones en el archivo `api/app.py`.
3. Para la entrega, levantarás la API y usarás `ngrok` (en caso de levantar la API localmente) para dar acceso al profesor a los endpoints.

## Links importantes

APIs

- https://aws.amazon.com/what-is/api/#:~:text=API%20stands%20for%20Application%20Programming,other%20using%20requests%20and%20responses.
- https://fastapi.tiangolo.com/features/#editor-support
- https://github.com/public-apis/public-apis?tab=readme-ov-file
- 

ngrok despliegue

- https://ngrok.com/
- https://ngrok.com/docs/api
- https://ngrok.com/docs/universal-gateway/agent-endpoints
- 

Pydantic y Structured Outputs

- https://docs.pydantic.dev/latest/why/#type-hints
- https://medium.com/@speaktoharisudhan/structured-outputs-from-llm-using-pydantic-1a36e6c3aa07
- https://medium.com/@adkananthi/one-framework-two-worlds-achieving-structured-outputs-for-llms-and-vlms-with-transformer-outlines-ae2eec6eb3fc
- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/10
- https://docs.langchain.com/oss/python/langchain/structured-output
- 


## Glosario y Conceptos Clave

* **CoT (Chain of Thought):** Técnica de prompting o entrenamiento donde el modelo genera pasos intermedios de razonamiento antes de dar la respuesta final.
* **SFT (Supervised Fine-Tuning):** Ajuste fino clásico usando pares de (instrucción, respuesta deseada).
* **RLVF (Reinforcement Learning with Verification Feedback):** Una variante de RLHF donde la recompensa no la dan humanos, sino un sistema verificador determinista (ej. ejecutar código y ver si funciona, o comprobar si una solución matemática es correcta).
* **GRPO (Group Relative Policy Optimization):** Un algoritmo de RL eficiente. En lugar de usar un modelo "crítico" para estimar el valor de una acción (lo que consume mucha memoria), GRPO muestrea un grupo de respuestas (ej. 8) para la misma pregunta. Calcula la recompensa de cada una y normaliza las puntuaciones basándose en la media de ese grupo. Las respuestas mejores que la media del grupo se refuerzan positivo, las peores negativo.
* **ReAct (Reason + Act):** Un paradigma para agentes donde el modelo alterna entre generar pensamientos verbales y generar acciones (llamadas a herramientas).


## Instrucciones útiles para el manejo del repo

Crear el entorno de python (uv-managed):

```bash
venv
```

Instalar dependencias de python

```bash
uv sync
```

Descargar modelo preentrenado:
```bash
python -m api.cache_model_assets
```

Levantar la api de ngrok:
```bash
cd api/
chmod +x ./up.sh && ./up.sh
chmod +x ./attach.sh && ./attach.sh  # esto abre una terminal dentro del contenedor
launch_ngrok  # esto pide el authtoken de ngrok y a continuación levanta el servicio
# Ctrl + C para terminar el servicio
```

Probar la api de ngrok a través de curl
```bash
# (en una nueva terminal)
curl -X POST "https://unrefractory-ella-overbulky.ngrok-free.dev/phase1/reasoning" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2 + 2?"}'
```

Acceder a la UI web:
- Copiar la URL de ./api/ngrok-url
- Pegarla en el navegador




Here is a clean README section you can paste directly.

---

## Repository Structure Overview

This repository implements a multi-stage reasoning system built around **Reasoning Language Models (RLM)**, **tool-use augmentation**, and an **API interface** for evaluation and interaction.
Below is a concise description of the most relevant modules and files.

---

### `rlm/` — Reasoning Language Model pipeline

Core training and inference logic for the base reasoning model without external tools.

**Key files**

* `config.py`
  Central configuration for training and inference (model name, decoding params, paths, LoRA settings, etc.).

* `train_sft.py`
  Supervised fine-tuning (SFT) on GSM8K-style reasoning traces.

* `train_grpo.py`
  GRPO reinforcement-style optimization stage that improves reasoning quality after SFT.

* `inference.py`
  Loads the trained LoRA adapter and generates reasoning traces for a prompt.

* `check_answers.py`
  Evaluation script comparing model outputs against QA benchmarks.

**Purpose**

This module provides the **baseline reasoning capability** used later by tool-use and API layers.

---

### `tool_use/` — Tool-augmented reasoning

Implements structured tool calling, execution, and training of models that can reason **with external tools**.

There are two implementations:

* `tool_use/langchain/` → production implementation using **LangChain tools**
* `tool_use/custom/` → reference / minimal custom implementation

#### Core concepts

* Structured reasoning format using:

  * `<think>` internal reasoning
  * `<tools>` execution trace
  * `<answer>` final validated output
* Deterministic **tool execution loop**
* Support for **multi-step reasoning with tool dependencies**

#### Important files (LangChain version)

* `config.py`
  Tool-use system prompt, decoding settings, and inference configuration.

* `tools.py`
  Registry of available tools (e.g., calculator) and their metadata.

* `tool_handler.py`
  **Critical component** that:

  * Parses model outputs
  * Validates tool calls and IDs
  * Executes tools
  * Formats `<tools>` blocks
  * Produces the final `<answer>`

* `tool_inference.py`
  Implements the **tool-execution loop** via:

  * HuggingFace chat wrapper
  * Iterative reasoning + tool execution
  * Final answer validation

* `train_tool_sft.py`
  Generates **tool-use transcripts** and performs LoRA SFT so the model learns to:

  * call tools correctly
  * reference tool outputs
  * produce validated answers

* `check_tool_answers.py`
  Benchmark evaluation for tool-augmented reasoning.

**Purpose**

This module upgrades the base RLM into a **reasoning agent capable of external computation and structured multi-step reasoning**.

---

### `api/app.py` — Unified inference API

FastAPI service exposing the different reasoning stages.

**Endpoints**

* `/phase1/reasoning`
  Base RLM reasoning without tools.

* `/phase2/tools`
  Tool-augmented reasoning using the full **tool-execution loop**.

* `/phase3/rag`
  Placeholder for retrieval-augmented generation.

* `/phase4/agent`
  Placeholder for ReAct-style agent orchestration.

**Startup behavior**

* Loads the RLM model and tokenizer once.
* Injects **tool descriptions into the system prompt**.
* Initializes LangChain tools.
* Serves both:

  * JSON API
  * Static web UI (`web/`).

**Role in architecture**

This file is the **runtime entry point** that connects:

```
RLM  →  Tool Use  →  API  →  Web UI
```

---

### Supporting directories (brief)

* `QA/` — Benchmark questions and expected answers.
* `rag/` — Retrieval pipeline and vector store utilities.
* `utils/` — Shared helpers (paths, LangChain utilities).
* `web/` — Minimal frontend for interactive testing.
* `weights/` — Stored LoRA checkpoints for RLM and tool-use models.

---