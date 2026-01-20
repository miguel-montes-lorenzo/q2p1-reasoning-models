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
