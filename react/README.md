# FASE 4: Agente ReAct (Integración Final)

El objetivo es unir todas las piezas (Modelo RLM + Herramientas + RAG) en un bucle autónomo de razonamiento y acción.

## Tareas

Implementa la clase `ReActAgent` en `agent.py`.

El agente debe seguir el ciclo:

1. **Thought (Pensamiento):** Usar el modelo RLM (Fase 1) para analizar la situación actual y decidir qué hacer.
2. **Action (Acción):** Si el modelo decide usar una herramienta, ejecutarla (Fase 2) o consultar el RAG (Fase 3).
3. **Observation (Observación):** Recibir el resultado de la acción.
4. **Repeat:** Añadir la observación al historial y volver al paso 1, hasta que el modelo decida dar una respuesta final (`Final Answer`).

Debes definir un prompt de sistema robusto que explique al agente este protocolo ReAct y las herramientas disponibles.

## Entregables

* Código funcional en `agent.py`.
* El endpoint de la API debe ser capaz de resolver tareas multi-paso complejas. Ejemplo: "¿Cuál es la población de la capital de Francia multiplicada por 0.001?". Esto requiere buscar la capital, buscar su población y usar la calculadora.

# Links de utilidad:

## Function calling vs ReAct:

https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling?hl=es

https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae

## Hub de prompts langsmith

https://smith.langchain.com/hub/

https://smith.langchain.com/hub/hwchase17
