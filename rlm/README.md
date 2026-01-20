# FASE 1: Entrenamiento de Razonamiento (SFT + GRPO)

El objetivo de esta fase es convertir un modelo de lenguaje genérico (ej. Qwen2.5-7B-Instruct) en un modelo de razonamiento (RLM) capaz de generar cadenas de pensamiento (CoT) coherentes antes de responder.

## Tareas

### Parte 1: Supervised Fine-Tuning (SFT)

Completa el script `train_sft.py`.

* **Objetivo:** Enseñar el formato de salida deseado. Por ejemplo, que el modelo aprenda a encerrar su razonamiento entre tokens especiales como `<think>` y `</think>`.
* **Dataset:** Usa un subconjunto de [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) o [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Debes formatear los datos para que la respuesta incluya el razonamiento explícito.
* **Herramienta:** Puedes usar `trl.SFTTrainer`.

### Parte 2: Reinforcement Learning (GRPO)

Completa el script `train_grpo.py`.

* **Objetivo:** Mejorar la capacidad de razonamiento matemático usando recompensas verificables.
* **Dataset:** Usa las preguntas de GSM8K (sin las respuestas en el prompt).
* **Método (RLVF + GRPO):**
  1. Para cada pregunta, genera N respuestas con sampling (temperatura > 0).
  2. Extrae la respuesta numérica final de cada una.
  3. Compara con la solución real. Recompensa = 1 si es correcta, 0 si no.
  4. Aplica la lógica de GRPO para actualizar los pesos basándote en la ventaja relativa de cada respuesta dentro de su grupo.
* **Nota:** Puedes intentar implementar el bucle customizado o investigar si la librería `trl` ya ofrece soporte experimental para GRPO o similar (ej. PPO sin crítico). Se valorará la implementación manual del bucle de sampling y scoring.

### Entregables

* Los scripts `train_sft.py` y `train_grpo.py` funcionales.
* Ejemplos de prompts para el API endpoint /phase1/reasoning y Endpoint URL de ngrok
* El script `inference.py` debe ser capaz de cargar tu modelo y generar una respuesta con razonamiento visible.
* TBD: Los pesos del adaptador LoRA final guardados en `weights/final_rlm_lora`.
