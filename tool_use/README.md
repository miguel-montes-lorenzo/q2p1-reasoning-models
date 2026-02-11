# FASE 2: Uso de Herramientas (Function Calling)

El objetivo de esta fase es dotar al modelo RLM de la Fase 1 de la capacidad de usar herramientas externas. No es necesario re-entrenar el modelo si este ya es bueno siguiendo instrucciones complejas.

## Tareas

1. **Definir Herramientas (`tools.py`):** Implementa versiones simples de herramientas (una calculadora y un "buscador" simulado que devuelva strings fijos). Define también sus esquemas JSON (nombre, descripción, argumentos). UTILIZAR APIS: https://github.com/public-apis/public-apis?tab=readme-ov-file
2. **Manejador de Herramientas (`tool_handler.py`):**

   * Crea un system prompt que explique al modelo qué herramientas tiene disponibles y en qué formato JSON debe llamarlas.
   * Implementa una función que, dado el output del modelo, detecte si hay una llamada a herramienta (ej. buscando un bloque JSON específico), parsee los argumentos y ejecute la función correspondiente de `tools.py`.

## Entregables

* Código funcional en `tools.py` y `tool_handler.py`.
* El endpoint de la API para esta fase debe ser capaz de recibir una pregunta como "Calcula la raíz cuadrada de 2543" y devolver la intención de llamada a herramienta en JSON.
* El endpoint de la APi para esta fase deberá ser capaz de utilizar las herramientas de la temática de tu equipo.
