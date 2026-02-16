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




## Notas propias


Nunca puede haber una respuesta final <answer>...</answer> vacía o null, salvo que el modelo las genere (cosa poco probable). Lo que tienes que hacer es:
1. identificar todas las etiquetas "padre" (que no están dentro de ninguna otra) que son las <question> ... </question>, las <think>...</think>, las <tools> .... </tools> y las <answer> ... </answer>
(Todas éstas, salvo la primera y la última, sólo son válidas si se encuentran al lado de una complementaria. e.d. si es una de apertura, al lado de una de cierre y viceversa. Todas las etiquetas "padre" que no sean válidas serán ignoradas y no supondrán ningún error. e.d. si hay etiquetas <think> o <answer> dentro de otras válidas simplemente se interpretarán como parte del razoanmiento o la respuesta)

2. Se identificarán todas las etiquetas <tool ...> y <id ...> dentro de los bloques de <think> y <answer>. Si hay alguna que no tenga un formato válido. Entonces en ese mismo momento se dejará de analizar el texto, se añadirá a la cadena de respuesta la sección <think>...</think> y se forzará una respuesta final <answer>...</answer> conteniendo algún error del tipo de:
	Error: bad format for <tool ...> tag (make sure to not detect false positive due to regex < or > characters)
	Error: bad format for <id ...>
	Error: <id ...> tag references unidentified result
	Error: content outside of any tag section
	Error: <tool ...> tag not allowed inside <answer> tag

3. Si no se produce ningún error:
- Se guardará una versión de la respuesta sin formatear ningún <id ...> ni evaluar ninguna <tool ...>
- Se construirá un grafo de dependencias tool -> id -> tool -> id -> ... 
- Si no hay dependencias circulares, se evaluará todo el grafo dando lugar a una sección de <tools>...</tools>

4. Si no se ha emitido un tag section de <answer>...</answer> y no se ha alcanzado un límite de iteraciones, se volverá a llamar al modelo concatenando al prompt anterior (inicialmente solo la sección <question>...</question>), el proceso <think>...<think> (sin ningún formateo de <tool ...>s o <id ...>s) seguido de una sección de <tools>...</tools> (con su apropiada estructura de json/diccionario) en la que sí se formatearán todos los valores de <id ...> (para eso se ha construido previamente el grafo de dependencias) permitiendo realizar operaciones secuenciales si se han planificado bien durante el <think> ... </think>.

5. Si se ha emitido un tag section de <answer>...</answer>, se evaluarán de la misma forma las tools, se añadirá la sección de <tools>...</tools> después de <think>...</think> y luego. en lugar de llamarse de nuevo al modelo concatenando <think>...</think> y <tools>...</tools> simplemente se añadirá la sección de <answer>...</answer> (evaluando todos los tag <id ...> que haya dentro)

Nota. véase que no tiene ningún sentido utilizar etiquetas <id ...> fuera de los argumentos de tools o del contenido de <answer>...</answer>, ya que no serán formateadas en ningún momento y no contribuirá a acelerar el razoanamiento en ninguna medida


NOTA ADICIONAL.
En el fichero de answers/tool_use.json con formato:
  {
    "question": ...,
    "base_output": ...,
    "parsed_base_output": ...,
    "best_checkpoint_output": ...,
    "parsed_best_checkpoint_output": ...
  },
las entradas no parsed_... deben contener la cadena think-tools-answer completa con la sección de answer sin formatear los tags <id ...>
las entradas sí parsed_... sólo deben contener la sección de answer y aquí sí, con los tags <id ...> formateados

