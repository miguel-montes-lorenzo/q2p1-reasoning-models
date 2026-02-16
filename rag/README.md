# FASE 3: Retrieval Augmented Generation (RAG)

El objetivo es conectar el **modelo que entrenaste en Fase 1** a una base de conocimiento externa y usarla como herramienta dentro del flujo del agente.

## Qué hay implementado

### Ingesta (`ingest_data.py`)

* **Carga:** `DirectoryLoader` sobre `./documents` (archivos `.txt`). Se añade metadata `title` a partir del nombre del archivo.
* **Chunking:** `RecursiveCharacterTextSplitter` con `chunk_size=1000` y `chunk_overlap=500`.
* **Embeddings:** modelo `all-MiniLM-L6-v2` de `sentence-transformers` vía `SentenceTransformerEmbeddings`.
* **Vectorstore:** Chroma persistido en `./vectorstore`.

Ejecutar desde el directorio `rag/`:

```bash
python ingest_data.py
```

### Motor RAG (`rag_engine.py`)

* **`retrieve_context(query, k=3)`:** usa `vectordb.similarity_search(query, k=k)` y devuelve los documentos más relevantes.
* **`format_rag_prompt(query, context_list)`:** construye el prompt aumentado inyectando el contexto recuperado para que el modelo responda con esa información.

La base de datos se carga desde `./vectorstore` con el mismo modelo de embeddings.

---

## Tareas para los alumnos

1. **ETL e ingesta (ampliar `ingest_data.py` si hace falta):**

   * **Extract, Transform, Load:** obtener documentos del **tema que hayáis escogido** (scraping de una web, llamada a una API, descarga de PDFs, etc.).
   * Mantener (o ajustar) el chunking y el uso de `all-MiniLM-L6-v2` (u otro de `sentence-transformers`).
   * Persistir en Chroma (u otra BD vectorial local como qdrant) en `./vectorstore`.
2. **Motor RAG (`rag_engine.py`):**

   * Tener operativas `retrieve_context` y la función que formatea el prompt aumentado (p. ej. `format_rag_prompt`). Asegurar que el formato del contexto (lista de strings o de documentos) sea el que espera vuestro flujo.
3. **Usar el modelo entrenado con RAG:**

   * En el endpoint de la API (Fase 3), **responder preguntas usando el modelo que entrenaste en Fase 1 (RLM)**:
     * Llamar a `retrieve_context(pregunta)` para obtener los fragmentos relevantes.
     * Construir el prompt con `format_rag_prompt(pregunta, context_list)` (o equivalente).
     * Generar la respuesta con vuestro modelo (inferencia de Fase 1) sobre ese prompt aumentado.
   * La respuesta debe basarse en el contenido recuperado de vuestra base de conocimiento.
4. **Base de datos vectorial como herramienta (tool):**

   * Exponer la búsqueda sobre la base vectorial como **una tool más** (junto con calculadora, búsqueda simulada, etc. de Fase 2).
   * Definir una herramienta (p. ej. “buscar en base de conocimiento” o “consultar documentos”) que internamente use `retrieve_context` y devuelva el contexto (o un resumen) al agente.
   * Integrar esta tool en `tool_handler` / agente para que, cuando el modelo decida buscar información sobre el tema de los documentos, use esta herramienta en lugar de (o además de) otras fuentes.
5. **Probar la base de datos vectorial:**

   * Incluir pruebas o scripts que **testeen la base de vectores**:
     * Varias preguntas sobre el tema elegido y comprobar que `retrieve_context` devuelve fragmentos relevantes.
     * Opcional: métricas de similitud, ejemplos de buenos/malos retrievals, o un pequeño script de evaluación (p. ej. preguntas con respuesta esperada y comprobar que el chunk correcto está entre los `k` recuperados).

---

## Entregables

* Scripts funcionales de **ingesta** y **recuperación** (y, si aplica, tests de la BD vectorial).
* **Endpoint de la API** (Fase 3) que responda preguntas usando:
  * El **modelo que entrenaste** en Fase 1.
  * El **contexto recuperado** con `retrieve_context` y el prompt formateado con vuestra función RAG.
* La **base de datos vectorial integrada como tool** en el agente/herramientas.
* Documentación o evidencia de que habéis **probado** la base de vectores con preguntas sobre el tema elegido.
