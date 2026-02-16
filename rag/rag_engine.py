from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "./vectorstore"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Cargar la BD existente
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def retrieve_context(query, k=3, metadata_filter=None):
    """
    Recupera los k documentos más relevantes para la query.
    Si se proporciona un filtro de metadata (dict), los resultados también se filtran por ese criterio.
    Retorna una lista de strings (el contenido de los documentos).
    """
    if metadata_filter:
        # El parametro de filter es un diccionario con formato tipo MONGO QUERY LANGUAGE
        docs = vectordb.similarity_search(query, k=k, filter=metadata_filter)
    else:
        docs = vectordb.similarity_search(query, k=k)
    return docs


def format_rag_prompt(query, context_list):
    """
    Crea el prompt final inyectando el contexto.
    """
    context_str = "\n\n".join(context_list)
    prompt = f"""Usa la siguiente información de contexto para responder a la pregunta del usuario. Si no sabes la respuesta basándote en el contexto, dilo.

Contexto:
{context_str}

Pregunta: {query}
Respuesta:"""
    return prompt


query = "¿Quién es Hermione Granger?"
context = retrieve_context(query, k=3)
print("\n", query)
print(context)


query = "¿Como derrotan a Voldemort?"
context = retrieve_context(query, k=3)
print("\n", query)
print(context)


query = "¿Quién es Kelsier?"
context = retrieve_context(query, k=3)
print("\n", query)
print(context)


query = "¿Quién mata al Lord Legislador?"
context = retrieve_context(query, k=3)
print("\n", query)
print(context)

# RESULTADO CON FILTRO
query = "¿Como derrotan a Voldemort?"
context = retrieve_context(query, k=3, metadata_filter={"title": "harry_potter"})
print("\n", query)
print(context)
