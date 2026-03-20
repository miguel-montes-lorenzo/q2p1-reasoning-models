from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from utils.paths import find_parent_with_markers

REPO_DIR: Path = find_parent_with_markers(start=Path.cwd())
DB_DIR: str = str(REPO_DIR / "rag" / "vectorstore")

_vectordb: Chroma | None = None
_embeddings: SentenceTransformerEmbeddings | None = None


def _get_vectordb() -> Chroma:
    """Lazy-load the Chroma vector store (only created on first call)."""
    global _vectordb, _embeddings
    if _vectordb is None:
        _embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectordb = Chroma(persist_directory=DB_DIR, embedding_function=_embeddings)
    return _vectordb


def retrieve_context(
    query: str, k: int = 3, metadata_filter: dict | None = None
) -> list[str]:
    """Retrieve the k most relevant document chunks for a query.

    Args:
        query: Search query string.
        k: Number of chunks to retrieve.
        metadata_filter: Optional metadata filter dict (Mongo-style).

    Returns:
        List of document content strings.
    """
    vectordb = _get_vectordb()
    if metadata_filter:
        docs = vectordb.similarity_search(query, k=k, filter=metadata_filter)
    else:
        docs = vectordb.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def format_rag_prompt(query: str, context_list: list[str]) -> str:
    """Build an augmented prompt by injecting retrieved context.

    Args:
        query: The user's question.
        context_list: List of relevant text chunks.

    Returns:
        A prompt string with context and question ready for the model.
    """
    context_str = "\n\n".join(context_list)
    return (
        "Use the following context to answer the user's question. "
        "If you cannot answer based on the context, say so.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}"
    )


if __name__ == "__main__":
    query = "Who is Hermione Granger?"
    context = retrieve_context(query, k=3)
    print("\n", query)
    print(context)

    query = "Who is Kelsier?"
    context = retrieve_context(query, k=3)
    print("\n", query)
    print(context)
