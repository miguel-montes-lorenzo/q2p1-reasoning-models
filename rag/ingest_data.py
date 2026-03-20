from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from repo root when executed as: python rag/ingest_data.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.paths import find_parent_with_markers

REPO_DIR: Path = find_parent_with_markers(start=Path.cwd())
DOCS_DIR: str = str(REPO_DIR / "rag" / "documents")
DB_DIR: str = str(REPO_DIR / "rag" / "vectorstore")


def ingest_documents() -> None:
    import os

    # 1. Load documents
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Add title metadata from filename
    for doc in documents:
        source_path = doc.metadata.get("source", "")
        title = (
            os.path.splitext(os.path.basename(source_path))[0]
            if source_path
            else "Untitled"
        )
        doc.metadata["title"] = title

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    print(f"Generated {len(texts)} chunks.")

    # 3. Embeddings and VectorStore
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=DB_DIR
    )
    vectordb.persist()
    print(f"Vector database created at {DB_DIR}")


if __name__ == "__main__":
    ingest_documents()
