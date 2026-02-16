from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
pip install chromadb
pip install sentence-transformers
pip install langchain-text-splitters
pip install langchain

Herramientas para scraping:
- BeautifulSoup
- Selenium Base
"""
DOCS_DIR = "./documents"
DB_DIR = "./vectorstore"


def ingest_documents():
    import os

    # 1. Cargar documentos
    # TODO: Configurar loaders para PDF si es necesario
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Cargados {len(documents)} documentos.")

    # Añadir metadata extra: título del documento (=nombre del archivo base)
    # DirectoryLoader con TextLoader suele incluir 'source' en metadata, que es la ruta.
    # Añadimos un campo 'title' basado en el nombre del archivo.
    for idx, doc in enumerate(documents):
        source_path = doc.metadata.get("source", "")
        title = (
            os.path.splitext(os.path.basename(source_path))[0]
            if source_path
            else "Untitled"
        )
        doc.metadata["title"] = title

    # 2. Splitter (Chunking)
    # TODO: Ajustar chunk_size y overlap según los documentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    print(f"Generados {len(texts)} chunks.")

    # 3. Embeddings y VectorStore
    # Usamos un modelo ligero para CPU
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Crear y persistir la base de datos Chroma
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=DB_DIR
    )
    vectordb.persist()
    print(f"Base de datos vectorial creada en {DB_DIR}")


if __name__ == "__main__":
    ingest_documents()
