import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from src.core import config


def create_retriever(k: int = config.TOP_K_RETRIEVED_DOCS) -> BaseRetriever:
    """
    Loads the pre-built FAISS index from disk and creates a vector store retriever.

    Args:
        k: The number of top documents to retrieve.

    Returns:
        The configured retriever.
    """
    if not os.path.exists(config.VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{config.VECTOR_STORE_PATH}'. "
            "Please run the build script first: 'PYTHONPATH=$PYTHONPATH:. python src/scripts/01_build_vector_store.py'"
        )

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    vectorstore = FAISS.load_local(
        folder_path=config.VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Necessary for loading FAISS indexes created with older versions of langchain
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})
