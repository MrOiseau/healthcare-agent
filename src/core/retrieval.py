import os
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from src.core import config


def create_retrieval_components() -> Tuple[BaseRetriever, CrossEncoderReranker]:
    """
    Creates and returns the base retriever and the reranker component separately.

    This allows for more granular control and logging between the retrieval and
    reranking steps.

    Returns:
        A tuple containing:
        - base_retriever (BaseRetriever): The FAISS retriever configured to fetch
          the initial set of documents.
        - reranker (CrossEncoderReranker): The cross-encoder reranker component.

    Raises:
        FileNotFoundError: If the FAISS index is not found at the specified path.
    """
    if not os.path.exists(config.VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{config.VECTOR_STORE_PATH}'. "
            "Please run the build script first: 'PYTHONPATH=$PYTHONPATH:. python src/scripts/01_build_vector_store.py'"
        )

    # Stage 1: Base retriever from the vector store
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        folder_path=config.VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.INITIAL_K_RETRIEVED_DOCS}
    )

    # Stage 2: Reranker model and compressor
    model = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=model, top_n=config.TOP_K_RERANKED_DOCS)

    return base_retriever, reranker
