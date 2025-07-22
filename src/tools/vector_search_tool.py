from langchain.tools import Tool
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from src.core.prompts import VECTOR_SEARCH_TOOL_DESCRIPTION
from typing import List


def create_vector_search_tool(retriever: BaseRetriever, reranker: BaseDocumentCompressor) -> Tool:
    """
    Creates a tool for performing a two-stage semantic search with logging.

    This tool first retrieves an initial set of documents using the base retriever,
    logs them, then uses the reranker to refine the results before returning them.

    Args:
        retriever: The base retriever (e.g., from FAISS).
        reranker: The reranker component (e.g., CrossEncoderReranker).

    Returns:
        A robust LangChain tool for semantic retrieval with transparent logging.
    """
    def run_and_format_retriever(query: str) -> str:
        """
        Invokes the retriever, logs results, reranks, and formats the final output.
        """
        print("\\n--- [Vector Search Tool] ---")
        print(f"Query: {query}")

        try:
            # --- Step 1: Get INITIAL_K_RETRIEVED_DOCS dense vectors from FAISS-a ---
            initial_docs: List[Document] = retriever.invoke(query)
            print(f"\\n[1. Retrieval] Retrieved {len(initial_docs)} documents from vector store.")

            # Logging INITIAL_K_RETRIEVED_DOCS records
            for i, doc in enumerate(initial_docs):
                print(f"  - Initial Doc {i+1} (Row {doc.metadata.get('row_index', 'N/A')}): {doc.page_content}")

            if not initial_docs:
                return "No relevant patient records were found for this query."

            # --- Step 2: Reranking ---
            print("\\n[2. Reranking] Applying cross-encoder to refine results...")
            reranked_docs: List[Document] = reranker.compress_documents(
                documents=initial_docs,
                query=query
            )
            print(f"   -> Reranked to {len(reranked_docs)} final documents.")

            if not reranked_docs:
                return "No relevant patient records were found after reranking."

            # --- Step 3: Formating final output ---
            formatted_results = [
                f"Record (from row index {doc.metadata.get('row_index', 'N/A')}):\\n{doc.page_content}"
                for doc in reranked_docs
            ]
            final_output = "\\n\\n---\\n\\n".join(formatted_results)
            print("--- [End Vector Search Tool] ---\\n")
            return final_output

        except Exception as e:
            error_message = f"Error during semantic search: {e}"
            print(error_message)
            return error_message

    return Tool(
        name="PatientRecordSemanticSearch",
        func=run_and_format_retriever,
        description=VECTOR_SEARCH_TOOL_DESCRIPTION
    )
    