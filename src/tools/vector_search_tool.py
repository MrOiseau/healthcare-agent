"""
Implements the PatientRecordSemanticSearch tool for the agent, enabling two-stage
semantic retrieval (vector search + reranking) over patient records.
Returns a detailed structured log for downstream use and LLM answer grounding.

Used as a core agent tool for semantic queries.
"""

from langchain.tools import Tool
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from src.core.prompts import VECTOR_SEARCH_TOOL_DESCRIPTION
from typing import List
import json


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
        Invokes the retriever, reranks, and returns a structured JSON output.
        """
        print("\\n--- [Vector Search Tool] ---")
        print(f"Query: {query}")

        try:
            # --- Step 1: Initial retrieval ---
            initial_docs: List[Document] = retriever.invoke(query)
            print(f"\\n[1. Retrieval] Retrieved {len(initial_docs)} documents from vector store.")

            if not initial_docs:
                return json.dumps({"final_output": "No relevant patient records were found for this query."})

            # --- Step 2: Reranking ---
            print("\\n[2. Reranking] Applying cross-encoder to refine results...")
            reranked_docs: List[Document] = reranker.compress_documents(
                documents=initial_docs,
                query=query
            )
            print(f"   -> Reranked to {len(reranked_docs)} final documents.")

            if not reranked_docs:
                return json.dumps({"final_output": "No relevant patient records were found after reranking."})

            # --- Step 3: Format final output string for the LLM ---
            formatted_results = [
                f"Record (from row index {doc.metadata.get('row_index', 'N/A')}):\\n{doc.page_content}"
                for doc in reranked_docs
            ]
            final_output_str = "\\n\\n---\\n\\n".join(formatted_results)

            # --- Step 4: Structure the full output for UI display ---
            structured_output = {
                "tool_input": query,
                "retrieved_docs": [doc.model_dump() for doc in initial_docs],
                "reranked_docs": [doc.model_dump() for doc in reranked_docs],
                "final_output": final_output_str
            }
            print("--- [End Vector Search Tool] ---\\n")
            return json.dumps(structured_output)

        except Exception as e:
            error_message = f"Error during semantic search: {e}"
            print(error_message)
            return json.dumps({"error": error_message, "tool_input": query})

    return Tool(
        name="PatientRecordSemanticSearch",
        func=run_and_format_retriever,
        description=VECTOR_SEARCH_TOOL_DESCRIPTION
    )
